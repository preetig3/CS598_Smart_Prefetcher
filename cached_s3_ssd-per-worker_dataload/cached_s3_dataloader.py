import boto3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, get_worker_info  # ← added get_worker_info
import torchvision.models as models
import numpy as np
import time
import os
import hashlib
import pickle
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score
from collections import OrderedDict
import threading
import multiprocessing as mp
import fcntl  # For file locking

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# env vars (create a config.env)
load_dotenv('../config.env')

# made a logger to track activity for each worker, and main process
logger = 'pipeline_activity_resnet152_cached.log'


class TwoTierCache:
    """
    Two-tier caching system with RAM (L1) and SSD (L2) storage
    Process-safe implementation for PyTorch DataLoader multiprocessing
    - L1 Cache: Fast in-memory LRU cache (per-process, not shared)
    - L2 Cache: Persistent SSD storage (shared across processes via filesystem)
    """
    def __init__(self, ram_cache_size_mb=1024, ssd_cache_dir='./ssd_cache', cache_stats_file='cache_stats.txt'):
        """
        Initialize two-tier cache
        
        Args:
            ram_cache_size_mb: Size of RAM cache in MB (default 1GB)
            ssd_cache_dir: Directory for SSD cache storage
            cache_stats_file: File for tracking cache statistics
        """
        # L1: RAM cache (in-memory LRU) - per process, not shared
        self.ram_cache_size_bytes = ram_cache_size_mb * 1024 * 1024
        self.ram_cache = OrderedDict()
        self.ram_cache_current_size = 0
        
        # L2: SSD cache (persistent disk) - shared via filesystem
        self.ssd_cache_dir = Path(ssd_cache_dir)
        self.ssd_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache statistics file (shared across processes)
        self.cache_stats_file = Path(cache_stats_file)
        self._init_stats_file()
        
        # Process-local statistics (accumulated per process)
        self._local_stats = {
            'ram_hits': 0,
            'ssd_hits': 0,
            's3_fetches': 0,
            'ram_evictions': 0
        }
        
        self._pid = os.getpid()
        
    def _init_stats_file(self):
        """Initialize the shared statistics file"""
        if not self.cache_stats_file.exists():
            with open(self.cache_stats_file, 'w') as f:
                f.write("ram_hits:0\n")
                f.write("ssd_hits:0\n")
                f.write("s3_fetches:0\n")
                f.write("ram_evictions:0\n")
    
    def _update_global_stats(self, stat_name, increment=1):
        """Update global statistics file with file locking for atomicity"""
        max_retries = 5
        retry_delay = 0.001  # 1ms
        
        for attempt in range(max_retries):
            try:
                # Open file with exclusive lock
                with open(self.cache_stats_file, 'r+') as f:
                    # Acquire exclusive lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    
                    try:
                        # Read current stats
                        stats = {}
                        f.seek(0)
                        for line in f:
                            if ':' in line:
                                key, val = line.strip().split(':', 1)
                                stats[key] = int(val)
                        
                        # Update stat
                        stats[stat_name] = stats.get(stat_name, 0) + increment
                        
                        # Write back
                        f.seek(0)
                        f.truncate()
                        for key, val in stats.items():
                            f.write(f"{key}:{val}\n")
                        
                        # Release lock (automatic on close)
                        return  # Success!
                        
                    finally:
                        # Release lock
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        
            except (IOError, OSError):
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    # Silently fail after retries
                    pass
            except Exception:
                # Silently fail on other errors
                pass
    
    def _get_cache_key(self, idx):
        """Generate cache key for an image index"""
        return f"image_{idx}"
    
    def _get_ssd_path(self, cache_key):
        """Get SSD cache file path for a cache key"""
        return self.ssd_cache_dir / f"{cache_key}.pkl"
    
    def _estimate_size(self, data):
        """Estimate size of cached data in bytes"""
        image_tensor, label = data
        return image_tensor.element_size() * image_tensor.nelement() + 8
    
    def get(self, idx):
        """
        Retrieve data from cache (checks RAM -> SSD -> returns None)
        
        Args:
            idx: Image index
            
        Returns:
            (image_tensor, label) tuple if found, None otherwise
        """
        cache_key = self._get_cache_key(idx)
        
        # Check L1: RAM cache (process-local)
        if cache_key in self.ram_cache:
            # Move to end (most recently used)
            self.ram_cache.move_to_end(cache_key)
            self._local_stats['ram_hits'] += 1
            self._update_global_stats('ram_hits')
            return self.ram_cache[cache_key]
        
        # Check L2: SSD cache (shared via filesystem)
        ssd_path = self._get_ssd_path(cache_key)
        if ssd_path.exists():
            try:
                with open(ssd_path, 'rb') as f:
                    data = pickle.load(f)
                
                self._local_stats['ssd_hits'] += 1
                self._update_global_stats('ssd_hits')
                
                # Promote to RAM cache
                self._put_ram(cache_key, data)
                
                return data
            except Exception:
                # If corrupted, remove the file
                try:
                    ssd_path.unlink(missing_ok=True)
                except:
                    pass
        
        # Cache miss
        return None
    
    def put(self, idx, image_tensor, label):
        """
        Store data in cache (both RAM and SSD)
        
        Args:
            idx: Image index
            image_tensor: Image tensor
            label: Image label
        """
        cache_key = self._get_cache_key(idx)
        data = (image_tensor, label)
        
        # Store in both L1 (RAM) and L2 (SSD)
        self._put_ram(cache_key, data)
        self._put_ssd(cache_key, data)
        
        self._local_stats['s3_fetches'] += 1
        self._update_global_stats('s3_fetches')
    
    def _put_ram(self, cache_key, data):
        """Store data in RAM cache with LRU eviction"""
        data_size = self._estimate_size(data)
        
        # Evict old entries if necessary (LRU)
        while (self.ram_cache_current_size + data_size > self.ram_cache_size_bytes 
               and len(self.ram_cache) > 0):
            # Remove least recently used (first item)
            evicted_key, evicted_data = self.ram_cache.popitem(last=False)
            evicted_size = self._estimate_size(evicted_data)
            self.ram_cache_current_size -= evicted_size
            self._local_stats['ram_evictions'] += 1
            self._update_global_stats('ram_evictions')
        
        # Add new data
        if cache_key in self.ram_cache:
            # Update existing entry
            old_size = self._estimate_size(self.ram_cache[cache_key])
            self.ram_cache_current_size -= old_size
        
        self.ram_cache[cache_key] = data
        self.ram_cache_current_size += data_size
    
    def _put_ssd(self, cache_key, data):
        """Store data in SSD cache"""
        ssd_path = self._get_ssd_path(cache_key)
        try:
            # Write atomically using temp file
            temp_path = ssd_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            temp_path.replace(ssd_path)
        except Exception:
            # Clean up temp file on error
            try:
                temp_path.unlink(missing_ok=True)
            except:
                pass
    
    def get_stats(self):
        """Get cache statistics from shared file"""
        stats = {
            'ram_hits': 0,
            'ssd_hits': 0,
            's3_fetches': 0,
            'ram_evictions': 0
        }
        
        try:
            if self.cache_stats_file.exists():
                with open(self.cache_stats_file, 'r') as f:
                    for line in f:
                        if ':' in line:
                            key, val = line.strip().split(':', 1)
                            if key in stats:
                                stats[key] = int(val)
        except:
            pass
        
        total_requests = stats['ram_hits'] + stats['ssd_hits'] + stats['s3_fetches']
        if total_requests > 0:
            stats['ram_hit_rate'] = stats['ram_hits'] / total_requests * 100
            stats['ssd_hit_rate'] = stats['ssd_hits'] / total_requests * 100
            stats['cache_hit_rate'] = (stats['ram_hits'] + stats['ssd_hits']) / total_requests * 100
            stats['s3_fetch_rate'] = stats['s3_fetches'] / total_requests * 100
        else:
            stats['ram_hit_rate'] = 0
            stats['ssd_hit_rate'] = 0
            stats['cache_hit_rate'] = 0
            stats['s3_fetch_rate'] = 0
        
        return stats
    
    def clear_ram_cache(self):
        """Clear RAM cache (keeps SSD cache)"""
        self.ram_cache.clear()
        self.ram_cache_current_size = 0
    
    def clear_all(self):
        """Clear both RAM and SSD caches"""
        self.clear_ram_cache()
        
        # Clear SSD cache
        for cache_file in self.ssd_cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except:
                pass
        
        # Reset stats
        self._init_stats_file()
        
        print("All caches cleared")


class CachedS3CIFAR10Dataset(Dataset):
    """
    Cached S3 CIFAR10 Dataset with two-tier caching (RAM + SSD)
    - Checks RAM cache first (fast)
    - Falls back to SSD cache (medium speed, persistent)
    - Falls back to S3 (slow, but caches result)
    
    Cache is created per-worker to avoid pickling issues
    """
    def __init__(self, bucket_name: str, aws_access_key_id: str, aws_secret_access_key: str, 
                 num_images: int = 50000, region: str = 'us-east-1',
                 ram_cache_size_mb: int = 1024, ssd_cache_dir: str = './ssd_cache',
                 cache_stats_file: str = 'cache_stats.txt'):
        self.bucket_name = bucket_name
        self.num_images = num_images
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region = region
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Cache configuration (not the cache itself - created per worker)
        self.ram_cache_size_mb = ram_cache_size_mb
        self.ssd_cache_dir = ssd_cache_dir
        self.cache_stats_file = cache_stats_file
        
        # Per-worker instances (created lazily)
        self._cache = None
        self._s3_client = None
        # Track which process created the cache (DataLoader forks)
        self._worker_pid = None
        
        print(f"Initialized CACHED S3 dataset with {num_images} images")
        print("Cache hierarchy: RAM -> SSD -> S3")
    
    def _get_cache(self):
        """Get or create cache for this worker process (separate SSD dir per worker)."""
        info = get_worker_info()
        worker_tag = f"w{info.id}" if info is not None else f"pid{os.getpid()}"
        worker_ssd_dir = os.path.join(self.ssd_cache_dir, worker_tag)
        os.makedirs(worker_ssd_dir, exist_ok=True)

        # Use shared stats file (self.cache_stats_file) so training summaries aggregate across workers.
        # If you prefer per-worker stats, set worker_stats_file below and pass that instead.
        worker_stats_file = self.cache_stats_file  # or: os.path.join(worker_ssd_dir, "cache_stats.txt")

        current_pid = os.getpid()
        if self._cache is None or self._worker_pid != current_pid:
            self._worker_pid = current_pid
            self._cache = TwoTierCache(
                ram_cache_size_mb=self.ram_cache_size_mb,
                ssd_cache_dir=worker_ssd_dir,          # per-worker SSD cache dir
                cache_stats_file=worker_stats_file     # shared stats by default
            )
        return self._cache
    
    def _get_s3_client(self):
        """Get or create S3 client (threadsafe for multiprocessing)"""
        if self._s3_client is None:
            self._s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.region
            )
        return self._s3_client
    
    def __len__(self):
        return self.num_images
    
    def __getitem__(self, idx):
        """
        Fetch image with caching: RAM -> SSD -> S3
        """
        start_time = time.time()
        worker_id = os.getpid()
        
        cache_source = 'S3'  # Track where data came from
        
        # Get cache for this worker
        cache = self._get_cache()
        
        # Try to get from cache first (RAM -> SSD)
        cached_data = cache.get(idx)
        if cached_data is not None:
            image_tensor, label = cached_data
            fetch_time = time.time() - start_time
            
            # Determine cache source for logging
            # Simple heuristic: if fetch was very fast, it was RAM
            if fetch_time < 0.001:
                cache_source = 'RAM'
            else:
                cache_source = 'SSD'
            
            log_worker_fetch(worker_id, idx, fetch_time, cache_source)
            return image_tensor, label, fetch_time, idx
        
        # Cache miss - fetch from S3
        s3_client = self._get_s3_client()
        s3_key = f'cifar10/images/{idx + 1}.raw'
        
        try:
            # S3 GET call
            response = s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            # Read raw image data
            raw_data = response['Body'].read()
            
            # Convert to 32x32x3 numpy array
            image_array = np.frombuffer(raw_data, dtype=np.uint8).reshape(32, 32, 3)
            image_array = image_array.copy()
            
            # Tensor and normalize
            image_tensor = torch.from_numpy(image_array).float()
            image_tensor = image_tensor.permute(2, 0, 1) 
            image_tensor = image_tensor / 255.0 
            
            # Get label
            try:
                label = int(response['Metadata'].get('label', '0'))
            except (KeyError, ValueError):
                np.random.seed(idx)
                label = np.random.randint(0, 10)
            
            # Store in cache for future use
            cache.put(idx, image_tensor.clone(), label)
            
            fetch_time = time.time() - start_time
            log_worker_fetch(worker_id, idx, fetch_time, cache_source)
            
            return image_tensor, label, fetch_time, idx
            
        except Exception as e:
            print(f"Error fetching image {idx + 1} from S3: {e}")
            return None, None, 0, -1


def collate_fn(batch):
    """Custom collate function to handle the fetch_time data and indices"""
    valid_batch = [item for item in batch if item[0] is not None]
    
    if not valid_batch:
        return None, None, 0, []
    
    images, labels, fetch_times, indices = zip(*valid_batch)
    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    avg_fetch_time = np.mean(fetch_times)
    
    return images_tensor, labels_tensor, avg_fetch_time, list(indices)


def get_epoch_stats_from_log(epoch_start_time, epoch_end_time):
    """Calculate accurate per-epoch cache statistics from log file"""
    stats = {
        'ram_hits': 0,
        'ssd_hits': 0,
        's3_fetches': 0
    }
    
    try:
        if not os.path.exists(logger):
            return stats
            
        with open(logger, 'r') as f:
            for line in f:
                if line.startswith('FETCH,'):
                    parts = line.strip().split(',')
                    if len(parts) >= 6:
                        timestamp = float(parts[1])
                        source = parts[5]
                        
                        # Check if this fetch happened during this epoch
                        if epoch_start_time <= timestamp <= epoch_end_time:
                            if source == 'RAM':
                                stats['ram_hits'] += 1
                            elif source == 'SSD':
                                stats['ssd_hits'] += 1
                            elif source == 'S3':
                                stats['s3_fetches'] += 1
    except Exception as e:
        print(f"Warning: Could not parse log file for epoch stats: {e}")
    
    return stats


def log_worker_fetch(worker_id, image_idx, fetch_time, source='S3'):
    """Log when a worker fetches an image"""
    try:
        with open(logger, 'a') as f:
            f.write(f"FETCH,{time.time():.3f},{worker_id},{image_idx},{fetch_time:.3f},{source}\n")
    except:
        pass


def log_main_process_batch(main_id, batch_start_time, batch_size, image_indices):
    """Log when main process starts processing a batch"""
    try:
        with open(logger, 'a') as f:
            f.write(f"PROCESS,{time.time():.3f},{main_id},{batch_size},{','.join(map(str, image_indices))}\n")
    except:
        pass


def clear_pipeline_log():
    """Clear the pipeline log for fresh monitoring"""
    try:
        if os.path.exists(logger):
            os.remove(logger)
        
        with open(logger, 'w') as f:
            f.write("# Pipeline Activity Log - ResNet152 with Caching\n")
            f.write("# Format: ACTIVITY_TYPE,TIMESTAMP,PROCESS_ID,DATA\n")
            f.write("# FETCH: Worker fetches image (timestamp,worker_id,image_idx,fetch_time,source)\n")
            f.write("#   source: RAM, SSD, or S3\n")
            f.write("# PROCESS: Main process processes batch (timestamp,main_id,batch_size,image_indices)\n")
            f.write("#\n")
    except:
        pass


def save_metrics_log(metrics, cache_stats):
    """Save metrics to cached_S3_dataloader_resnet152_metrics.log"""
    try:
        # Calculate percentages from cache_stats
        total_ops = cache_stats['ram_hits'] + cache_stats['ssd_hits'] + cache_stats['s3_fetches']
        if total_ops > 0:
            ram_pct = cache_stats['ram_hits'] / total_ops * 100
            ssd_pct = cache_stats['ssd_hits'] / total_ops * 100
            s3_pct = cache_stats['s3_fetches'] / total_ops * 100
            cache_hit_rate = (cache_stats['ram_hits'] + cache_stats['ssd_hits']) / total_ops * 100
        else:
            ram_pct = ssd_pct = s3_pct = cache_hit_rate = 0
        
        with open('cached_S3_dataloader_resnet152_metrics.log', 'w') as f:
            f.write("# Cached S3 DataLoader Metrics - ResNet152\n")
            f.write("# Generated on: {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
            f.write("#\n")
            f.write("# Cache Performance (from log file):\n")
            f.write(f"RAM Cache Hits: {cache_stats['ram_hits']} ({ram_pct:.1f}%)\n")
            f.write(f"SSD Cache Hits: {cache_stats['ssd_hits']} ({ssd_pct:.1f}%)\n")
            f.write(f"S3 Fetches: {cache_stats['s3_fetches']} ({s3_pct:.1f}%)\n")
            f.write(f"Total Cache Operations: {total_ops}\n")
            f.write(f"Total Cache Hit Rate: {cache_hit_rate:.1f}%\n")
            f.write("\n")
            f.write("# Performance Summary:\n")
            f.write(f"Total Data Requests: {metrics['total_s3_calls']}\n")
            f.write(f"Total Fetch Time: {metrics['total_fetch_time']:.2f}s\n")
            f.write(f"Total DataLoader Time: {metrics['total_dataloader_time']:.2f}s ({metrics['dataloader_percentage']:.1f}%)\n")
            f.write(f"Total GPU Time: {metrics['total_gpu_time']:.2f}s ({metrics['gpu_percentage']:.1f}%)\n")
            f.write(f"Total Training Time: {metrics['total_training_time']:.2f}s\n")
            f.write(f"Average Fetch Time per Image: {metrics['avg_fetch_time']:.3f}s\n")
            f.write(f"Data Requests per Second: {metrics['data_requests_per_second']:.1f}\n")
            f.write("\n")
            f.write("# Analysis:\n")
            f.write(f"Cache eliminated {cache_hit_rate:.1f}% of S3 calls\n")
            f.write(f"DataLoader Time: {metrics['dataloader_percentage']:.1f}% (waiting for data)\n")
            f.write(f"GPU Time: {metrics['gpu_percentage']:.1f}% (actual compute)\n")
    
    except Exception as e:
        print(f"Error saving metrics: {e}")


def train_model(dataloader, cache_stats_file, num_epochs: int = 5, learning_rate: float = 0.001):
    """Train the model with two-tier caching and ResNet152"""
    print("Starting CACHED training with ResNet152 (RAM + SSD caching)")
    print("-----")
    
    # Load ResNet152 model
    model = models.resnet152(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    print(f"ResNet152 model moved to {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    clear_pipeline_log()
    main_process_id = os.getpid()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training metrics
    total_s3_calls = 0
    total_fetch_time = 0.0
    total_dataloader_time = 0.0
    total_gpu_time = 0.0
    total_training_time = 0.0
    epoch_times = []
    
    # Helper to read cache stats from file
    def get_cache_stats():
        stats = {
            'ram_hits': 0,
            'ssd_hits': 0,
            's3_fetches': 0,
            'ram_evictions': 0
        }
        try:
            if Path(cache_stats_file).exists():
                with open(cache_stats_file, 'r') as f:
                    for line in f:
                        if ':' in line:
                            key, val = line.strip().split(':', 1)
                            if key in stats:
                                stats[key] = int(val)
        except:
            pass
        
        total_requests = stats['ram_hits'] + stats['ssd_hits'] + stats['s3_fetches']
        if total_requests > 0:
            stats['ram_hit_rate'] = stats['ram_hits'] / total_requests * 100
            stats['ssd_hit_rate'] = stats['ssd_hits'] / total_requests * 100
            stats['cache_hit_rate'] = (stats['ram_hits'] + stats['ssd_hits']) / total_requests * 100
            stats['s3_fetch_rate'] = stats['s3_fetches'] / total_requests * 100
        else:
            stats['ram_hit_rate'] = 0
            stats['ssd_hit_rate'] = 0
            stats['cache_hit_rate'] = 0
            stats['s3_fetch_rate'] = 0
        
        return stats
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_s3_calls = 0
        epoch_fetch_time = 0.0
        epoch_loss = 0.0
        batch_count = 0
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-----")
        
        # Calculate expected number of batches
        dataset_size = len(dataloader.dataset)
        batch_size = dataloader.batch_size
        expected_batches = (dataset_size + batch_size - 1) // batch_size
        print(f"Dataset size: {dataset_size}, Batch size: {batch_size}, Expected batches: {expected_batches}")
        
        model.train()
        dataloader_iter = iter(dataloader)
        epoch_dataloader_time = 0.0
        
        while True:
            try:
                dataloader_start = time.time()
                batch_data = next(dataloader_iter)
                dataloader_time = time.time() - dataloader_start
                epoch_dataloader_time += dataloader_time
                
                if batch_data is None:
                    print(f"Batch {batch_count} failed - skipping")
                    continue
                    
                images, labels, avg_fetch_time, image_indices = batch_data
                
                batch_size = len(images)
                log_main_process_batch(main_process_id, time.time(), batch_size, image_indices)
                
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_s3_calls += len(images)
                epoch_fetch_time += avg_fetch_time * len(images)
                epoch_loss += loss.item()
                batch_count += 1
                
                if batch_count % 10 == 0:
                    # Show cache stats
                    current_stats = get_cache_stats()
                    print(f"Batch {batch_count:3d}/{expected_batches:3d}: Loss={loss.item():.4f}, "
                          f"DataLoader={dataloader_time:.3f}s, "
                          f"Cache Hit Rate={current_stats['cache_hit_rate']:.1f}%")
                          
            except StopIteration:
                break
        
        epoch_time = time.time() - epoch_start
        epoch_gpu_time = epoch_time - epoch_dataloader_time
        
        total_s3_calls += epoch_s3_calls
        total_fetch_time += epoch_fetch_time
        total_dataloader_time += epoch_dataloader_time
        total_gpu_time += epoch_gpu_time
        total_training_time += epoch_time
        epoch_times.append(epoch_time)
        
        # Get accurate per-epoch cache stats from log file
        epoch_cache_stats = get_epoch_stats_from_log(epoch_start, time.time())
        
        # Get overall cache stats
        cache_stats_end = get_cache_stats()
        epoch_cache_hit_rate = cache_stats_end['cache_hit_rate']
        
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
        else:
            avg_loss = 0.0
            
        if epoch_s3_calls > 0:
            avg_fetch_per_image = epoch_fetch_time / epoch_s3_calls
        else:
            avg_fetch_per_image = 0.0
            
        dataloader_percentage = epoch_dataloader_time / epoch_time * 100 if epoch_time > 0 else 0.0
        gpu_percentage = epoch_gpu_time / epoch_time * 100 if epoch_time > 0 else 0.0
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Batches Processed: {batch_count}/{expected_batches}")
        print(f"  Data Requests: {epoch_s3_calls}")
        print(f"  Epoch Cache Stats (from log):")
        print(f"    S3 Fetches: {epoch_cache_stats['s3_fetches']}")
        print(f"    SSD Hits: {epoch_cache_stats['ssd_hits']}")
        print(f"    RAM Hits: {epoch_cache_stats['ram_hits']}")
        epoch_total = sum(epoch_cache_stats.values())
        if epoch_total > 0:
            epoch_cache_pct = (epoch_cache_stats['ssd_hits'] + epoch_cache_stats['ram_hits']) / epoch_total * 100
            print(f"    Cache Hit Rate: {epoch_cache_pct:.1f}%")
        print(f"  DataLoader Time: {epoch_dataloader_time:.2f}s ({dataloader_percentage:.1f}%)")
        print(f"  GPU Time: {epoch_gpu_time:.2f}s ({gpu_percentage:.1f}%)")
        print(f"  Total Time: {epoch_time:.2f}s")
        print(f"  Training Loss: {avg_loss:.4f}")
    
    # Final summary - calculate from log file for accuracy
    final_cache_stats = get_epoch_stats_from_log(0, time.time())  # All time
    
    print("\n")
    print("----")
    print("Cached ResNet152 training is complete")
    print("----")
    print(f"\nCache Performance (from log):")
    print(f"  RAM Hits: {final_cache_stats['ram_hits']} ({final_cache_stats['ram_hits']/(final_cache_stats['ram_hits']+final_cache_stats['ssd_hits']+final_cache_stats['s3_fetches'])*100:.1f}%)")
    print(f"  SSD Hits: {final_cache_stats['ssd_hits']} ({final_cache_stats['ssd_hits']/(final_cache_stats['ram_hits']+final_cache_stats['ssd_hits']+final_cache_stats['s3_fetches'])*100:.1f}%)")
    print(f"  S3 Fetches: {final_cache_stats['s3_fetches']} ({final_cache_stats['s3_fetches']/(final_cache_stats['ram_hits']+final_cache_stats['ssd_hits']+final_cache_stats['s3_fetches'])*100:.1f}%)")
    total_cache_ops = final_cache_stats['ram_hits'] + final_cache_stats['ssd_hits'] + final_cache_stats['s3_fetches']
    cache_hit_rate = (final_cache_stats['ram_hits'] + final_cache_stats['ssd_hits']) / total_cache_ops * 100 if total_cache_ops > 0 else 0
    print(f"  Overall Cache Hit Rate: {cache_hit_rate:.1f}%")
    print(f"  Total Cache Operations: {total_cache_ops}")
    
    print(f"\nTotal Data Requests: {total_s3_calls}")
    print(f"Total DataLoader Time: {total_dataloader_time:.2f}s ({total_dataloader_time/total_training_time*100:.1f}%)")
    print(f"Total GPU Time: {total_gpu_time:.2f}s ({total_gpu_time/total_training_time*100:.1f}%)")
    print(f"Total Training Time: {total_training_time:.2f}s")
    print(f"Average Fetch Time per Image: {total_fetch_time/total_s3_calls:.3f}s")
    print(f"Data Requests per Second: {total_s3_calls/total_training_time:.1f}")
    
    print(f"\nTime Breakdown:")
    print(f"  DataLoader Time: {total_dataloader_time/total_training_time*100:.1f}% (waiting for data)")
    print(f"  GPU Time: {total_gpu_time/total_training_time*100:.1f}% (actual compute)")
    
    print(f"\nPipeline activity logged to {logger}")
    
    return {
        'total_s3_calls': total_s3_calls,
        'total_fetch_time': total_fetch_time,
        'total_dataloader_time': total_dataloader_time,
        'total_gpu_time': total_gpu_time,
        'total_training_time': total_training_time,
        'avg_fetch_time': total_fetch_time / total_s3_calls,
        'dataloader_percentage': total_dataloader_time / total_training_time * 100,
        'gpu_percentage': total_gpu_time / total_training_time * 100,
        'data_requests_per_second': total_s3_calls / total_training_time
    }, final_cache_stats


if __name__ == "__main__":
    BUCKET_NAME = os.getenv('BUCKET_NAME')
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    NUM_IMAGES = int(os.getenv('NUM_IMAGES', 50000))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
    NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', 3))
    NUM_WORKERS = int(os.getenv('NUM_WORKERS', 4))
    
    # Cache configuration
    RAM_CACHE_SIZE_MB = int(os.getenv('RAM_CACHE_SIZE_MB', 2048))  # 2GB default
    SSD_CACHE_DIR = os.getenv('SSD_CACHE_DIR', './ssd_cache')
    CACHE_STATS_FILE = 'cache_stats.txt'
    CLEAR_CACHE_ON_START = os.getenv('CLEAR_CACHE_ON_START', 'false').lower() == 'true'
    
    print("CACHED: Two-Tier Caching (RAM + SSD) with ResNet152")
    print("This improves on baseline by caching frequently accessed data")
    print("----")
    print(f"Configuration:")
    print(f"  Dataset: {NUM_IMAGES} images")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Workers: {NUM_WORKERS}")
    print(f"  Expected Batches per Epoch: {(NUM_IMAGES + BATCH_SIZE - 1) // BATCH_SIZE}")
    print(f"----")
    print(f"Cache Settings:")
    print(f"  RAM Cache: {RAM_CACHE_SIZE_MB} MB per worker")
    print(f"  SSD Cache: {SSD_CACHE_DIR}")
    print(f"  Clear Cache on Start: {CLEAR_CACHE_ON_START}")
    
    # Clear caches if requested
    if CLEAR_CACHE_ON_START:
        print("\nClearing existing caches...")
        ssd_cache_path = Path(SSD_CACHE_DIR)
        if ssd_cache_path.exists():
            import shutil
            shutil.rmtree(ssd_cache_path)
            print(f"  Deleted SSD cache directory: {SSD_CACHE_DIR}")
        ssd_cache_path.mkdir(parents=True, exist_ok=True)
        print("  Caches cleared!")
    else:
        print("\nWARNING: SSD cache from previous runs may exist!")
        ssd_cache_path = Path(SSD_CACHE_DIR)
        if ssd_cache_path.exists():
            cache_files = list(ssd_cache_path.glob("*.pkl"))
            if cache_files:
                print(f"  Found {len(cache_files)} cached files in {SSD_CACHE_DIR}")
                print(f"  Set CLEAR_CACHE_ON_START=true in config.env to start fresh")
        else:
            ssd_cache_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize cache stats file (shared across workers for aggregated metrics)
    cache_stats_path = Path(CACHE_STATS_FILE)
    if cache_stats_path.exists():
        cache_stats_path.unlink()
    with open(CACHE_STATS_FILE, 'w') as f:
        f.write("ram_hits:0\n")
        f.write("ssd_hits:0\n")
        f.write("s3_fetches:0\n")
        f.write("ram_evictions:0\n")
    print(f"\nInitialized cache statistics file: {CACHE_STATS_FILE}")
    
    # Create dataset with cache configuration (cache created per worker)
    dataset = CachedS3CIFAR10Dataset(
        bucket_name=BUCKET_NAME,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        num_images=NUM_IMAGES,
        ram_cache_size_mb=RAM_CACHE_SIZE_MB,
        ssd_cache_dir=SSD_CACHE_DIR,
        cache_stats_file=CACHE_STATS_FILE
    )

    # Configure dataloader
    pin_memory = device.type == 'cuda'
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=pin_memory, 
        persistent_workers=True  
    )
    
    print(f"DataLoader created with {NUM_WORKERS} workers and caching enabled")
    
    metrics, cache_stats = train_model(dataloader, CACHE_STATS_FILE, num_epochs=NUM_EPOCHS)
    
    print(f"\nCached ResNet152 metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print(f"\nFinal Cache Statistics (from log):")
    for key, value in cache_stats.items():
        print(f"  {key}: {value}")
    
    # Save metrics to log file
    save_metrics_log(metrics, cache_stats)
    print(f"\nMetrics saved to cached_S3_dataloader_resnet152_metrics.log")
    
    print(f"\nThis version uses two-tier caching (RAM + SSD) to reduce S3 calls")
    total_ops = cache_stats['ram_hits'] + cache_stats['ssd_hits'] + cache_stats['s3_fetches']
    if total_ops > 0:
        hit_rate = (cache_stats['ram_hits'] + cache_stats['ssd_hits']) / total_ops * 100
        print(f"Cache eliminated {hit_rate:.1f}% of S3 calls!")
