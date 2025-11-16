import os
import time
import pickle
import fcntl
import boto3
import torch
import numpy as np
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from pathlib import Path
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader, get_worker_info

# ---- Redis (pip install redis) ----
try:
    import redis
except ImportError as e:
    raise ImportError("Please `pip install redis` to use the Redis-backed cache.") from e

# -------------------- Device & Env --------------------
device = torch.device('cuda' if torch.cuda.is_available()
                      else 'mps' if torch.backends.mps.is_available()
                      else 'cpu')
print(f"Using device: {device}")
load_dotenv('../config.env')

logger = 'pipeline_activity_resnet152_cached.log'


# ==================== Helper Logging ====================
def log_worker_fetch(worker_id, image_idx, fetch_time, source='S3'):
    try:
        with open(logger, 'a') as f:
            f.write(f"FETCH,{time.time():.3f},{worker_id},{image_idx},{fetch_time:.3f},{source}\n")
    except Exception:
        pass


def log_main_process_batch(main_id, batch_start_time, batch_size, image_indices):
    try:
        with open(logger, 'a') as f:
            f.write(f"PROCESS,{time.time():.3f},{main_id},{batch_size},{','.join(map(str, image_indices))}\n")
    except Exception:
        pass


def clear_pipeline_log():
    try:
        if os.path.exists(logger):
            os.remove(logger)
        with open(logger, 'w') as f:
            f.write("# Pipeline Activity Log - ResNet152 with RAM+SSD+Redis Caching\n")
            f.write("# FETCH: timestamp,worker_id,image_idx,fetch_time,source (RAM|SSD|REDIS|S3)\n")
            f.write("# PROCESS: timestamp,main_id,batch_size,indices\n")
    except Exception:
        pass


def get_epoch_stats_from_log(epoch_start_time, epoch_end_time):
    stats = {'ram_hits': 0, 'ssd_hits': 0, 'redis_hits': 0, 's3_fetches': 0}
    try:
        if not os.path.exists(logger):
            return stats
        with open(logger, 'r') as f:
            for line in f:
                if line.startswith('FETCH,'):
                    parts = line.strip().split(',')
                    if len(parts) >= 6:
                        ts = float(parts[1]); src = parts[5]
                        if epoch_start_time <= ts <= epoch_end_time:
                            if src == 'RAM':
                                stats['ram_hits'] += 1
                            elif src == 'SSD':
                                stats['ssd_hits'] += 1
                            elif src == 'REDIS':
                                stats['redis_hits'] += 1
                            elif src == 'S3':
                                stats['s3_fetches'] += 1
    except Exception as e:
        print(f"Warning: Could not parse log file for epoch stats: {e}")
    return stats


def save_metrics_log(metrics, cache_stats):
    try:
        total_ops = (cache_stats.get('ram_hits', 0) +
                     cache_stats.get('ssd_hits', 0) +
                     cache_stats.get('redis_hits', 0) +
                     cache_stats.get('s3_fetches', 0))
        if total_ops > 0:
            ram_pct = cache_stats['ram_hits'] / total_ops * 100
            ssd_pct = cache_stats['ssd_hits'] / total_ops * 100
            redis_pct = cache_stats['redis_hits'] / total_ops * 100
            s3_pct = cache_stats['s3_fetches'] / total_ops * 100
            cache_hit_rate = (cache_stats['ram_hits'] + cache_stats['ssd_hits'] + cache_stats['redis_hits']) / total_ops * 100
        else:
            ram_pct = ssd_pct = redis_pct = s3_pct = cache_hit_rate = 0.0

        with open('cached_S3_dataloader_resnet152_metrics.log', 'w') as f:
            f.write("# Cached S3 DataLoader Metrics - ResNet152 (RAM + SSD + Redis)\n")
            f.write("# Generated on: {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
            f.write("\n# Cache Performance (from log):\n")
            f.write(f"RAM Cache Hits:   {cache_stats.get('ram_hits', 0)} ({ram_pct:.1f}%)\n")
            f.write(f"SSD Cache Hits:   {cache_stats.get('ssd_hits', 0)} ({ssd_pct:.1f}%)\n")
            f.write(f"Redis Cache Hits: {cache_stats.get('redis_hits', 0)} ({redis_pct:.1f}%)\n")
            f.write(f"S3 Fetches:       {cache_stats.get('s3_fetches', 0)} ({s3_pct:.1f}%)\n")
            f.write(f"Total Cache Ops:  {total_ops}\n")
            f.write(f"Total Cache Hit Rate: {cache_hit_rate:.1f}%\n\n")

            f.write("# Performance Summary:\n")
            f.write(f"Total Data Requests: {metrics['total_s3_calls']}\n")
            f.write(f"Total Fetch Time: {metrics['total_fetch_time']:.2f}s\n")
            f.write(f"Total DataLoader Time: {metrics['total_dataloader_time']:.2f}s ({metrics['dataloader_percentage']:.1f}%)\n")
            f.write(f"Total GPU Time: {metrics['total_gpu_time']:.2f}s ({metrics['gpu_percentage']:.1f}%)\n")
            f.write(f"Total Training Time: {metrics['total_training_time']:.2f}s\n")
            f.write(f"Average Fetch Time per Image: {metrics['avg_fetch_time']:.3f}s\n")
            f.write(f"Data Requests per Second: {metrics['data_requests_per_second']:.1f}\n\n")

            f.write("# Analysis:\n")
            f.write(f"L1+L2+L3 eliminated ~{cache_hit_rate:.1f}% of S3 calls.\n")
    except Exception as e:
        print(f"Error saving metrics: {e}")


# ==================== Redis cleanup helper ====================
def cleanup_redis_keys(host, port, db, password, prefix, mode="scan"):
    """
    Clear Redis keys at the end of the run.

    mode="scan"    -> delete only keys starting with prefix (recommended)
    mode="flushdb" -> FLUSHDB (dangerous if the DB is shared)
    """
    r = redis.Redis(host=host, port=port, db=db, password=password, decode_responses=False)
    try:
        if mode.lower() == "flushdb":
            r.flushdb()
            print(f"Redis cleanup: FLUSHDB on db={db} complete.")
            return

        # Targeted cleanup via SCAN on '<prefix>*'
        if prefix.endswith(":"):
            pattern = prefix + "*"
        else:
            pattern = prefix + ":*"

        cursor = 0
        removed = 0
        pipe = r.pipeline(transaction=False)
        while True:
            cursor, keys = r.scan(cursor=cursor, match=pattern, count=1000)
            if keys:
                for k in keys:
                    pipe.unlink(k)
                pipe.execute()
                removed += len(keys)
            if cursor == 0:
                break
        print(f"Redis cleanup: removed {removed} keys with prefix pattern '{pattern}'.")
    except Exception as e:
        print(f"Redis cleanup warning: {e}")


# ==================== L1 (RAM) + L2 (SSD/worker) + L3 (Redis) Cache ====================
class RamSsdRedisCache:
    """
    Per-worker L1 RAM cache + per-worker L2 SSD cache + shared L3 Redis cache.

    Lookup order: RAM -> SSD -> Redis -> (caller falls back to S3)
    On S3 fetch: put() writes to Redis (L3), SSD (L2), and RAM (L1).

    SSD files contain pickled tuples: (torch_float32_CHW_tensor, int label)
    Redis entries store compact payload (uint8 CHW ndarray, label), pickled.
    """

    def __init__(
        self,
        ram_cache_size_mb: int,
        ssd_dir: str,
        redis_client: "redis.Redis",
        cache_stats_file: str = 'cache_stats.txt',
        redis_ttl_sec: int = 7 * 24 * 3600,
        redis_key_prefix: str = "cifar10:image:"
    ):
        # L1 (RAM)
        self.ram_cache_size_bytes = ram_cache_size_mb * 1024 * 1024
        self.ram_cache = OrderedDict()
        self.ram_cache_current_size = 0

        # L2 (SSD per worker)
        self.ssd_dir = Path(ssd_dir)
        self.ssd_dir.mkdir(parents=True, exist_ok=True)

        # L3 (Redis shared)
        self.r = redis_client
        self.ttl = int(redis_ttl_sec)
        self.key_prefix = redis_key_prefix

        # Stats
        self.cache_stats_file = Path(cache_stats_file)
        self._init_stats_file()

        self._local_stats = {
            'ram_hits': 0,
            'ssd_hits': 0,
            'redis_hits': 0,
            's3_fetches': 0,
            'ram_evictions': 0,
        }

    # ---------- Stats helpers ----------
    def _init_stats_file(self):
        if not self.cache_stats_file.exists():
            with open(self.cache_stats_file, 'w') as f:
                f.write("ram_hits:0\n")
                f.write("ssd_hits:0\n")
                f.write("redis_hits:0\n")
                f.write("s3_fetches:0\n")
                f.write("ram_evictions:0\n")

    def _update_global_stats(self, stat_name, increment=1):
        max_retries = 5
        retry_delay = 0.001
        for attempt in range(max_retries):
            try:
                with open(self.cache_stats_file, 'r+') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        stats = {}
                        f.seek(0)
                        for line in f:
                            if ':' in line:
                                k, v = line.strip().split(':', 1)
                                stats[k] = int(v)
                        stats[stat_name] = stats.get(stat_name, 0) + increment
                        f.seek(0)
                        f.truncate()
                        for k, v in stats.items():
                            f.write(f"{k}:{v}\n")
                        return
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except (IOError, OSError):
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    pass
            except Exception:
                pass

    # ---------- Keys & file paths ----------
    def _cache_key(self, idx: int) -> str:
        return f"image_{idx}"

    def _ssd_path(self, cache_key: str) -> Path:
        return self.ssd_dir / f"{cache_key}.pkl"

    def _redis_key(self, idx: int) -> str:
        return f"{self.key_prefix}{idx}"

    # ---------- Size in RAM ----------
    def _estimate_ram_size(self, data):
        # data: (torch.float32 CHW [0..1], label)
        image_tensor, label = data
        return image_tensor.element_size() * image_tensor.nelement() + 8

    # ---------- L1: RAM ----------
    def _put_ram(self, cache_key: str, data):
        data_size = self._estimate_ram_size(data)
        while (self.ram_cache_current_size + data_size > self.ram_cache_size_bytes) and len(self.ram_cache) > 0:
            evicted_key, evicted_data = self.ram_cache.popitem(last=False)
            evicted_size = self._estimate_ram_size(evicted_data)
            self.ram_cache_current_size -= evicted_size
            self._local_stats['ram_evictions'] += 1
            self._update_global_stats('ram_evictions')
        if cache_key in self.ram_cache:
            old_size = self._estimate_ram_size(self.ram_cache[cache_key])
            self.ram_cache_current_size -= old_size
        self.ram_cache[cache_key] = data
        self.ram_cache_current_size += data_size

    # ---------- L2: SSD ----------
    def _put_ssd(self, cache_key: str, data):
        path = self._ssd_path(cache_key)
        tmp = path.with_suffix('.tmp')
        try:
            with open(tmp, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(path)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    def _get_ssd(self, cache_key: str):
        path = self._ssd_path(cache_key)
        if not path.exists():
            return None
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)  # (tensor, label)
        except Exception:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            return None

    # ---------- L3: Redis ----------
    @staticmethod
    def _pack_for_redis(image_tensor: torch.Tensor, label: int) -> bytes:
        # CHW float32 [0..1] -> uint8 CHW ndarray; then pickle (ndarray, label)
        arr = (image_tensor.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
        return pickle.dumps((arr, int(label)), protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _unpack_from_redis(blob: bytes):
        arr, label = pickle.loads(blob)  # arr uint8 CHW
        t = torch.from_numpy(arr).to(torch.float32) / 255.0
        return t, int(label)

    # ---------- Public API ----------
    def get(self, idx: int):
        key = self._cache_key(idx)

        # L1 RAM
        if key in self.ram_cache:
            self.ram_cache.move_to_end(key)
            self._local_stats['ram_hits'] += 1
            self._update_global_stats('ram_hits')
            return self.ram_cache[key], 'RAM'

        # L2 SSD
        data = self._get_ssd(key)
        if data is not None:
            self._local_stats['ssd_hits'] += 1
            self._update_global_stats('ssd_hits')
            # promote to RAM
            self._put_ram(key, data)
            return data, 'SSD'

        # L3 Redis
        try:
            blob = self.r.get(self._redis_key(idx))
            if blob is not None:
                img_t, label = self._unpack_from_redis(blob)
                data = (img_t, label)
                # promote to SSD and RAM
                self._put_ssd(key, data)
                self._put_ram(key, data)
                self._local_stats['redis_hits'] += 1
                self._update_global_stats('redis_hits')
                return data, 'REDIS'
        except Exception:
            pass

        return None, 'MISS'

    def put(self, idx: int, image_tensor: torch.Tensor, label: int):
        key = self._cache_key(idx)
        data = (image_tensor, label)

        # L3 Redis
        try:
            payload = self._pack_for_redis(image_tensor, label)
            if self.ttl > 0:
                self.r.set(self._redis_key(idx), payload, ex=self.ttl)
            else:
                self.r.set(self._redis_key(idx), payload)
        except Exception:
            pass

        # L2 SSD
        self._put_ssd(key, data)

        # L1 RAM
        self._put_ram(key, data)

        self._local_stats['s3_fetches'] += 1
        self._update_global_stats('s3_fetches')

    def clear_ram(self):
        self.ram_cache.clear()
        self.ram_cache_current_size = 0


# ==================== Dataset ====================
class CachedS3CIFAR10Dataset(Dataset):
    """
    CIFAR-10 from S3 with per-worker L1 (RAM) + L2 (SSD) and shared L3 (Redis).
    Lookup: RAM -> SSD -> Redis -> S3
    On S3: writes into Redis, SSD, and RAM.
    """
    def __init__(
        self,
        bucket_name: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        num_images: int = 50000,
        region: str = 'us-east-1',
        ram_cache_size_mb: int = 1024,
        base_ssd_cache_dir: str = './ssd_cache',
        cache_stats_file: str = 'cache_stats.txt',
        redis_host: str = '127.0.0.1',
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: str = None,
        redis_ttl_sec: int = 7 * 24 * 3600,
        redis_key_prefix: str = "cifar10:image:",
    ):
        self.bucket_name = bucket_name
        self.num_images = num_images
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region = region

        self.ram_cache_size_mb = ram_cache_size_mb
        self.base_ssd_cache_dir = base_ssd_cache_dir
        self.cache_stats_file = cache_stats_file

        self.redis_host = redis_host
        self.redis_port = int(redis_port)
        self.redis_db = int(redis_db)
        self.redis_password = redis_password
        self.redis_ttl_sec = int(redis_ttl_sec)
        self.redis_key_prefix = redis_key_prefix

        # lazily created per-worker
        self._cache = None
        self._s3 = None
        self._r = None
        self._worker_pid = None
        self._worker_ssd_dir = None

        print(f"Initialized Redis-backed CACHED S3 dataset with {num_images} images")
        print("Lookup order: RAM (per-worker) -> SSD (per-worker) -> Redis (shared) -> S3")

    def _get_redis(self):
        if (self._r is None) or (self._worker_pid != os.getpid()):
            self._r = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                password=self.redis_password,
                decode_responses=False,
                socket_keepalive=True,
                socket_timeout=5,
            )
        return self._r

    def _get_ssd_dir_for_worker(self):
        info = get_worker_info()
        tag = f"w{info.id}" if info is not None else f"pid{os.getpid()}"
        worker_ssd_dir = Path(self.base_ssd_cache_dir) / tag
        worker_ssd_dir.mkdir(parents=True, exist_ok=True)
        return worker_ssd_dir

    def _get_cache(self):
        if (self._cache is None) or (self._worker_pid != os.getpid()):
            self._worker_pid = os.getpid()
            self._worker_ssd_dir = self._get_ssd_dir_for_worker()
            rclient = self._get_redis()
            self._cache = RamSsdRedisCache(
                ram_cache_size_mb=self.ram_cache_size_mb,
                ssd_dir=str(self._worker_ssd_dir),
                redis_client=rclient,
                cache_stats_file=self.cache_stats_file,
                redis_ttl_sec=self.redis_ttl_sec,
                redis_key_prefix=self.redis_key_prefix,
            )
        return self._cache

    def _get_s3(self):
        if (self._s3 is None) or (self._worker_pid != os.getpid()):
            self._s3 = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.region,
            )
        return self._s3

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx: int):
        start = time.time()
        cache = self._get_cache()

        # L1 -> L2 -> L3
        data, src = cache.get(idx)
        if data is not None:
            img_t, label = data
            fetch_time = time.time() - start
            log_worker_fetch(os.getpid(), idx, fetch_time, src)
            return img_t, label, fetch_time, idx

        # Miss -> S3
        s3 = self._get_s3()
        s3_key = f'cifar10/images/{idx + 1}.raw'
        try:
            resp = s3.get_object(Bucket=self.bucket_name, Key=s3_key)
            raw = resp['Body'].read()

            # 32x32x3 HWC uint8 -> CHW float32
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(32, 32, 3)
            arr = np.ascontiguousarray(arr)
            img_t = torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0

            try:
                label = int(resp['Metadata'].get('label', '0'))
            except Exception:
                np.random.seed(idx)
                label = int(np.random.randint(0, 10))

            # store to L3 (Redis), L2 (SSD), L1 (RAM)
            cache.put(idx, img_t.clone(), label)

            fetch_time = time.time() - start
            log_worker_fetch(os.getpid(), idx, fetch_time, 'S3')
            return img_t, label, fetch_time, idx

        except Exception as e:
            print(f"Error fetching image {idx + 1} from S3: {e}")
            return None, None, 0.0, -1


# ==================== Dataloader utils ====================
def collate_fn(batch):
    valid = [b for b in batch if b[0] is not None]
    if not valid:
        return None, None, 0.0, []
    imgs, labels, fetch_times, idxs = zip(*valid)
    return torch.stack(imgs), torch.tensor(labels, dtype=torch.long), float(np.mean(fetch_times)), list(idxs)


# ==================== Training ====================
def train_model(dataloader, cache_stats_file, num_epochs: int = 5, learning_rate: float = 0.001):
    print("Starting CACHED training with ResNet152 (RAM + SSD + Redis)")
    print("-----")
    model = models.resnet152(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    print(f"ResNet152 model moved to {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    clear_pipeline_log()
    main_pid = os.getpid()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    total_s3_calls = 0
    total_fetch_time = 0.0
    total_dataloader_time = 0.0
    total_gpu_time = 0.0
    total_training_time = 0.0

    def read_stats_file():
        stats = {'ram_hits': 0, 'ssd_hits': 0, 'redis_hits': 0, 's3_fetches': 0, 'ram_evictions': 0}
        try:
            p = Path(cache_stats_file)
            if p.exists():
                with open(p, 'r') as f:
                    for line in f:
                        if ':' in line:
                            k, v = line.strip().split(':', 1)
                            if k in stats:
                                stats[k] = int(v)
        except Exception:
            pass
        total = stats['ram_hits'] + stats['ssd_hits'] + stats['redis_hits'] + stats['s3_fetches']
        if total > 0:
            stats['ram_hit_rate'] = stats['ram_hits'] / total * 100
            stats['ssd_hit_rate'] = stats['ssd_hits'] / total * 100
            stats['redis_hit_rate'] = stats['redis_hits'] / total * 100
            stats['cache_hit_rate'] = (stats['ram_hits'] + stats['ssd_hits'] + stats['redis_hits']) / total * 100
            stats['s3_fetch_rate'] = stats['s3_fetches'] / total * 100
        else:
            stats['ram_hit_rate'] = stats['ssd_hit_rate'] = stats['redis_hit_rate'] = stats['cache_hit_rate'] = stats['s3_fetch_rate'] = 0.0
        return stats

    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_s3_calls = 0
        epoch_fetch_time = 0.0
        epoch_loss = 0.0
        batch_count = 0

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-----")

        dataset_size = len(dataloader.dataset)
        bsz = dataloader.batch_size
        expected_batches = (dataset_size + bsz - 1) // bsz
        print(f"Dataset size: {dataset_size}, Batch size: {bsz}, Expected batches: {expected_batches}")

        model.train()
        it = iter(dataloader)
        epoch_dl_time = 0.0

        while True:
            try:
                t0 = time.time()
                batch = next(it)
                dl_time = time.time() - t0
                epoch_dl_time += dl_time

                if batch is None:
                    print(f"Batch {batch_count} failed - skipping")
                    continue

                images, labels, avg_fetch_time, image_indices = batch
                log_main_process_batch(main_pid, time.time(), len(images), image_indices)

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                out = model(images)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

                n = len(images)
                epoch_s3_calls += n
                epoch_fetch_time += float(avg_fetch_time) * n
                epoch_loss += float(loss.item())
                batch_count += 1

                if batch_count % 10 == 0:
                    cs = read_stats_file()
                    print(f"Batch {batch_count:3d}/{expected_batches:3d}: "
                          f"Loss={loss.item():.4f}, DL={dl_time:.3f}s, "
                          f"Cache Hit Rate={cs['cache_hit_rate']:.1f}% "
                          f"(RAM {cs['ram_hit_rate']:.1f}%, SSD {cs['ssd_hit_rate']:.1f}%, Redis {cs['redis_hit_rate']:.1f}%)")

            except StopIteration:
                break

        epoch_time = time.time() - epoch_start
        epoch_gpu_time = epoch_time - epoch_dl_time

        total_s3_calls += epoch_s3_calls
        total_fetch_time += epoch_fetch_time
        total_dataloader_time += epoch_dl_time
        total_gpu_time += epoch_gpu_time
        total_training_time += epoch_time

        e_stats = get_epoch_stats_from_log(epoch_start, time.time())
        cs_all = read_stats_file()

        avg_loss = (epoch_loss / batch_count) if batch_count > 0 else 0.0
        avg_fetch_per_img = (epoch_fetch_time / epoch_s3_calls) if epoch_s3_calls > 0 else 0.0
        dl_pct = (epoch_dl_time / epoch_time * 100) if epoch_time > 0 else 0.0
        gpu_pct = (epoch_gpu_time / epoch_time * 100) if epoch_time > 0 else 0.0

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Batches: {batch_count}/{expected_batches}")
        print(f"  Data Requests (images): {epoch_s3_calls}")
        print(f"  Epoch Cache Stats (from log window):")
        print(f"    S3 Fetches:  {e_stats['s3_fetches']}")
        print(f"    Redis Hits:  {e_stats['redis_hits']}")
        print(f"    SSD Hits:    {e_stats['ssd_hits']}")
        print(f"    RAM Hits:    {e_stats['ram_hits']}")
        etotal = sum(e_stats.values())
        if etotal > 0:
            e_hit = (e_stats['ram_hits'] + e_stats['ssd_hits'] + e_stats['redis_hits']) / etotal * 100
            print(f"    Cache Hit Rate: {e_hit:.1f}%")
        print(f"  Avg Fetch/Image: {avg_fetch_per_img:.4f}s")
        print(f"  DataLoader Time: {epoch_dl_time:.2f}s ({dl_pct:.1f}%)")
        print(f"  GPU Time: {epoch_gpu_time:.2f}s ({gpu_pct:.1f}%)")
        print(f"  Total Time: {epoch_time:.2f}s")
        print(f"  Training Loss: {avg_loss:.4f}")

    final_stats = get_epoch_stats_from_log(0, time.time())
    total_ops = (final_stats['ram_hits'] + final_stats['ssd_hits'] +
                 final_stats['redis_hits'] + final_stats['s3_fetches'])
    overall_hit = ((final_stats['ram_hits'] + final_stats['ssd_hits'] + final_stats['redis_hits']) / total_ops * 100) if total_ops > 0 else 0.0

    print("\n----")
    print("Cached (RAM + SSD + Redis) ResNet152 training is complete")
    print("----")
    print("\nCache Performance (from log):")
    if total_ops > 0:
        print(f"  RAM Hits:   {final_stats['ram_hits']} ({final_stats['ram_hits']/total_ops*100:.1f}%)")
        print(f"  SSD Hits:   {final_stats['ssd_hits']} ({final_stats['ssd_hits']/total_ops*100:.1f}%)")
        print(f"  Redis Hits: {final_stats['redis_hits']} ({final_stats['redis_hits']/total_ops*100:.1f}%)")
        print(f"  S3 Fetches: {final_stats['s3_fetches']} ({final_stats['s3_fetches']/total_ops*100:.1f}%)")
    else:
        print("  No cache operations recorded.")
    print(f"  Overall Cache Hit Rate: {overall_hit:.1f}%")
    print(f"  Total Cache Operations: {total_ops}")

    print(f"\nTotal Data Requests (images): {total_s3_calls}")
    print(f"Total DataLoader Time: {total_dataloader_time:.2f}s ({(total_dataloader_time/total_training_time*100 if total_training_time>0 else 0.0):.1f}%)")
    print(f"Total GPU Time: {total_gpu_time:.2f}s ({(total_gpu_time/total_training_time*100 if total_training_time>0 else 0.0):.1f}%)")
    print(f"Total Training Time: {total_training_time:.2f}s")
    print(f"Average Fetch Time per Image: {(total_fetch_time/total_s3_calls if total_s3_calls>0 else 0.0):.3f}s")
    print(f"Data Requests per Second: {(total_s3_calls/total_training_time if total_training_time>0 else 0.0):.1f}")
    print(f"\nPipeline activity logged to {logger}")

    return {
        'total_s3_calls': total_s3_calls,
        'total_fetch_time': total_fetch_time,
        'total_dataloader_time': total_dataloader_time,
        'total_gpu_time': total_gpu_time,
        'total_training_time': total_training_time,
        'avg_fetch_time': (total_fetch_time / total_s3_calls) if total_s3_calls > 0 else 0.0,
        'dataloader_percentage': (total_dataloader_time / total_training_time * 100) if total_training_time > 0 else 0.0,
        'gpu_percentage': (total_gpu_time / total_training_time * 100) if total_training_time > 0 else 0.0,
        'data_requests_per_second': (total_s3_calls / total_training_time) if total_training_time > 0 else 0.0
    }, final_stats


# ==================== Main ====================
if __name__ == "__main__":
    BUCKET_NAME = os.getenv('BUCKET_NAME')
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

    NUM_IMAGES = int(os.getenv('NUM_IMAGES', 50000))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
    NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', 3))
    NUM_WORKERS = int(os.getenv('NUM_WORKERS', 4))

    # Cache settings
    RAM_CACHE_SIZE_MB = int(os.getenv('RAM_CACHE_SIZE_MB', 2048))
    BASE_SSD_CACHE_DIR = os.getenv('SSD_CACHE_DIR', './ssd_cache')
    CACHE_STATS_FILE = 'cache_stats.txt'
    CLEAR_CACHE_ON_START = os.getenv('CLEAR_CACHE_ON_START', 'false').lower() == 'true'

    # Redis settings
    REDIS_HOST = os.getenv('REDIS_HOST', '127.0.0.1')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD') or None
    REDIS_TTL_SEC = int(os.getenv('REDIS_TTL_SEC', 7 * 24 * 3600))
    REDIS_KEY_PREFIX = os.getenv('REDIS_KEY_PREFIX', 'cifar10:image:')
    REDIS_CLEANUP_MODE = os.getenv('REDIS_CLEANUP_MODE', 'scan')  # "scan" or "flushdb"

    print("CACHED: RAM (per-worker) + SSD (per-worker) + Redis (shared) with ResNet152")
    print("---- Configuration ----")
    print(f"  Dataset: {NUM_IMAGES} images")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Workers: {NUM_WORKERS}")
    print(f"  RAM Cache / worker: {RAM_CACHE_SIZE_MB} MB")
    print(f"  SSD Cache base dir: {BASE_SSD_CACHE_DIR}")
    print("  Redis:")
    print(f"    host={REDIS_HOST} port={REDIS_PORT} db={REDIS_DB} ttl={REDIS_TTL_SEC}s prefix='{REDIS_KEY_PREFIX}'")
    print(f"    cleanup_mode={REDIS_CLEANUP_MODE}")
    print("-----------------------")

    # Clear SSD cache if requested (base dir)
    base_ssd_path = Path(BASE_SSD_CACHE_DIR)
    if CLEAR_CACHE_ON_START and base_ssd_path.exists():
        import shutil
        shutil.rmtree(base_ssd_path, ignore_errors=True)
        print(f"Cleared SSD cache base dir: {BASE_SSD_CACHE_DIR}")
    base_ssd_path.mkdir(parents=True, exist_ok=True)

    # Initialize aggregated cache stats file
    p = Path(CACHE_STATS_FILE)
    if p.exists():
        p.unlink()
    with open(CACHE_STATS_FILE, 'w') as f:
        f.write("ram_hits:0\n")
        f.write("ssd_hits:0\n")
        f.write("redis_hits:0\n")
        f.write("s3_fetches:0\n")
        f.write("ram_evictions:0\n")
    print(f"Initialized cache statistics file: {CACHE_STATS_FILE}")

    # Dataset (per-worker cache is created lazily and SSD subdir per worker is used)
    dataset = CachedS3CIFAR10Dataset(
        bucket_name=BUCKET_NAME,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        num_images=NUM_IMAGES,
        ram_cache_size_mb=RAM_CACHE_SIZE_MB,
        base_ssd_cache_dir=BASE_SSD_CACHE_DIR,
        cache_stats_file=CACHE_STATS_FILE,
        redis_host=REDIS_HOST,
        redis_port=REDIS_PORT,
        redis_db=REDIS_DB,
        redis_password=REDIS_PASSWORD,
        redis_ttl_sec=REDIS_TTL_SEC,
        redis_key_prefix=REDIS_KEY_PREFIX,
    )

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

    print(f"DataLoader created with {NUM_WORKERS} workers and RAM+SSD+Redis caching enabled")

    metrics, cache_stats = train_model(dataloader, CACHE_STATS_FILE, num_epochs=NUM_EPOCHS)

    print("\nCached ResNet152 metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\nFinal Cache Statistics (from log):")
    for k, v in cache_stats.items():
        print(f"  {k}: {v}")

    save_metrics_log(metrics, cache_stats)
    print("\nMetrics saved to cached_S3_dataloader_resnet152_metrics.log")

    total_ops = (cache_stats.get('ram_hits', 0) +
                 cache_stats.get('ssd_hits', 0) +
                 cache_stats.get('redis_hits', 0) +
                 cache_stats.get('s3_fetches', 0))
    if total_ops > 0:
        hit_rate = (cache_stats['ram_hits'] + cache_stats['ssd_hits'] + cache_stats['redis_hits']) / total_ops * 100
        print(f"\nCache eliminated {hit_rate:.1f}% of S3 calls!")

    # --------- Final Redis cleanup (safe) ---------
    try:
        cleanup_redis_keys(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            prefix=REDIS_KEY_PREFIX,
            mode=REDIS_CLEANUP_MODE
        )
    except Exception as e:
        print(f"Final Redis cleanup failed: {e}")
