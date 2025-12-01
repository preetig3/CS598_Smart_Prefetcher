#!/usr/bin/env python3
import os
import time
import socket
import pickle
import threading
import struct

from pathlib import Path
from collections import OrderedDict, deque

import boto3
import redis
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv

load_dotenv('./config.env')

device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu'
)
print(f"[INFO] Using device: {device}")

LOG_FILE = "pipeline_activity_rdma.log"



# LOGGING HELPERS
def log_fetch(worker_id, idx, fetch_time, source):
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(f"FETCH,{time.time():.3f},{worker_id},{idx},{fetch_time:.4f},{source}\n")
    except Exception:
        pass


def log_batch(main_pid, batch_indices):
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(f"PROCESS,{time.time():.3f},{main_pid},{len(batch_indices)},{batch_indices}\n")
    except Exception:
        pass


def reset_log():
    with open(LOG_FILE, 'w') as f:
        f.write("# FETCH and PROCESS log\n")


def get_epoch_stats_from_log(epoch_start_time, epoch_end_time):
    """Parse log file for epoch stats"""
    stats = {'ram_hits': 0, 'ssd_hits': 0, 'rdma_hits': 0, 's3_fetches': 0, 's3_fallbacks': 0}
    try:
        if not os.path.exists(LOG_FILE):
            return stats
        with open(LOG_FILE, 'r') as f:
            for line in f:
                if line.startswith('FETCH,'):
                    parts = line.strip().split(',')
                    if len(parts) >= 6:
                        ts = float(parts[1])
                        src = parts[5]
                        if epoch_start_time <= ts <= epoch_end_time:
                            if src == 'RAM':
                                stats['ram_hits'] += 1
                            elif src == 'SSD':
                                stats['ssd_hits'] += 1
                            elif src == 'RDMA':
                                stats['rdma_hits'] += 1
                            elif src == 'S3':
                                stats['s3_fetches'] += 1
                            elif src == 'S3_FALLBACK':
                                stats['s3_fallbacks'] += 1
    except Exception as e:
        print(f"Warning: Could not parse log file for epoch stats: {e}")
    return stats


def save_metrics_log(metrics, cache_stats):
    """Save metrics to log file"""
    try:
        total_ops = (cache_stats.get('ram_hits', 0) +
                     cache_stats.get('ssd_hits', 0) +
                     cache_stats.get('rdma_hits', 0) +
                     cache_stats.get('s3_fetches', 0) +
                     cache_stats.get('s3_fallbacks', 0))
        if total_ops > 0:
            ram_pct = cache_stats['ram_hits'] / total_ops * 100
            ssd_pct = cache_stats['ssd_hits'] / total_ops * 100
            rdma_pct = cache_stats['rdma_hits'] / total_ops * 100
            s3_pct = cache_stats['s3_fetches'] / total_ops * 100
            s3_fallback_pct = cache_stats['s3_fallbacks'] / total_ops * 100
            cache_hit_rate = (cache_stats['ram_hits'] + cache_stats['ssd_hits'] + cache_stats['rdma_hits']) / total_ops * 100
        else:
            ram_pct = ssd_pct = rdma_pct = s3_pct = s3_fallback_pct = cache_hit_rate = 0.0

        with open('cached_rdma_dataloader_resnet152_metrics.log', 'w') as f:
            f.write("# Cached S3 DataLoader Metrics - ResNet152 (RAM + SSD + RDMA)\n")
            f.write("# Generated on: {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
            f.write("\n# Cache Performance (from log):\n")
            f.write(f"RAM Cache Hits:      {cache_stats.get('ram_hits', 0)} ({ram_pct:.1f}%)\n")
            f.write(f"SSD Cache Hits:      {cache_stats.get('ssd_hits', 0)} ({ssd_pct:.1f}%)\n")
            f.write(f"RDMA Hits:           {cache_stats.get('rdma_hits', 0)} ({rdma_pct:.1f}%)\n")
            f.write(f"S3 Fetches (Worker): {cache_stats.get('s3_fetches', 0)} ({s3_pct:.1f}%)\n")
            f.write(f"S3 Fallbacks:        {cache_stats.get('s3_fallbacks', 0)} ({s3_fallback_pct:.1f}%)\n")
            f.write(f"Total Cache Ops:     {total_ops}\n")
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
            f.write(f"L1+L2+RDMA eliminated ~{cache_hit_rate:.1f}% of S3 calls.\n")
    except Exception as e:
        print(f"Error saving metrics: {e}")


# REDIS OWNER REGISTRY
class OwnerRegistry:
    """
    Redis:
      key:   prefix:owner:<idx>
      value: "ip:port"
    Only workers write these keys.
    Trainer only GETs.
    """
    def __init__(self, host, port, db, pw, prefix, owner_ttl_sec=600):
        self.r = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=pw,
            decode_responses=True,
            socket_timeout=3,
        )
        self.prefix = prefix if prefix.endswith(":") else prefix + ":"
        self.owner_ttl_sec = int(owner_ttl_sec)

    def _key(self, idx):
        return f"{self.prefix}owner:{idx}"

    def set_owner(self, idx, endpoint):
        try:
            # TTL so stale owners disappear automatically
            self.r.set(self._key(idx), endpoint, ex=self.owner_ttl_sec)
        except Exception:
            pass

    def get_owner(self, idx):
        try:
            return self.r.get(self._key(idx))
        except Exception:
            return None

    def clear_owner(self, idx):
        try:
            self.r.delete(self._key(idx))
        except Exception:
            pass



# ==============================================================================
# RDMA COMMUNICATION LAYER (Simulated over TCP sockets)
# ==============================================================================
# Note: True RDMA requires InfiniBand/RoCE hardware and libraries like pyverbs.
# For portability, we simulate RDMA semantics over TCP with zero-copy-like behavior.

class RDMAServer:
    """
    RDMA-style server for workers
    Listens on a TCP socket and serves data requests with minimal overhead
    """
    def __init__(self, host, port, get_data_callback):
        """
        Args:
            host: Bind address
            port: Port to listen on (0 = auto-assign)
            get_data_callback: Function(idx) -> (image_tensor, label) or None
        """
        self.host = host
        self.port = port
        self.get_data = get_data_callback
        self.server_socket = None
        self.running = False

        self._start_server()

    def _start_server(self):
        """Start the RDMA server in background thread"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))

        # Get actual port if auto-assigned
        self.port = self.server_socket.getsockname()[1]
        self.endpoint = f"{self.host}:{self.port}"

        self.server_socket.listen(128)  # High backlog for concurrent connections
        self.running = True

        print(f"[RDMA Server] Listening on {self.endpoint}")

        # Start accept loop in daemon thread
        t = threading.Thread(target=self._accept_loop, daemon=True)
        t.start()

    def _accept_loop(self):
        """Accept incoming connections"""
        while self.running:
            try:
                conn, addr = self.server_socket.accept()
                # Handle each connection in separate thread
                t = threading.Thread(target=self._handle_connection, args=(conn, addr), daemon=True)
                t.start()
            except Exception as e:
                if self.running:
                    print(f"[RDMA Server] Accept error: {e}")

    def _handle_connection(self, conn, addr):
        """
        Handle a single RDMA request
        Protocol:
          1. Client sends: 4 bytes (message length) + request data (pickled idx)
          2. Server responds: 4 bytes (response length) + response data (pickled (image, label) or None)
        """
        try:
            # Read request length (4 bytes)
            length_data = self._recv_exact(conn, 4)
            if not length_data:
                return

            req_len = struct.unpack('!I', length_data)[0]

            # Read request data
            req_data = self._recv_exact(conn, req_len)
            if not req_data:
                return

            idx = pickle.loads(req_data)

            # Get data from cache
            data = self.get_data(idx)

            # Serialize response
            if data is not None:
                img_t, label = data
                # Convert to uint8 numpy for efficient transfer
                arr = (img_t * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                response = pickle.dumps((arr, int(label)), protocol=pickle.HIGHEST_PROTOCOL)
            else:
                response = pickle.dumps(None, protocol=pickle.HIGHEST_PROTOCOL)

            # Send response length + response
            resp_len = len(response)
            conn.sendall(struct.pack('!I', resp_len))
            conn.sendall(response)

        except Exception as e:
            pass  # Silently handle errors
        finally:
            conn.close()

    def _recv_exact(self, conn, n):
        """Receive exactly n bytes"""
        data = b''
        while len(data) < n:
            chunk = conn.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def shutdown(self):
        """Shutdown the server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()


class RDMAClient:
    """
    RDMA-style client for trainer
    Makes high-performance data requests to worker RDMA servers
    """
    def __init__(self, timeout=3.0):
        self.timeout = timeout

    def fetch(self, endpoint, idx):
        """
        Fetch image from worker via RDMA

        Args:
            endpoint: "host:port" of worker RDMA server
            idx: Image index to fetch

        Returns:
            (image_tensor, label) or (None, None) on failure
        """
        try:
            host, port = endpoint.split(':')
            port = int(port)

            # Create connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((host, port))

            # Send request
            req_data = pickle.dumps(idx, protocol=pickle.HIGHEST_PROTOCOL)
            req_len = len(req_data)
            sock.sendall(struct.pack('!I', req_len))
            sock.sendall(req_data)

            # Receive response length
            length_data = self._recv_exact(sock, 4)
            if not length_data:
                sock.close()
                return None, None

            resp_len = struct.unpack('!I', length_data)[0]

            # Receive response data
            resp_data = self._recv_exact(sock, resp_len)
            sock.close()

            if not resp_data:
                return None, None

            # Deserialize
            result = pickle.loads(resp_data)
            if result is None:
                return None, None

            arr, label = result
            # Convert back to float tensor
            img_t = torch.from_numpy(arr).float() / 255.0

            return img_t, int(label)

        except Exception:
            # Silently fail - caller will handle retry/fallback
            return None, None

    def _recv_exact(self, sock, n):
        """Receive exactly n bytes"""
        data = b''
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data


# 2 Tier Cache

class WorkerCache:
    """
    Two-tier cache per worker:
    - L1: RAM (LRU, fast access)
    - L2: SSD (persistent storage)

    Each worker maintains its own separate cache instance.
    """
    def __init__(self, ram_mb, ssd_dir, registry, rdma_host, rdma_port):
        self.ram_cap_bytes = ram_mb * 1024 * 1024
        self.ram = OrderedDict()
        self.ram_size = 0

        self.ssd_dir = Path(ssd_dir)
        self.ssd_dir.mkdir(parents=True, exist_ok=True)

        self.registry = registry
        self.rdma_host = rdma_host
        self.rdma_port = rdma_port

        # Start RDMA server
        self.rdma_server = RDMAServer(
            host=rdma_host,
            port=rdma_port,
            get_data_callback=self.get_local
        )

        # Update port if auto-assigned
        self.rdma_port = self.rdma_server.port
        self.endpoint = self.rdma_server.endpoint

    # INTERNAL: RAM handling
    def _ram_sizeof(self, data):
        img, lbl = data
        return img.nelement() * img.element_size() + 8

    def _ram_put(self, key, data):
        size = self._ram_sizeof(data)

        # evict if needed
        while self.ram_size + size > self.ram_cap_bytes and len(self.ram) > 0:
            k, v = self.ram.popitem(last=False)
            self.ram_size -= self._ram_sizeof(v)

        # insert/update
        if key in self.ram:
            self.ram_size -= self._ram_sizeof(self.ram[key])
        self.ram[key] = data
        self.ram.move_to_end(key)
        self.ram_size += size

    # INTERNAL: SSD handling
    def _ssd_path(self, key):
        return self.ssd_dir / f"{key}.pkl"

    def _ssd_put(self, key, data):
        p = self._ssd_path(key)
        tmp = p.with_suffix(".tmp")
        with open(tmp, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(p)

    def _ssd_get(self, key):
        p = self._ssd_path(key)
        if not p.exists():
            return None
        try:
            with open(p, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

    # PUBLIC CACHE GET (workers only)
    def get_local(self, idx):
        key = f"img_{idx}"

        # RAM
        if key in self.ram:
            self.ram[key] = self.ram.pop(key)
            return self.ram[key]

        # SSD
        data = self._ssd_get(key)
        if data is not None:
            self._ram_put(key, data)
            return data

        return None

    # PUBLIC CACHE PUT (workers only, after S3)
    def put(self, idx, img_t, label):
        key = f"img_{idx}"
        data = (img_t, label)

        self._ssd_put(key, data)
        self._ram_put(key, data)

        # Register ownership in Redis so trainer can find us
        self.registry.set_owner(idx, f"{self.rdma_host}:{self.rdma_port}")




class TrainerRDMADataset(Dataset):
    """
    Trainer-side dataset that:
    1. Checks Redis for owner
    2. Fetches via RDMA from worker if owner exists
    3. Falls back to S3 if no owner or RDMA fails
    4. Uses basic prefetching for sequential patterns
    """
    def __init__(
        self,
        num_images,
        registry: OwnerRegistry,
        bucket_name,
        aws_access_key_id,
        aws_secret_access_key,
        aws_region='us-east-1',
        rdma_timeout=3.0,
    ):
        self.num_images = num_images
        self.registry = registry
        self.rdma_client = RDMAClient(timeout=rdma_timeout)

        # S3 client for fallback
        self._s3_client = None
        self.bucket_name = bucket_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_region = aws_region

    def _get_s3_client(self):
        """Get or create S3 client (lazy init for multiprocessing)"""
        if self._s3_client is None:
            self._s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region
            )
        return self._s3_client

    def __len__(self):
        return self.num_images

    def _fetch_from_s3(self, idx):
        """Fetch image directly from S3 (fallback)"""
        s3_key = f'cifar10/images/{idx + 1}.raw'

        try:
            s3 = self._get_s3_client()
            resp = s3.get_object(Bucket=self.bucket_name, Key=s3_key)
            raw = resp['Body'].read()

            arr = np.frombuffer(raw, dtype=np.uint8).reshape(32, 32, 3)
            arr = np.ascontiguousarray(arr)
            img_t = torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0

            try:
                label = int(resp['Metadata'].get('label', '0'))
            except Exception:
                np.random.seed(idx)
                label = int(np.random.randint(0, 10))

            return img_t, label

        except Exception as e:
            print(f"[Trainer] S3 fetch failed for idx={idx}: {e}")
            return None, None

    def __getitem__(self, idx):
        """
        Fetch image via RDMA or S3 fallback

        Returns:
            (image_tensor, label, idx)
        """
        idx = int(idx)
        start_time = time.time()
        worker_id = os.getpid()

        # 1. Check Redis for owner
        owner = self.registry.get_owner(idx)

        if owner:
            # 2. Try RDMA fetch
            img_t, label = self.rdma_client.fetch(owner, idx)

            if img_t is not None:
                # RDMA success
                fetch_time = time.time() - start_time
                log_fetch(worker_id, idx, fetch_time, "RDMA")
                return img_t, label, idx

        # 3. Fallback to S3
        img_t, label = self._fetch_from_s3(idx)

        if img_t is not None:
            fetch_time = time.time() - start_time
            log_fetch(worker_id, idx, fetch_time, "S3_FALLBACK")
            return img_t, label, idx

        # Should rarely happen - return random data as last resort
        img_t = torch.rand(3, 32, 32)
        label = 0
        return img_t, label, idx


# ==============================================================================
# DATALOADER COLLATE FUNCTION
# ==============================================================================
def trainer_collate_fn(batch):
    """
    Collate function for trainer DataLoader

    Args:
        batch: List of (image_tensor, label, idx) tuples

    Returns:
        (images, labels, indices) as batched tensors
    """
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None, None, []

    imgs, labels, idxs = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    idxs = list(idxs)

    return imgs, labels, idxs

# Training Loop
def train_model(
    dataloader: DataLoader,
    num_epochs: int = 3,
    learning_rate: float = 1e-3
):
    """
    Training loop with ResNet152 on CIFAR-10
    """
    print("=== Starting Distributed Cached Training (RDMA + S3 Fallback) ===")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {dataloader.batch_size}")
    print("==================================================================")

    # Model
    model = models.resnet152(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Metrics
    total_train_time = 0.0
    total_dl_time = 0.0
    total_gpu_time = 0.0
    total_images = 0
    total_s3_calls = 0
    total_fetch_time = 0.0

    reset_log()
    main_pid = os.getpid()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        batches = 0

        print(f"\n----- Epoch {epoch+1}/{num_epochs} -----")
        it = iter(dataloader)

        while True:
            try:
                # DataLoader timing
                t_dl0 = time.time()
                batch = next(it)
                t_dl1 = time.time()
                dl_time = t_dl1 - t_dl0
                total_dl_time += dl_time

                if batch is None or batch[0] is None:
                    continue

                images, labels, idxs = batch
                log_batch(main_pid, idxs)

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                bsz = images.size(0)
                total_images += bsz
                total_s3_calls += bsz

                # GPU compute time
                t_gpu0 = time.time()
                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                t_gpu1 = time.time()
                gpu_time = t_gpu1 - t_gpu0
                total_gpu_time += gpu_time

                running_loss += float(loss.item())
                batches += 1

                if batches % 10 == 0:
                    print(f"Batch {batches:4d}: "
                          f"Loss={loss.item():.4f}, "
                          f"DL={dl_time:.3f}s, GPU={gpu_time:.3f}s, "
                          f"Images={bsz}")

            except StopIteration:
                break

        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time
        avg_loss = running_loss / batches if batches > 0 else 0.0

        # Get epoch stats from log
        epoch_end = time.time()
        cache_stats = get_epoch_stats_from_log(epoch_start, epoch_end)
        total_epoch_ops = sum(cache_stats.values())

        print(f"\n[Epoch {epoch+1} Summary]")
        print(f"  Batches:          {batches}")
        print(f"  Images processed: {total_images}")
        print(f"  Avg loss:         {avg_loss:.4f}")
        print(f"  Epoch time:       {epoch_time:.2f}s")
        if total_epoch_ops > 0:
            print(f"  Cache stats: RAM={cache_stats['ram_hits']} SSD={cache_stats['ssd_hits']} "
                  f"RDMA={cache_stats['rdma_hits']} S3={cache_stats['s3_fetches']} Fallback={cache_stats['s3_fallbacks']}")

    # Final metrics
    print("\nTraining Complete")
    print(f"Total images seen:      {total_images}")
    print(f"Total training time:    {total_train_time:.2f}s")
    print(f"Total DataLoader time:  {total_dl_time:.2f}s")
    print(f"Total GPU compute time: {total_gpu_time:.2f}s")

    dl_pct = (total_dl_time / total_train_time * 100) if total_train_time > 0 else 0.0
    gpu_pct = (total_gpu_time / total_train_time * 100) if total_train_time > 0 else 0.0

    print(f"DL time % of total:     {dl_pct:.1f}%")
    print(f"GPU time % of total:    {gpu_pct:.1f}%")

    if total_train_time > 0:
        print(f"Overall throughput:     {total_images / total_train_time:.1f} images/s")

    # Get final cache stats from entire log
    final_cache_stats = get_epoch_stats_from_log(0, time.time())

    metrics = {
        "total_images": total_images,
        "total_train_time": total_train_time,
        "total_dataloader_time": total_dl_time,
        "total_gpu_time": total_gpu_time,
        "throughput_img_per_s": (total_images / total_train_time) if total_train_time > 0 else 0.0,
        "dl_time_pct": dl_pct,
        "gpu_time_pct": gpu_pct,
        "total_s3_calls": total_s3_calls,
        "total_fetch_time": total_dl_time,  # approximate
        "avg_fetch_time": (total_dl_time / total_s3_calls) if total_s3_calls > 0 else 0.0,
        "data_requests_per_second": (total_s3_calls / total_train_time) if total_train_time > 0 else 0.0,
        "dataloader_percentage": dl_pct,
        "gpu_percentage": gpu_pct,
    }

    # Save metrics
    save_metrics_log(metrics, final_cache_stats)

    return metrics


# ==============================================================================
# HELPER: Build OwnerRegistry from environment
# ==============================================================================
def build_owner_registry_from_env():
    """Build OwnerRegistry from environment variables"""
    REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None
    REDIS_KEY_PREFIX = os.getenv("REDIS_OWNER_PREFIX", "cifar10")
    OWNER_TTL_SEC = int(os.getenv("OWNER_TTL_SEC", 600))

    print(f"[OwnerRegistry] host={REDIS_HOST} port={REDIS_PORT} db={REDIS_DB} "
          f"prefix='{REDIS_KEY_PREFIX}' ttl={OWNER_TTL_SEC}s")

    registry = OwnerRegistry(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        pw=REDIS_PASSWORD,
        prefix=REDIS_KEY_PREFIX,
        owner_ttl_sec=OWNER_TTL_SEC,
    )
    return registry


# ==============================================================================
# WORKER PROCESS
# ==============================================================================
def run_worker():
    """Worker process - caches data and serves via RDMA"""
    print("=== Starting WORKER (RDMA Server) ===")

    # S3 / dataset config
    BUCKET_NAME = os.getenv("BUCKET_NAME")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

    if not BUCKET_NAME or not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        raise RuntimeError("[WORKER] BUCKET_NAME / AWS credentials must be set")

    NUM_IMAGES = int(os.getenv("NUM_IMAGES", 50000))

    # Cache config
    RAM_CACHE_SIZE_MB = int(os.getenv("RAM_CACHE_SIZE_MB", 2048))
    BASE_SSD_CACHE_DIR = os.getenv("SSD_CACHE_DIR", "./ssd_cache_worker")

    # RDMA bind
    RDMA_BIND_HOST = os.getenv("RDMA_BIND_HOST", "0.0.0.0")
    RDMA_PORT = int(os.getenv("RDMA_PORT", 0))  # 0 = auto-pick

    # Sharding config
    SHARD_ID = int(os.getenv("SHARD_ID", 0))
    NUM_SHARDS = int(os.getenv("NUM_SHARDS", 1))

    shard_size = (NUM_IMAGES + NUM_SHARDS - 1) // NUM_SHARDS
    start_idx = SHARD_ID * shard_size
    end_idx = min(NUM_IMAGES, (SHARD_ID + 1) * shard_size)

    print(f"[WORKER] SHARD_ID={SHARD_ID} NUM_SHARDS={NUM_SHARDS}")
    print(f"[WORKER] Responsible for indices [{start_idx}, {end_idx})")
    print(f"[WORKER] RAM cache={RAM_CACHE_SIZE_MB} MB, SSD dir={BASE_SSD_CACHE_DIR}")

    # Build registry and cache
    registry = build_owner_registry_from_env()
    worker_ssd_dir = str(Path(BASE_SSD_CACHE_DIR) / f"worker_{SHARD_ID}")
    Path(worker_ssd_dir).mkdir(parents=True, exist_ok=True)

    cache = WorkerCache(
        ram_mb=RAM_CACHE_SIZE_MB,
        ssd_dir=worker_ssd_dir,
        registry=registry,
        rdma_host=RDMA_BIND_HOST,
        rdma_port=RDMA_PORT,
    )

    # S3 client
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )

    print(f"[WORKER] RDMA endpoint: {cache.endpoint}")
    print("=======================================================")

    # Warm loop: cache shard from S3
    while True:
        print(f"[WORKER {SHARD_ID}] Warming shard [{start_idx}, {end_idx})")
        t_shard_start = time.time()

        for idx in range(start_idx, end_idx):
            t0 = time.time()
            data = cache.get_local(idx)

            if data is not None:
                # Already cached - re-register ownership
                cache.registry.set_owner(idx, cache.endpoint)
                latency = time.time() - t0
                log_fetch(os.getpid(), idx, latency, "LOCAL")
                continue

            # MISS: fetch from S3 and cache
            s3_key = f"cifar10/images/{idx + 1}.raw"
            try:
                resp = s3.get_object(Bucket=BUCKET_NAME, Key=s3_key)
                raw = resp["Body"].read()

                arr = np.frombuffer(raw, dtype=np.uint8).reshape(32, 32, 3)
                arr = np.ascontiguousarray(arr)
                img_t = torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0

                try:
                    label = int(resp["Metadata"].get("label", "0"))
                except Exception:
                    np.random.seed(idx)
                    label = int(np.random.randint(0, 10))

                # Cache and register ownership
                cache.put(idx, img_t, label)
                latency = time.time() - t0
                log_fetch(os.getpid(), idx, latency, "S3")

            except Exception as e:
                print(f"[WORKER {SHARD_ID}] Error fetching idx={idx}: {e}")

        shard_time = time.time() - t_shard_start
        print(f"[WORKER {SHARD_ID}] Shard warm complete in {shard_time:.2f}s")

        print(f"[WORKER {SHARD_ID}] Sleeping 60s before next warm cycle")
        time.sleep(60)


# ==============================================================================
# TRAINER PROCESS
# ==============================================================================
def run_trainer():
    """Trainer process - fetches via RDMA with S3 fallback"""
    print("=== Starting TRAINER (RDMA Client + S3 Fallback) ===")

    # Dataset config
    NUM_IMAGES = int(os.getenv("NUM_IMAGES", 50000))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 3))
    NUM_WORKERS = int(os.getenv("DATALOADER_WORKERS", 4))

    # S3 config (for fallback)
    BUCKET_NAME = os.getenv("BUCKET_NAME")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

    if not BUCKET_NAME or not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        raise RuntimeError("[TRAINER] BUCKET_NAME / AWS credentials must be set")

    print(f"[TRAINER] NUM_IMAGES={NUM_IMAGES}")
    print(f"[TRAINER] BATCH_SIZE={BATCH_SIZE}")
    print(f"[TRAINER] NUM_EPOCHS={NUM_EPOCHS}")
    print(f"[TRAINER] DATALOADER_WORKERS={NUM_WORKERS}")

    # Build registry
    registry = build_owner_registry_from_env()

    # Create dataset
    dataset = TrainerRDMADataset(
        num_images=NUM_IMAGES,
        registry=registry,
        bucket_name=BUCKET_NAME,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_region=AWS_REGION,
        rdma_timeout=float(os.getenv("RDMA_TIMEOUT_SEC", 3.0)),
    )

    # Create DataLoader
    pin_memory = (device.type == "cuda")
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=trainer_collate_fn,
        pin_memory=pin_memory,
        persistent_workers=(NUM_WORKERS > 0),
    )

    print("[TRAINER] DataLoader created (RDMA + S3 fallback)")

    # Train
    metrics = train_model(
        dataloader=dataloader,
        num_epochs=NUM_EPOCHS,
        learning_rate=float(os.getenv("LEARNING_RATE", 1e-3)),
    )

    # Print final stats
    print("\n[TRAINER] Final metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    ROLE = os.getenv("ROLE", "trainer").lower()
    print(f"=== cached_resnet_rdma.py starting with ROLE='{ROLE}' ===\n")

    if ROLE == "worker":
        run_worker()
    else:
        run_trainer()
