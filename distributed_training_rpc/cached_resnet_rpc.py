#!/usr/bin/env python3
import os
import time
import socket
import pickle
import threading

from pathlib import Path
from collections import OrderedDict

from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
from socketserver import ThreadingMixIn
import xmlrpc.client

import boto3
import redis
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv

# GLOBAL ENV + DEVICE
load_dotenv('./config.env')

device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu'
)
print(f"[INFO] Using device: {device}")

LOG_FILE = "pipeline_activity_rpc.log"



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


# XML-RPC WORKER SERVER
class ThreadedXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    daemon_threads = True
    allow_reuse_address = True


class WorkerRPC:
    """Worker-side: serves fetch(idx) from RAM+SSD only."""
    def __init__(self, get_local):
        self._get_local = get_local

    def fetch(self, idx):
        data = self._get_local(idx)
        if data is None:
            return None
        img_t, label = data
        arr = (img_t * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        payload = pickle.dumps((arr, int(label)), protocol=pickle.HIGHEST_PROTOCOL)
        return xmlrpc.client.Binary(payload)


# WORKER CACHE:
#   L1 = RAM (ordered dict, capacity-limited)
#   L2 = SSD (per-worker directory)
#   Redis owner registry is used so trainer can find which worker has idx.
#
# Workers DO hit S3.
# Trainer NEVER hits S3.
class WorkerCache:
    def __init__(self, ram_mb, ssd_dir, registry, rpc_host, rpc_port):
        self.ram_cap_bytes = ram_mb * 1024 * 1024
        self.ram = OrderedDict()
        self.ram_size = 0

        self.ssd_dir = Path(ssd_dir)
        self.ssd_dir.mkdir(parents=True, exist_ok=True)

        self.registry = registry
        self.rpc_host = rpc_host
        self.rpc_port = rpc_port

        # start RPC server immediately
        self._start_rpc_server()

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
        self.registry.set_owner(idx, f"{self.rpc_host}:{self.rpc_port}")

    # RPC SERVER
    def _start_rpc_server(self):
        # choose port automatically if 0
        if self.rpc_port == 0:
            s = socket.socket()
            s.bind((self.rpc_host, 0))
            self.rpc_port = s.getsockname()[1]
            s.close()

        self.endpoint = f"http://{self.rpc_host}:{self.rpc_port}"
        print(f"[WORKER RPC] Starting server at {self.endpoint}")

        class Req(SimpleXMLRPCRequestHandler):
            rpc_paths = ("/RPC2",)

        server = ThreadedXMLRPCServer(
            (self.rpc_host, self.rpc_port),
            requestHandler=Req,
            allow_none=True,
            logRequests=False
        )
        server.register_instance(WorkerRPC(self.get_local))

        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()



# TRAINER DATASET (RPC-only, NO S3, BLOCKING)
class TrainerRPCDataset(Dataset):
    def __init__(
        self,
        num_images,
        registry: OwnerRegistry,
        rpc_timeout=3.0,
        retry_sleep=0.01,
    ):
        """
        num_images:   total logical images (e.g., 50k)
        registry:     OwnerRegistry pointing at Redis
        rpc_timeout:  per-RPC timeout in seconds
        retry_sleep:  sleep between retries when owner/RPC missing
        """
        self.num_images = num_images
        self.registry = registry
        self.rpc_timeout = float(rpc_timeout)
        self.retry_sleep = float(retry_sleep)

    def __len__(self):
        return self.num_images

    def _rpc_fetch(self, owner, idx, start_t):
        """
        Do one XML-RPC call to owner ("host:port") for idx,
        log as RPC on success. Return (img_t, lbl) or (None, None) on failure.
        """
        try:
            url = f"http://{owner}"
            proxy = xmlrpc.client.ServerProxy(url, allow_none=True)

            old_to = socket.getdefaulttimeout()
            try:
                socket.setdefaulttimeout(self.rpc_timeout)
                res = proxy.fetch(int(idx))
            finally:
                socket.setdefaulttimeout(old_to)

            if not isinstance(res, xmlrpc.client.Binary):
                return None, None

            arr, lbl = pickle.loads(res.data)  # arr: uint8 CHW
            img_t = torch.from_numpy(arr).float() / 255.0
            lbl = int(lbl)

            latency = time.time() - start_t
            log_fetch(os.getpid(), idx, latency, "RPC")
            return img_t, lbl
        except Exception:
            # treat as miss; caller will retry
            return None, None

    def __getitem__(self, idx):
        """
        BLOCKING semantics:
          - Wait until some worker owns this idx in Redis
          - Then keep retrying RPC until it succeeds
        """
        idx = int(idx)
        start = time.time()
        wait_loops = 0

        while True:
            owner = self.registry.get_owner(idx)

            if owner:
                img_t, lbl = self._rpc_fetch(owner, idx, start)
                if img_t is not None:
                    return img_t, lbl, idx

                # We had an owner, but RPC failed (worker restarting, etc.)
                time.sleep(self.retry_sleep)
                wait_loops += 1
                if wait_loops % 500 == 0:
                    print(f"[TRAINER] Still waiting for RPC success for idx {idx}, owner={owner}")
                continue

            # No owner yet; wait for workers to warm this idx
            time.sleep(self.retry_sleep)
            wait_loops += 1
            if wait_loops % 500 == 0:
                print(f"[TRAINER] Waiting for owner of idx {idx} (no Redis entry yet)")


# DATALOADER COLLATE FN (Trainer side)
def trainer_collate_fn(batch):
    """
    Batch elements: (img_t, label, idx)
    We drop any None entries (failed fetches) – but with blocking
    TrainerRPCDataset, you shouldn't get None.
    """
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None, None, []

    imgs, labels, idxs = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    idxs = list(idxs)
    return imgs, labels, idxs



# TRAINING LOOP (Trainer only, uses TrainerRPCDataset)
def train_model(
    dataloader: DataLoader,
    num_epochs: int = 3,
    learning_rate: float = 1e-3
):
    """
    Classic single-GPU training, but all data comes from workers
    via RPC. No S3 calls from trainer.
    """
    print("=== Starting Distributed Cached Training (Trainer RPC-only) ===")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {dataloader.batch_size}")
    print("===============================================================")

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
                # DataLoader timing (RPC fetch happens under the hood)
                t_dl0 = time.time()
                batch = next(it)
                t_dl1 = time.time()
                dl_time = t_dl1 - t_dl0
                total_dl_time += dl_time

                if batch is None:
                    print(f"[TRAINER] Got empty batch, skipping.")
                    continue

                images, labels, idxs = batch
                log_batch(main_pid, idxs)

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                bsz = images.size(0)
                total_images += bsz

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
                          f"Images this batch={bsz}")

            except StopIteration:
                break

        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time
        avg_loss = running_loss / batches if batches > 0 else 0.0

        print(f"\n[Epoch {epoch+1} Summary]")
        print(f"  Batches:          {batches}")
        print(f"  Images processed: {total_images}")
        print(f"  Avg loss:         {avg_loss:.4f}")
        print(f"  Epoch time:       {epoch_time:.2f}s")

    # Final Metrics 
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

    metrics = {
        "total_images": total_images,
        "total_train_time": total_train_time,
        "total_dataloader_time": total_dl_time,
        "total_gpu_time": total_gpu_time,
        "throughput_img_per_s": (total_images / total_train_time) if total_train_time > 0 else 0.0,
        "dl_time_pct": dl_pct,
        "gpu_time_pct": gpu_pct,
    }
    return metrics



# HELPER: Build shared OwnerRegistry from env
def build_owner_registry_from_env():
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



# WORKER PROCESS
def run_worker():
    print("=== Starting WORKER ===")

    # ---- S3 / dataset env ----
    BUCKET_NAME = os.getenv("BUCKET_NAME")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

    if not BUCKET_NAME or not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        raise RuntimeError("[WORKER] BUCKET_NAME / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY must be set in env.")

    NUM_IMAGES = int(os.getenv("NUM_IMAGES", 50000))

    # ---- Cache config (per worker) ----
    RAM_CACHE_SIZE_MB = int(os.getenv("RAM_CACHE_SIZE_MB", 2048))
    BASE_SSD_CACHE_DIR = os.getenv("SSD_CACHE_DIR", "./ssd_cache_worker")

    # RPC bind (this host must be reachable from trainer)
    RPC_BIND_HOST = os.getenv("RPC_BIND_HOST", "0.0.0.0")
    RPC_PORT = int(os.getenv("RPC_PORT", 0))  # 0 = auto-pick

    # ---- Sharding config ----
    SHARD_ID = int(os.getenv("SHARD_ID", 0))
    NUM_SHARDS = int(os.getenv("NUM_SHARDS", 1))

    shard_size = (NUM_IMAGES + NUM_SHARDS - 1) // NUM_SHARDS
    start_idx = SHARD_ID * shard_size
    end_idx = min(NUM_IMAGES, (SHARD_ID + 1) * shard_size)

    print(f"[WORKER] SHARD_ID={SHARD_ID} NUM_SHARDS={NUM_SHARDS}")
    print(f"[WORKER] Responsible for indices [{start_idx}, {end_idx}) out of 0..{NUM_IMAGES-1}")
    print(f"[WORKER] RAM cache={RAM_CACHE_SIZE_MB} MB, SSD base dir={BASE_SSD_CACHE_DIR}")

    # Build OwnerRegistry and cache (starts XML-RPC server)
    registry = build_owner_registry_from_env()
    worker_ssd_dir = str(Path(BASE_SSD_CACHE_DIR) / f"worker_{SHARD_ID}")
    Path(worker_ssd_dir).mkdir(parents=True, exist_ok=True)

    cache = WorkerCache(
        ram_mb=RAM_CACHE_SIZE_MB,
        ssd_dir=worker_ssd_dir,
        registry=registry,
        rpc_host=RPC_BIND_HOST,
        rpc_port=RPC_PORT,
    )

    # S3 client
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )

    print("[WORKER] S3 client initialized.")
    print("[WORKER] Cache RPC endpoint should be printed above as [WORKER RPC] Starting server at ...")
    print("=======================================================")

    # Warm loop: read shard from S3 into cache (once per miss)
    while True:
        print(f"[WORKER {SHARD_ID}] Warming shard [{start_idx}, {end_idx})")
        t_shard_start = time.time()
        for idx in range(start_idx, end_idx):
            t0 = time.time()
            data = cache.get_local(idx)

            if data is not None:
                # Already hot in RAM/SSD; re-register owner in Redis
                img_t, label = data
                cache.registry.set_owner(idx, f"{cache.rpc_host}:{cache.rpc_port}")
                latency = time.time() - t0
                log_fetch(os.getpid(), idx, latency, "LOCAL")
                continue

            # MISS: this worker becomes the first owner – fetch from S3
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

                # Write into cache (L2 SSD + L1 RAM) and register ownership in Redis
                cache.put(idx, img_t, label)
                latency = time.time() - t0
                log_fetch(os.getpid(), idx, latency, "S3")
            except Exception as e:
                print(f"[WORKER {SHARD_ID}] Error fetching idx={idx} s3_key={s3_key}: {e}")

        shard_time = time.time() - t_shard_start
        print(f"[WORKER {SHARD_ID}] Shard warm complete in {shard_time:.2f}s, sleeping 60s")
        time.sleep(60)


# TRAINER PROCESS
def run_trainer():
    print("=== Starting TRAINER (RPC-only, no S3) ===")

    NUM_IMAGES = int(os.getenv("NUM_IMAGES", 50000))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 3))
    NUM_WORKERS = int(os.getenv("DATALOADER_WORKERS", 4))

    print(f"[TRAINER] NUM_IMAGES={NUM_IMAGES}")
    print(f"[TRAINER] BATCH_SIZE={BATCH_SIZE}")
    print(f"[TRAINER] NUM_EPOCHS={NUM_EPOCHS}")
    print(f"[TRAINER] DATALOADER_WORKERS={NUM_WORKERS}")

    registry = build_owner_registry_from_env()

    dataset = TrainerRPCDataset(
        num_images=NUM_IMAGES,
        registry=registry,
        rpc_timeout=float(os.getenv("RPC_TIMEOUT_SEC", 3.0)),
        retry_sleep=float(os.getenv("TRAINER_RETRY_SLEEP", 0.01)),
    )

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

    print("[TRAINER] DataLoader created (RPC-backed).")

    metrics = train_model(
        dataloader=dataloader,
        num_epochs=NUM_EPOCHS,
        learning_rate=float(os.getenv("LEARNING_RATE", 1e-3)),
    )

    print("\n[TRAINER] Final metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


# MAIN ENTRYPOINT
if __name__ == "__main__":
    ROLE = os.getenv("ROLE", "trainer").lower()
    print(f"=== cached_resnet_rpc.py starting with ROLE='{ROLE}' ===")

    if ROLE == "worker":
        run_worker()
    else:
        run_trainer()
