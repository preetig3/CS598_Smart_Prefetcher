import os, time, numpy as np, boto3, multiprocessing as mp, queue as std_queue
from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Tuple
from collections import deque

from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")
load_dotenv('../config.env')

logger = 'pipeline_activity_seeded_prefetch_resnet152.log'
METRICS_LOG = os.getenv('METRICS_LOG', 'seeded_prefetch_resnet152_metrics.log')

# ---------------- Dataset ----------------
class BaselineS3CIFAR10Dataset(Dataset):
    """
    S3-backed CIFAR-10 (raw) with lazy per-process S3 client.
    __getitem__ logs fetch activity via log_worker_fetch().
    """
    def __init__(self, bucket_name: str, aws_access_key_id: str, aws_secret_access_key: str,
                 num_images: int = 50000, region: str = 'us-east-1'):
        self.bucket_name = bucket_name
        self.num_images = num_images
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region = region
        self.class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        self._s3_client = None
        print(f"Initialized baseline S3 dataset with {num_images} images")
        print("Using SeededPrefetchDataLoader (epoch-bridging, deterministic shuffle)")

    def _get_s3_client(self):
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
        start_time = time.time()
        worker_id = os.getpid()
        s3 = self._get_s3_client()
        s3_key = f'cifar10/images/{idx + 1}.raw'
        try:
            resp = s3.get_object(Bucket=self.bucket_name, Key=s3_key)
            raw = resp['Body'].read()
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(32, 32, 3).copy()
            img = torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0

            try:
                label = int(resp.get('Metadata', {}).get('label', '0'))
            except (KeyError, ValueError):
                # fallback label only for testing if metadata missing
                np.random.seed(idx)
                label = np.random.randint(0, 10)

            fetch_time = time.time() - start_time
            log_worker_fetch(worker_id, idx, fetch_time)
            return img, label, fetch_time, idx
        except Exception as e:
            print(f"Error fetching image {idx + 1} from S3: {e}")
            return None, None, 0, -1

def collate_fn(batch):
    valid = [b for b in batch if b[0] is not None]
    if not valid:
        return None
    images, labels, fetch_times, indices = zip(*valid)
    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    avg_fetch_time = float(np.mean(fetch_times))
    return images_tensor, labels_tensor, avg_fetch_time, list(indices)

def log_worker_fetch(worker_id, image_idx, fetch_time):
    try:
        with open(logger, 'a') as f:
            f.write(f"FETCH,{time.time():.3f},{worker_id},{image_idx},{fetch_time:.3f}\n")
    except:
        pass

def log_main_process_batch(main_id, batch_start_time, batch_size, image_indices):
    try:
        with open(logger, 'a') as f:
            f.write(f"PROCESS,{time.time():.3f},{main_id},{batch_size},{','.join(map(str, image_indices))}\n")
    except:
        pass

def log_queue_state(main_id, queue_state):
    """Log current queue state showing actual queue elements with their epochs"""
    try:
        with open(logger, 'a') as f:
            queue_elements = []
            for epoch, seq, batch_indices in queue_state:
                indices_str = ','.join(map(str, batch_indices))
                queue_elements.append(f"{epoch}#{seq}:[{indices_str}]")
            queue_info = ','.join(queue_elements)
            f.write(f"QUEUE,{time.time():.3f},{main_id},{queue_info}\n")
    except:
        pass

def clear_pipeline_log():
    try:
        if os.path.exists(logger):
            os.remove(logger)
        with open(logger, 'w') as f:
            f.write("# Pipeline Activity Log - Seeded Prefetch ResNet152\n")
            f.write("# FETCH: timestamp,worker_id,image_idx,fetch_time\n")
            f.write("# PROCESS: timestamp,main_pid,batch_size,indices...\n")
            f.write("# QUEUE: timestamp,main_pid,epoch#seq:[batch_indices],...\n\n")
    except:
        pass

def save_metrics_log(metrics):
    try:
        with open(METRICS_LOG, 'w') as f:
            f.write("# Seeded Prefetch DataLoader Metrics - ResNet152\n")
            f.write("# Generated on: {}\n\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
        print(f"\nMetrics saved to {METRICS_LOG}")
    except Exception as e:
        print(f"Error saving metrics: {e}")

# Prefetcher 
@dataclass
class BatchTask:
    kind: str                 
    payload: Optional[Any]     # List[int] of indices
    epoch: Optional[int] = None
    seq: Optional[int] = None  # batch sequence within epoch

def _worker_loop(dataset: BaselineS3CIFAR10Dataset,
                 in_q: mp.Queue, out_q: mp.Queue, _collate, stop_event: mp.Event):
    dataset._s3_client = None
    while True:
        if stop_event.is_set():
            break
        task: BatchTask = in_q.get()
        if task is None:
            continue
        if task.kind == "STOP":
            break
        if task.kind == "IDX_BATCH":
            if stop_event.is_set():
                break
            idxs: List[int] = task.payload
            samples = []
            for idx in idxs:
                if stop_event.is_set():
                    break
                s = dataset[idx]
                if s is not None:
                    samples.append(s)
            try:
                if not stop_event.is_set():
                    out_q.put((task.epoch, task.seq, _collate(samples)))
            except Exception:
                out_q.put(None)

class SeededPrefetchDataLoader:
    """
    Deterministic epoch-bridging prefetcher with strict ordering:
    - Batches are assigned a per-epoch sequence number (seq).
    - Main thread yields batches in seq order for that epoch, regardless of worker completion order.
    - Next-epoch prefetch is allowed, but next-epoch batches are stashed and not yielded until the next epoch.
    """
    def __init__(self, dataset: BaselineS3CIFAR10Dataset, batch_size: int,
                 num_workers: int = 4, prefetch_batches: int = 8,
                 shuffle: bool = True, seed: int = 12345, collate_fn=collate_fn,
                 next_epoch_warmup: int = 4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_batches = max(prefetch_batches, num_workers)
        self.shuffle = shuffle
        self.seed = seed
        self.collate_fn = collate_fn
        self.next_epoch_warmup = max(0, int(next_epoch_warmup))

        ctx = mp.get_context("spawn")
        self._in_q = ctx.Queue(maxsize=self.prefetch_batches * 2)
        self._out_q = ctx.Queue(maxsize=self.prefetch_batches * 2)
        self._stop_event = ctx.Event()

        self._procs: List[mp.Process] = []
        for _ in range(num_workers):
            p = ctx.Process(target=_worker_loop,
                            args=(self.dataset, self._in_q, self._out_q, self.collate_fn, self._stop_event),
                            daemon=True)
            p.start()
            self._procs.append(p)

    
        self._carry_payloads: Dict[int, Dict[int, Any]] = {}
        # how many next-epoch batches we already enqueued last time
        self._carry_enqueued: Dict[int, int] = {}

    def _epoch_indices(self, epoch: int) -> np.ndarray:
        n = len(self.dataset)
        idxs = np.arange(n, dtype=np.int64)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + epoch)
            rng.shuffle(idxs)
        return idxs

    def _to_batches(self, idxs: np.ndarray) -> List[List[int]]:
        return [idxs[i:i+self.batch_size].tolist()
                for i in range(0, len(idxs), self.batch_size)]

    def iter_epoch(self, epoch: int, log_queue_callback=None, is_last_epoch=False):
        cur_idxs = self._epoch_indices(epoch)
        cur_batches = self._to_batches(cur_idxs)
        nxt_batches = [] if is_last_epoch else self._to_batches(self._epoch_indices(epoch + 1))

        total_cur = len(cur_batches)
        enq_cur = 0
        deq_cur = 0
        enq_nxt = 0

        # If we carried over prefetched next-epoch work, prime our bookkeeping
        # We will not re-enqueue those first enq_nxt batches next epoch.
        # For current epoch, check if we have carried payloads (should not happen)
        if epoch in self._carry_enqueued:
            # this happens when previous epoch prefetched some of OUR epoch batches
            prefetched_n = self._carry_enqueued.pop(epoch)
            enq_cur = min(prefetched_n, total_cur)
        else:
            enq_cur = 0
        cur_buffer: Dict[int, Any] = {}
        expected_seq = 0

 
        queue_state: deque[Tuple[int,int,List[int]]] = deque()

        # prime with current epoch up to capacity, starting from enq_cur (skip any already-enqueued from carry)
        while enq_cur < min(self.prefetch_batches, total_cur):
            seq = enq_cur
            batch_indices = cur_batches[seq]
            self._in_q.put(BatchTask("IDX_BATCH", batch_indices, epoch=epoch, seq=seq))
            queue_state.append((epoch, seq, batch_indices))
            enq_cur += 1

        pending_estimate = enq_cur  # how many submitted but not yet yielded for current epoch
        # preload next-epoch payloads that were produced during prior epoch bridging
        if epoch in self._carry_payloads:
            cur_buffer.update(self._carry_payloads.pop(epoch))

        while deq_cur < total_cur:
            # top-up current-epoch submissions
            while enq_cur < total_cur and pending_estimate < self.prefetch_batches:
                try:
                    seq = enq_cur
                    batch_indices = cur_batches[seq]
                    self._in_q.put(BatchTask("IDX_BATCH", batch_indices, epoch=epoch, seq=seq), block=False)
                    queue_state.append((epoch, seq, batch_indices))
                    enq_cur += 1
                    pending_estimate += 1
                except Exception:
                    break

            # (enqueue next epoch early)
            if self.next_epoch_warmup > 0 and nxt_batches:
                remaining_cur = total_cur - deq_cur
                if remaining_cur <= self.next_epoch_warmup:
                    while enq_nxt < len(nxt_batches) and pending_estimate < (self.prefetch_batches * 2):
                        try:
                            nxt_seq = enq_nxt  # sequence within next epoch
                            batch_indices = nxt_batches[nxt_seq]
                            self._in_q.put(BatchTask("IDX_BATCH", batch_indices, epoch=epoch+1, seq=nxt_seq), block=False)
                            queue_state.append((epoch+1, nxt_seq, batch_indices))
                            enq_nxt += 1
                            # NOTE: do not change pending_estimate for current-epoch accounting
                        except Exception:
                            break

            # receive one completed batch (could be from current or next epoch)
            rec = self._out_q.get()
            if rec is None:
                continue
            batch_epoch, batch_seq, batch_payload = rec

            # optional queue state log
            if log_queue_callback and queue_state:
                log_queue_callback(list(queue_state))

            # pop one from the logging queue_state if it matches what we just got
            # (best-effort: head of queue_state might not match if completion order differs)
            if queue_state and queue_state[0][0] == batch_epoch and queue_state[0][1] == batch_seq:
                queue_state.popleft()

            if batch_epoch == epoch:
                # buffer it and try to yield in order
                cur_buffer[batch_seq] = batch_payload
                # if next in sequence available, yield as many as possible
                while expected_seq in cur_buffer:
                    payload = cur_buffer.pop(expected_seq)
                    # yield strictly in-seq order
                    yield payload
                    deq_cur += 1
                    pending_estimate = max(0, pending_estimate - 1)
                    expected_seq += 1
            else:
                # stash next-epoch payloads for later
                if batch_epoch not in self._carry_payloads:
                    self._carry_payloads[batch_epoch] = {}
                self._carry_payloads[batch_epoch][batch_seq] = batch_payload

            # after finishing current epoch, keep topping up next-epoch enqueues if we haven't already
            if deq_cur == total_cur and nxt_batches and enq_nxt < len(nxt_batches):
                # record how many next-epoch batches have been enqueued so next iter skips re-enqueue
                self._carry_enqueued[epoch+1] = enq_nxt
                # optionally keep enqueuing more next-epoch work here (not required, but okay)
                while enq_nxt < len(nxt_batches) and pending_estimate < self.prefetch_batches * 2:
                    try:
                        nxt_seq = enq_nxt
                        batch_indices = nxt_batches[nxt_seq]
                        self._in_q.put(BatchTask("IDX_BATCH", batch_indices, epoch=epoch+1, seq=nxt_seq), block=False)
                        queue_state.append((epoch+1, nxt_seq, batch_indices))
                        enq_nxt += 1
                    except Exception:
                        break

        # ensure we remember how many next-epoch batches were enqueued (if we ended before setting it)
        if nxt_batches and (epoch+1) not in self._carry_enqueued:
            self._carry_enqueued[epoch+1] = enq_nxt

    def shutdown(self):
        try:
            self._stop_event.set()
        except Exception:
            pass

        try:
            while True:
                self._in_q.get_nowait()
        except (std_queue.Empty, AttributeError):
            pass
        except Exception:
            pass

        for _ in range(self.num_workers):
            try:
                self._in_q.put(BatchTask("STOP", None), block=False)
            except Exception:
                pass

        for p in self._procs:
            p.join(timeout=5)
        try:
            self._in_q.close(); self._out_q.close()
        except Exception:
            pass

# ---------------- Training ----------------
def train_model(prefetcher: SeededPrefetchDataLoader, num_epochs: int = 3, lr: float = 1e-3):
    print("starting training with SeededPrefetchDataLoader (ResNet152)")
    model = models.resnet152(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    print(f"ResNet152 -> {device}; params={sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    clear_pipeline_log()
    main_pid = os.getpid()

    totals = dict(total_s3_calls=0, total_fetch_time=0.0,
                  total_dataloader_time=0.0, total_gpu_time=0.0, total_training_time=0.0)

    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_s3 = 0
        epoch_fetch = 0.0
        epoch_dl_accum = 0.0
        epoch_loss = 0.0
        batches = 0

        model.train()
        print(f"\nEpoch {epoch+1}/{num_epochs}\n-----")

        def log_queue_callback(queue_state):
            log_queue_state(main_pid, queue_state)

        is_last_epoch = (epoch == num_epochs - 1)
        it = prefetcher.iter_epoch(epoch, log_queue_callback=log_queue_callback, is_last_epoch=is_last_epoch)

        while True:
            try:
                dl_start = time.time()
                batch = next(it)
                dl_time = time.time() - dl_start
                epoch_dl_accum += dl_time

                if batch is None:
                    continue

                images, labels, avg_fetch_time, indices = batch
                log_main_process_batch(main_pid, time.time(), len(images), indices)

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                epoch_s3 += len(images)
                epoch_fetch += avg_fetch_time * len(images)
                epoch_loss += loss.item()
                batches += 1

                if batches % 10 == 0:
                    print(f"Batch {batches:3d}: loss={loss.item():.4f}, DataLoader={dl_time:.3f}s")

            except StopIteration:
                break

        epoch_time = time.time() - epoch_start
        epoch_dl = epoch_dl_accum
        epoch_gpu = epoch_time - epoch_dl

        dataloader_pct = (epoch_dl / max(1e-9, epoch_time)) * 100.0
        gpu_pct = (epoch_gpu / max(1e-9, epoch_time)) * 100.0
        avg_loss = epoch_loss / max(1, batches)
        avg_fetch_img = epoch_fetch / max(1, epoch_s3)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  S3 Calls: {epoch_s3}")
        print(f"  DataLoader Time: {epoch_dl:.2f}s ({dataloader_pct:.1f}%)")
        print(f"  GPU Time: {epoch_gpu:.2f}s ({gpu_pct:.1f}%)")
        print(f"  Total Time: {epoch_time:.2f}s")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Avg S3 Fetch/Image: {avg_fetch_img:.3f}s")

        totals['total_s3_calls'] += epoch_s3
        totals['total_fetch_time'] += epoch_fetch
        totals['total_dataloader_time'] += epoch_dl
        totals['total_gpu_time'] += epoch_gpu
        totals['total_training_time'] += epoch_time

    final = {
        **totals,
        'avg_fetch_time': totals['total_fetch_time'] / max(1, totals['total_s3_calls']),
        'dataloader_percentage': totals['total_dataloader_time'] / max(1e-9, totals['total_training_time']) * 100.0,
        'gpu_percentage': totals['total_gpu_time'] / max(1e-9, totals['total_training_time']) * 100.0,
        's3_calls_per_second': totals['total_s3_calls'] / max(1e-9, totals['total_training_time'])
    }
    save_metrics_log(final)
    return final

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    BUCKET_NAME = os.getenv('BUCKET_NAME')
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

    NUM_IMAGES = int(os.getenv('NUM_IMAGES', 50000))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
    NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', 3))
    NUM_WORKERS = int(os.getenv('NUM_WORKERS', 4))

    print("\nSeededPrefetchDataLoader: deterministic epoch-bridging prefetch")
    print(f"Workers={NUM_WORKERS} | Batch={BATCH_SIZE}\n")

    dataset = BaselineS3CIFAR10Dataset(
        bucket_name=BUCKET_NAME,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        num_images=NUM_IMAGES
    )

    prefetcher = SeededPrefetchDataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_batches=8,
        shuffle=True,
        seed=42,                # PRNG seed; order = rng(seed + epoch)
        collate_fn=collate_fn,
        next_epoch_warmup=4     # start fetching from next epoch early
    )

    try:
        metrics = train_model(prefetcher, num_epochs=NUM_EPOCHS, lr=1e-3)
        print("\nFinal Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print(f"\nMetrics saved to {METRICS_LOG}")
    finally:
        prefetcher.shutdown()

