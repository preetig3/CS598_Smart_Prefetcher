import boto3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import numpy as np
import time
import os
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# env vars (create a config.env)
load_dotenv('../config.env')

# made a logger to track activity for each worker, and main process
logger = 'pipeline_activity_resnet152.log'

# the dataset (dataloader uses getitem to get next indices with S3 calls)
class BaselineS3CIFAR10Dataset(Dataset):
    """
    baseline S3 cifar10 Dataset with pytorch's builtin prefetching
    uses num_workers for background data loading
    """
    def __init__(self, bucket_name: str, aws_access_key_id: str, aws_secret_access_key: str, 
                 num_images: int = 50000, region: str = 'us-east-1'):
        self.bucket_name = bucket_name
        self.num_images = num_images
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region = region
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Create S3 client once per worker process (not per sample)
        self._s3_client = None
        
        print(f"Initialized baseline S3 dataset with {num_images} images")
        print("Baseline: Using PyTorch's num_workers for prefetching")
    
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
        fetch image from S3 and return fetch time
        this is what is called by the background workers
        """
        start_time = time.time()
        
        # get worker ID for tracking
        worker_id = os.getpid()
        
        # use shared S3 client (created once per worker process)
        s3_client = self._get_s3_client()
        
        # s3 key for image
        s3_key = f'cifar10/images/{idx + 1}.raw'  # the images are 1 indexed
        
        try:
            # S3 GET call
            response = s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            # read raw image data
            raw_data = response['Body'].read()
            
            # Convert to  32/ 32/ 3 numpy array
            image_array = np.frombuffer(raw_data, dtype=np.uint8).reshape(32, 32, 3)
            
            image_array = image_array.copy()
            
            # tensor and normalize
            image_tensor = torch.from_numpy(image_array).float()
            image_tensor = image_tensor.permute(2, 0, 1) 
            image_tensor = image_tensor / 255.0 
            
            # try to get label from S3 metadata or use random (cuz label doesn't matter)
            try:
                label = int(response['Metadata'].get('label', '0'))
            except (KeyError, ValueError):
                # use random label if metadata not found
                np.random.seed(idx)
                label = np.random.randint(0, 10)
            
            fetch_time = time.time() - start_time
            
            # log worker fetch activity
            log_worker_fetch(worker_id, idx, fetch_time)
            
            return image_tensor, label, fetch_time, idx
            
        except Exception as e:
            print(f"Error fetching image {idx + 1} from S3: {e}")
            return None, None, 0, -1

def collate_fn(batch):
    """
    Custom collate function to handle the fetch_time data and indices
    """
    # Filter out None values (failed S3 calls)
    valid_batch = [item for item in batch if item[0] is not None]
    
    if not valid_batch:
        return None, None, 0, []
    
    images, labels, fetch_times, indices = zip(*valid_batch)
    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    avg_fetch_time = np.mean(fetch_times)
    
    return images_tensor, labels_tensor, avg_fetch_time, list(indices)

def log_worker_fetch(worker_id, image_idx, fetch_time):
    """log when a worker fetches an image"""
    try:
        with open(logger, 'a') as f:
            f.write(f"FETCH,{time.time():.3f},{worker_id},{image_idx},{fetch_time:.3f}\n")
    except:
        pass

def log_main_process_batch(main_id, batch_start_time, batch_size, image_indices):
    """log when main process starts processing a batch"""
    try:
        with open(logger, 'a') as f:
            f.write(f"PROCESS,{time.time():.3f},{main_id},{batch_size},{','.join(map(str, image_indices))}\n")
    except:
        pass

def clear_pipeline_log():
    """clear the pipeline log for fresh monitoring"""
    try:
        if os.path.exists(logger):
            os.remove(logger)
        
        with open(logger, 'w') as f:
            f.write("# Pipeline Activity Log - ResNet152\n")
            f.write("# Format: ACTIVITY_TYPE,TIMESTAMP,PROCESS_ID,DATA\n")
            f.write("# FETCH: Worker fetches image from S3 (timestamp,worker_id,image_idx,fetch_time)\n")
            f.write("# PROCESS: Main process processes batch (timestamp,main_id,batch_size,image_indices)\n")
            
            f.write("#\n")
    except:
        pass

def save_metrics_log(metrics):
    """save metrics to baseline_S3_dataloader_resnet152_metrics.log"""
    try:
        with open('baseline_S3_dataloader_resnet152_metrics.log', 'w') as f:
            f.write("# Baseline S3 DataLoader Metrics - ResNet152\n")
            f.write("# Generated on: {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
            f.write("#\n")
            f.write("# Performance Summary:\n")
            f.write(f"Total S3 Calls: {metrics['total_s3_calls']}\n")
            f.write(f"Total Fetch Time: {metrics['total_fetch_time']:.2f}s\n")
            f.write(f"Total DataLoader Time: {metrics['total_dataloader_time']:.2f}s ({metrics['dataloader_percentage']:.1f}%)\n")
            f.write(f"Total GPU Time: {metrics['total_gpu_time']:.2f}s ({metrics['gpu_percentage']:.1f}%)\n")
            f.write(f"Total Training Time: {metrics['total_training_time']:.2f}s\n")
            f.write(f"Average S3 Fetch Time per Image: {metrics['avg_fetch_time']:.3f}s\n")
            f.write(f"S3 Calls per Second: {metrics['s3_calls_per_second']:.1f}\n")
            f.write("\n")
            f.write("# Detailed Metrics:\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            f.write("# Analysis:\n")
            f.write(f"DataLoader Time: {metrics['dataloader_percentage']:.1f}% (waiting for data)\n")
            f.write(f"GPU Time: {metrics['gpu_percentage']:.1f}% (actual compute)\n")
            f.write(f"Bottleneck: {metrics['dataloader_percentage']:.1f}% waiting time\n")
    
    except Exception as e:
        print(f"Error saving metrics: {e}")

def train_model(dataloader, num_epochs: int = 5, learning_rate: float = 0.001):
    """
    Train the baseline model with ResNet152 and PyTorch's built-in prefetching
    """
    print("starting baseline training with ResNet152 (PyTorch num_workers)")
    print("-----")
    
    # Load ResNet152 model
    model = models.resnet152(pretrained=False)  # No pretrained weights for fair comparison
    
    # Modify the final layer for CIFAR-10 (10 classes)
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    # use device for model
    model = model.to(device)
    print(f"ResNet152 model moved to {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # clear pipeline log for fresh monitoring
    clear_pipeline_log()
    
    # get main process id
    main_process_id = os.getpid()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # train metrcs
    total_s3_calls = 0
    total_fetch_time = 0.0
    total_dataloader_time = 0.0
    total_gpu_time = 0.0
    total_training_time = 0.0
    epoch_times = []
    
    for epoch in range(num_epochs):
        time_model_train_start = time.time()
        epoch_start = time.time()
        epoch_s3_calls = 0
        epoch_fetch_time = 0.0
        epoch_gpu_time = 0.0
        epoch_gpu_wait_time = 0.0
        epoch_loss = 0.0
        batch_count = 0
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-----")
        
        # train
        model.train()
        
        # Create iterator to measure DataLoader blocking time
        dataloader_iter = iter(dataloader)
        epoch_dataloader_time = 0.0
        
        while True:
            try:
                # Measure DataLoader blocking time (time waiting for data)
                dataloader_start = time.time()
                batch_data = next(dataloader_iter)
                dataloader_time = time.time() - dataloader_start
                epoch_dataloader_time += dataloader_time
                
                if batch_data is None:
                    print(f"Batch {batch_count} failed - skipping")
                    continue
                    
                images, labels, avg_fetch_time, image_indices = batch_data
                
                # log main process batch processing with actual indices
                batch_size = len(images)
                log_main_process_batch(main_process_id, time.time(), batch_size, image_indices)
                
                # move data to gpu/device
                images = images.to(device)
                labels = labels.to(device)
                
                # forward pass
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # backward pass
                loss.backward()
                optimizer.step()
                
                # update metrics
                epoch_s3_calls += len(images)
                epoch_fetch_time += avg_fetch_time * len(images)
                epoch_loss += loss.item()
                batch_count += 1
                
                # print progress
                if batch_count % 10 == 0:
                    print(f"Batch {batch_count:3d}: Loss={loss.item():.4f}, "
                          f"DataLoader={dataloader_time:.3f}s")
                          
            except StopIteration:
                break
        
        epoch_time = time.time() - epoch_start
        
        # Simple metrics: DataLoader time vs GPU time
        epoch_gpu_time = epoch_time - epoch_dataloader_time
        
        # update metrics
        total_s3_calls += epoch_s3_calls
        total_fetch_time += epoch_fetch_time
        total_dataloader_time += epoch_dataloader_time
        total_gpu_time += epoch_gpu_time
        total_training_time += epoch_time
        epoch_times.append(epoch_time)
        
        # print epoch summary
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
        print(f"  S3 Calls: {epoch_s3_calls}")
        print(f"  DataLoader Time: {epoch_dataloader_time:.2f}s ({dataloader_percentage:.1f}%)")
        print(f"  GPU Time: {epoch_gpu_time:.2f}s ({gpu_percentage:.1f}%)")
        print(f"  Total Time: {epoch_time:.2f}s")
        print(f"  Training Loss: {avg_loss:.4f}")
    
    # final summary
    print("\n")
    print("----")
    print("Baseline ResNet152 training is complete")
    print("----")
    print(f"Total S3 Calls: {total_s3_calls}")
    print(f"Total DataLoader Time: {total_dataloader_time:.2f}s ({total_dataloader_time/total_training_time*100:.1f}%)")
    print(f"Total GPU Time: {total_gpu_time:.2f}s ({total_gpu_time/total_training_time*100:.1f}%)")
    print(f"Total Training Time: {total_training_time:.2f}s")
    print(f"Average S3 Fetch Time per Image: {total_fetch_time/total_s3_calls:.3f}s")
    print(f"S3 Calls per Second: {total_s3_calls/total_training_time:.1f}")
    
    print(f"\nSimple Time Breakdown:")
    print(f"  DataLoader Time: {total_dataloader_time/total_training_time*100:.1f}% (waiting for data)")
    print(f"  GPU Time: {total_gpu_time/total_training_time*100:.1f}% (actual compute)")
    
    # pipeline activity logged to pipeline_activity.log
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
        's3_calls_per_second': total_s3_calls / total_training_time
    }

if __name__ == "__main__":
    BUCKET_NAME = os.getenv('BUCKET_NAME')
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    NUM_IMAGES = int(os.getenv('NUM_IMAGES', 50000))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
    NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', 3))
    NUM_WORKERS = int(os.getenv('NUM_WORKERS', 4)) 
    
    print("Baseline: pyTorch num_workers prefetching with ResNet152")
    print("This represents the current state-of-the-art with basic prefetching")
    print("----")
    print(f"Using {NUM_WORKERS} background workers for prefetching")
    
    # make dataset 
    dataset = BaselineS3CIFAR10Dataset(
        bucket_name=BUCKET_NAME,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        num_images=NUM_IMAGES
    )

    # configure dataloader
    pin_memory = device.type == 'cuda'
    #when you iterate over the dataloader, itcalls get item(idx) from the dataset
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=pin_memory, 
        persistent_workers=True  
    )
    
    print(f"dataLoader created with {NUM_WORKERS} workers")
    print("background workers will prefetch data while GPU is training")
    
    metrics = train_model(dataloader, num_epochs=NUM_EPOCHS)
    
    print(f"\nBaseline ResNet152 metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Save metrics to log file
    save_metrics_log(metrics)
    print(f"\nMetrics saved to baseline_S3_dataloader_resnet152_metrics.log")
    
    print(f"\nthis baseline uses PyTorch's num_workers={NUM_WORKERS} for prefetching with ResNet152")
