import boto3
import pickle
import os
import urllib.request
import tarfile
from botocore.exceptions import ClientError
from dotenv import load_dotenv

def add_labels_to_existing_s3(bucket_name, aws_access_key_id, aws_secret_access_key, region='us-east-1'):
    """Add labels as metadata to existing S3 images"""
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region
    )
    
    if not os.path.exists("cifar-10-python.tar.gz"):
        print("Downloading CIFAR-10 dataset...")
        urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", "cifar-10-python.tar.gz")
    
    if not os.path.exists("cifar-10-batches-py"):
        print("Extracting CIFAR-10 dataset...")
        with tarfile.open("cifar-10-python.tar.gz", "r:gz") as tar:
            tar.extractall()
    
    all_labels = []
    for batch_num in range(1, 6):
        batch_file = f'cifar-10-batches-py/data_batch_{batch_num}'
        print(f"Loading labels from batch {batch_num}...")
        
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        
        labels = batch[b'labels']
        all_labels.extend(labels)
    
    print(f"Loaded {len(all_labels)} total labels from CIFAR-10 dataset")
    
    updated_count = 0
    error_count = 0
    
    for image_id, label in enumerate(all_labels, 1):
        s3_key = f'cifar10/images/{image_id}.raw'
        
        try:

            s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            
            s3_client.copy_object(
                CopySource={'Bucket': bucket_name, 'Key': s3_key},
                Bucket=bucket_name,
                Key=s3_key,
                Metadata={'label': str(label)},
                MetadataDirective='REPLACE'
            )
            
            updated_count += 1
            if updated_count % 100 == 0:
                print(f"Updated {updated_count} images...")
                
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                print(f"Image {image_id} not found, skipping...")
            else:
                print(f"Error updating image {image_id}: {e}")
                error_count += 1
        except Exception as e:
            print(f"Error updating image {image_id}: {e}")
            error_count += 1
    
    print(f"\nSuccessfully updated {updated_count} images with labels")
    print(f"Failed to update {error_count} images")

def verify_labels(bucket_name, aws_access_key_id, aws_secret_access_key, num_samples=5):
    """Verify labels were added correctly"""
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name='us-east-1'
    )
    
    
    for i in range(1, num_samples + 1):
        s3_key = f'cifar10/images/{i}.raw'
        try:
            response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            label = response.get('Metadata', {}).get('label', 'Not found')
            print(f"Image {i}: Label={label}")
        except Exception as e:
            print(f"Image {i}: Error - {e}")

if __name__ == "__main__":
    load_dotenv('config.env')
    
    BUCKET_NAME = os.getenv('BUCKET_NAME')
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    if not all([BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
        exit(1)
    
    print("Adding labels to S3 images")
    
    add_labels_to_existing_s3(BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    verify_labels(BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
