import boto3
import pickle
import urllib.request
import tarfile
import os
from dotenv import load_dotenv

def upload_cifar10_to_s3(bucket_name, aws_access_key_id, aws_secret_access_key, region='us-east-1'):
    """Download CIFAR-10 and upload to S3"""
    
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
    
    image_id = 1
    uploaded_count = 0
    
    for batch_num in range(1, 6):
        batch_file = f'cifar-10-batches-py/data_batch_{batch_num}'
        print(f"Processing batch {batch_num}...")
        
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        
        images = batch[b'data']
        
        for i in range(len(images)):
            raw_image_data = images[i].tobytes()
            s3_key = f'cifar10/images/{image_id}.raw'
            
            try:
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=s3_key,
                    Body=raw_image_data,
                    ContentType='application/octet-stream'
                )
                
                uploaded_count += 1
                if uploaded_count % 1000 == 0:
                    print(f"Uploaded {uploaded_count} images...")
                    
            except Exception as e:
                print(f"Error uploading image {image_id}: {e}")
                break
            
            image_id += 1
    
    print(f"Successfully uploaded {uploaded_count} images to S3")

if __name__ == "__main__":
    load_dotenv('config.env')
    
    BUCKET_NAME = os.getenv('BUCKET_NAME')
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    if not all([BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
        print("Missing AWS credentials in config.env")
        exit(1)
    
    print("Uploading CIFAR-10 to S3")
    
    upload_cifar10_to_s3(BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
