import argparse
import sys
import os
from minio import Minio 
from minio.error import S3Error
import yaml
import glob

def upload_directory(client, local_path, minio_bucket, minio_path):
    assert os.path.isdir(local_path)

    print("MinIO path " + minio_path)

    for local_file in glob.glob(local_path + '/**'):
        local_file = local_file.replace(os.sep, "/")
        if os.path.isdir(local_file):
            upload_directory(client, local_file, minio_bucket, minio_path + "/" + os.path.basename(local_file))
        else:
            remote_path = os.path.join(minio_path, local_file[1 + len(local_path):])
            remote_path = remote_path.replace(os.sep, "/")
            client.fput_object(minio_bucket, remote_path, local_file)

def upload_model_folder(model, minio_url, minio_access_key, minio_secret_key, minio_path):
    print("Connecting to " + minio_url)
    
    client = Minio(minio_url, minio_access_key, minio_secret_key, secure = False)

    tmp = minio_path.split("/")
    minio_bucket = tmp[0]
    minio_filename = tmp[1] + "/1"

    found = client.bucket_exists(minio_bucket)
    if not found:
        client.make_bucket(minio_bucket)
    
    upload_directory(client, model, minio_bucket, minio_filename)

def upload_model_file(model, minio_url, minio_access_key, minio_secret_key, minio_filename):
    print("Connecting to " + minio_url)
    
    client = Minio(minio_url, minio_access_key, minio_secret_key, secure = False)

    found = client.bucket_exists(minio_filename)
    if not found:
        client.make_bucket(minio_filename)

    client.fput_object(minio_filename, minio_filename + ".joblib", model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Input', type = str, help = "Path of the encoder model to be stored in Persisten Volume")
    parser.add_argument('--Input2', type = str, help = "URL of a MinIO instance")
    parser.add_argument('--Input3', type = str, help = "MinIO access key of the MinIO instance") 
    parser.add_argument('--Input4', type = str, help = "MinIO secret key of the MinIO instance") 
    parser.add_argument('--Input5', type = str, help = "Path of the graph deployment yaml file")
    args = parser.parse_args()

    if len(sys.argv) != 11:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if os.path.isdir(args.Input):
        upload_model_folder(args.Input, args.Input2, args.Input3, args.Input4, args.Input5)
    else:
        upload_model_file(args.Input, args.Input2, args.Input3, args.Input4, args.Input5)