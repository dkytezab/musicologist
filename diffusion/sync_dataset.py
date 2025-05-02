import os
import argparse
import boto3
from botocore.exceptions import NoCredentialsError

def sync_directory_to_s3(local_directory,
                         bucket_name,
                         diff_steps = [],
                         s3_prefix=""):
    """
    Syncs a local directory to an S3 bucket.
    Args:
        local_directory: Path to the local directory
        bucket_name: Name of the S3 bucket
        diff_steps: List of diff steps to sync
        s3_prefix: Prefix in the S3 bucket (like a folder path)
    """
    s3_client = boto3.client('s3')

    for diff_step in diff_steps:
        # Create a directory for each diff step
        diff_step_path = os.path.join(local_directory, f"diff_step_{diff_step}")
        if not os.path.exists(diff_step_path):
            print(f"Directory {diff_step_path} does not exist.")
            continue

        for root, dirs, files in os.walk(local_directory):
            for filename in files:
                local_path = os.path.join(root, filename)
                relative_path = os.path.relpath(local_path, local_directory)
                s3_path = os.path.join(s3_prefix, relative_path).replace("\\", "/")

                try:
                    s3_client.upload_file(local_path, bucket_name, s3_path)
                    print(f"Uploaded {local_path} to s3://{bucket_name}/{s3_path}")
                except NoCredentialsError:
                    print("Credentials not available.")
                    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync local directory to S3 bucket")
    parser.add_argument("--local_directory", type=str, default="dataset/" help="Path to the local directory")
    parser.add_argument("--bucket_name", type=str, default="musicologistbucket", help="Name of the S3 bucket")
    parser.add_argument("--s3_prefix", type=str, default="", help="Prefix in the S3 bucket (like a folder path)")

    args = parser.parse_args()

    sync_directory_to_s3(args.local_directory, args.bucket_name, args.s3_prefix)
