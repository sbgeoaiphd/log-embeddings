import argparse
import boto3
import os
from tqdm import tqdm

def download_files(state, year, local_dest):
    # Set up the client
    s3 = boto3.client(
        "s3",
        endpoint_url="https://data.source.coop",
        aws_access_key_id="",
        aws_secret_access_key=""
    )

    bucket = "clay"
    if year:
        prefix = f"clay-v1-5-naip-2/{state}/{year}/"
    else:
        prefix = f"clay-v1-5-naip-2/{state}/"
    # Set local_base using the provided destination folder
    local_base = os.path.join(local_dest, state, year) if year else os.path.join(local_dest, state)

    # First pass: collect all files to download
    print("Listing all files...")
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    all_keys = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if year:
                relative_path = key[len(prefix):]
            else:
                parts = key.split('/')
                # parts: ['clay-v1-5-naip-2', state, actual_year, ...]
                extracted_year = parts[2] if len(parts) > 2 else ""
                local_base = os.path.join(local_dest, state, extracted_year)
                relative_path = os.path.join(*parts[3:]) if len(parts) > 3 else ""
            local_path = os.path.join(local_base, relative_path)
            if not os.path.exists(local_path):
                all_keys.append((key, local_path))

    print(f"{len(all_keys)} files to download.")

    # Download with progress bar
    for key, local_path in tqdm(all_keys, desc="Downloading files"):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket, key, local_path)

def main():
    parser = argparse.ArgumentParser(description="Download embeddings for a specific state and year.")
    parser.add_argument("--state", required=True, help="State code (e.g. 'or').")
    parser.add_argument("--year", help="Year (e.g. '2020'). If not provided, downloads all years.")
    # Optional local destination folder; default is the data directory in the repo
    parser.add_argument("--dest", default="/mnt/c/repos/log-embeddings/data/clay_naip",
                        help="Local base folder for downloads.")
    args = parser.parse_args()

    download_files(args.state, args.year, args.dest)

if __name__ == "__main__":
    main()