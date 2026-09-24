# download_from_s3.py

import os
from os.path import join
import s3fs

def download_files_from_s3(
    files_to_download,
    s3_subfolder,
    local_output_dir
):
    """
    Download NetCDF files from S3 using EDITO datalab Onyxia environment.

    Parameters:
    - files_to_download: list of filenames
    - s3_subfolder: folder name on S3
    - local_output_dir: full local directory path where files should be saved
    """
    # Onyxia S3 credentials and bucket setup
    onyxia_user_name = os.environ["GIT_USER_NAME"]
    #bucket_name = f"oidc-{onyxia_user_name}" ## USE FOR PRIVATE BUCKETS
    bucket_name = f"project-foccus" ## USE FOR PROJECT BUCKETS
    S3_ENDPOINT_URL = os.environ["S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})

    # Create local dir if it doesn't exist
    os.makedirs(local_output_dir, exist_ok=True)

    # Download each file
    for file in files_to_download:
        s3_path = f"{bucket_name}/{s3_subfolder}/{file}"
        local_path = join(local_output_dir, file)
        # try:
        fs.download(s3_path, local_path)
        #     print(f"✅ Downloaded: {s3_path} → {local_path}")
        # except Exception as e:
        #     print(f"❌ Failed to download {s3_path}: {e}")
