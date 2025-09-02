import os
import argparse
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from tqdm import tqdm
from pathlib import Path


def authenticate(service_account_json):
    """Authenticate PyDrive2 with a service account."""
    gauth = GoogleAuth()
    # Configure PyDrive2 to use service account credentials directly
    gauth.settings["client_config_backend"] = "service"
    gauth.settings["service_config"] = {
        "client_json_file_path": service_account_json,
        # Provide the key to satisfy PyDrive2 even if not impersonating
        "client_user_email": "drive-bot@web-design-396514.iam.gserviceaccount.com",
    }
    gauth.ServiceAuth()
    drive = GoogleDrive(gauth)
    return drive


def list_files_with_paths(drive, folder_id, prefix=""):
    """Recursively collect all files with their relative paths from a folder."""
    items = []
    query = f"'{folder_id}' in parents and trashed=false"
    for file in drive.ListFile({"q": query, "maxResults": 1000}).GetList():
        if file["mimeType"] == "application/vnd.google-apps.folder":
            sub_prefix = (
                os.path.join(prefix, file["title"]) if prefix else file["title"]
            )
            items += list_files_with_paths(drive, file["id"], sub_prefix)
        else:
            rel_path = os.path.join(prefix, file["title"]) if prefix else file["title"]
            items.append((file, rel_path))
    return items


def download_folder(folder_id, dest, service_account_json):
    drive = authenticate(service_account_json)
    os.makedirs(dest, exist_ok=True)

    print(f"Listing files in folder {folder_id}...")
    files_with_paths = list_files_with_paths(drive, folder_id)
    print(f"Found {len(files_with_paths)} files. Downloading...")

    for file, rel_path in tqdm(files_with_paths, desc="Downloading", unit="file"):
        out_path = os.path.join(dest, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        file.GetContentFile(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Download a full Google Drive folder using a service account"
    )
    parser.add_argument("folder_id", help="Google Drive folder ID")
    parser.add_argument("output_dir", help="Directory to save files")
    parser.add_argument(
        "--service-account",
        default="service_account.json",
        help="Path to your Google service account JSON key file",
    )
    args = parser.parse_args()

    download_folder(args.folder_id, args.output_dir, args.service_account)


if __name__ == "__main__":
    # also, mkdir -p dataset/
    path = Path("./dataset")
    path.mkdir(exists_ok=True)

    main()
