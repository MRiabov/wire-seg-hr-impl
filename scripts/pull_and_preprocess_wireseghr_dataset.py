import os
import argparse
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from tqdm import tqdm
from pathlib import Path

thread_local = threading.local()


def _get_thread_drive(service_account_json: str) -> GoogleDrive:
    d = getattr(thread_local, "drive", None)
    if d is None:
        d = authenticate(service_account_json)
        thread_local.drive = d
    return d


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
    params = {
        "q": query,
        "maxResults": 1000,
        # Request only needed fields (Drive API v2 uses 'items')
        "fields": "items(id,title,mimeType,fileSize,md5Checksum),nextPageToken",
    }
    for file in drive.ListFile(params).GetList():
        if file["mimeType"] == "application/vnd.google-apps.folder":
            sub_prefix = (
                os.path.join(prefix, file["title"]) if prefix else file["title"]
            )
            items += list_files_with_paths(drive, file["id"], sub_prefix)
        else:
            rel_path = os.path.join(prefix, file["title"]) if prefix else file["title"]
            size = int(file.get("fileSize", 0)) if "fileSize" in file else 0
            items.append(
                {
                    "id": file["id"],
                    "rel_path": rel_path,
                    "size": size,
                    "md5": file.get("md5Checksum", ""),
                    "mimeType": file["mimeType"],
                }
            )
    return items


def download_folder(folder_id, dest, service_account_json, workers: int):
    drive = authenticate(service_account_json)
    os.makedirs(dest, exist_ok=True)

    print(f"Listing files in folder {folder_id}...")
    files_with_paths = list_files_with_paths(drive, folder_id)
    total = len(files_with_paths)
    print(f"Found {total} files. Planning downloads...")

    # Prepare tasks and skip already downloaded files by size
    tasks = []
    skipped = 0
    for meta in files_with_paths:
        out_path = os.path.join(dest, meta["rel_path"])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if meta["size"] > 0 and os.path.exists(out_path) and os.path.getsize(out_path) == meta["size"]:
            skipped += 1
            continue
        tasks.append((meta["id"], out_path))

    print(f"Skipping {skipped} existing files; {len(tasks)} to download.")

    def _download_one(file_id: str, out_path: str):
        d = _get_thread_drive(service_account_json)
        f = d.CreateFile({"id": file_id})
        f.GetContentFile(out_path)

    if len(tasks) == 0:
        print("All files are up to date.")
        return

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_download_one, fid, path) for fid, path in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading", unit="file"):
            pass


def pull(args=None):
    parser = argparse.ArgumentParser(
        description="Download a full Google Drive folder using a service account"
    )
    parser.add_argument(
        "--folder-id",
        dest="folder_id",
        default="1fgy3wn_yuHEeMNbfiHNVl1-jEdYOfu6p",
        help="Google Drive folder ID",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default="dataset/",
        help="Directory to save files",
    )
    parser.add_argument(
        "--service-account",
        default="secrets/drive-json.json",
        help="Path to your Google service account JSON key file",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel download workers",
    )
    parsed = parser.parse_args(args=args)

    download_folder(
        parsed.folder_id, parsed.output_dir, parsed.service_account, parsed.workers
    )


def _index_numeric_pairs(images_dir: Path, masks_dir: Path):
    assert images_dir.exists() and images_dir.is_dir(), f"Missing images_dir: {images_dir}"
    assert masks_dir.exists() and masks_dir.is_dir(), f"Missing masks_dir: {masks_dir}"
    img_files = sorted([p for p in images_dir.glob("*.jpg") if p.is_file()])
    img_files += sorted([p for p in images_dir.glob("*.jpeg") if p.is_file()])
    assert len(img_files) > 0, f"No .jpg/.jpeg images in {images_dir}"
    ids = []
    for p in img_files:
        stem = p.stem
        assert stem.isdigit(), f"Non-numeric filename encountered: {p.name}"
        ids.append(int(stem))
    ids = sorted(ids)
    pairs = []
    for i in ids:
        ip_jpg = images_dir / f"{i}.jpg"
        ip_jpeg = images_dir / f"{i}.jpeg"
        ip = ip_jpg if ip_jpg.exists() else ip_jpeg
        assert ip.exists(), f"Missing image for {i}: {ip_jpg} or {ip_jpeg}"
        mp = masks_dir / f"{i}.png"
        assert mp.exists(), f"Missing mask for {i}: {mp}"
        pairs.append((ip, mp))
    assert len(pairs) > 0, "No numeric pairs found"
    return pairs


def split_test_train_val(args=None):
    parser = argparse.ArgumentParser(
        description="Split dataset into train/val/test = 85/5/10 with numeric pairs"
    )
    parser.add_argument("--images-dir", required=True, help="Path to images directory")
    parser.add_argument("--masks-dir", required=True, help="Path to masks directory")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output root dir where train/ val/ test/ will be created",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--link-method",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to place files into splits",
    )
    parsed = parser.parse_args(args=args)

    images_dir = Path(parsed.images_dir)
    masks_dir = Path(parsed.masks_dir)
    out_root = Path(parsed.out_dir)
    pairs = _index_numeric_pairs(images_dir, masks_dir)

    n = len(pairs)
    n_train = int(0.85 * n)
    n_val = int(0.05 * n)
    rng = random.Random(parsed.seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train : n_train + n_val]
    test_idx = idxs[n_train + n_val :]

    def _ensure_dirs(root: Path):
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "gts").mkdir(parents=True, exist_ok=True)

    def _place(src: Path, dst: Path):
        if parsed.link_method == "symlink":
            try:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                os.symlink(src, dst)
            except FileExistsError:
                pass
        else:  # copy
            if dst.exists():
                dst.unlink()
            # use hardlink if possible to be fast and space efficient
            try:
                os.link(src, dst)
            except OSError:
                import shutil

                shutil.copy2(src, dst)

    for split_name, split_ids in (
        ("train", train_idx),
        ("val", val_idx),
        ("test", test_idx),
    ):
        root = out_root / split_name
        _ensure_dirs(root)
        for k in split_ids:
            img_p, mask_p = pairs[k]
            (root / "images" / img_p.name).parent.mkdir(parents=True, exist_ok=True)
            (root / "gts" / mask_p.name).parent.mkdir(parents=True, exist_ok=True)
            _place(img_p, root / "images" / img_p.name)
            _place(mask_p, root / "gts" / mask_p.name)
    print(
        f"Split written to {out_root} | train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
    )


if __name__ == "__main__":
    # also, mkdir -p dataset/
    path = Path("./dataset")
    path.mkdir(exist_ok=True)

    # Subcommands
    top = argparse.ArgumentParser(description="WireSegHR data utilities")
    subs = top.add_subparsers(dest="cmd", required=True)

    sp_pull = subs.add_parser("pull", help="Download dataset from Google Drive")
    sp_pull.add_argument("--folder-id", dest="folder_id", default="1fgy3wn_yuHEeMNbfiHNVl1-jEdYOfu6p")
    sp_pull.add_argument("--output-dir", dest="output_dir", default="dataset/")
    sp_pull.add_argument("--service-account", default="secrets/drive-json.json")
    sp_pull.add_argument("--workers", type=int, default=8)

    sp_split = subs.add_parser(
        "split_test_train_val", help="Create 85/5/10 train/val/test split"
    )
    sp_split.add_argument("--images-dir", required=True)
    sp_split.add_argument("--masks-dir", required=True)
    sp_split.add_argument("--out-dir", required=True)
    sp_split.add_argument("--seed", type=int, default=42)
    sp_split.add_argument(
        "--link-method", choices=["symlink", "copy"], default="symlink"
    )

    ns = top.parse_args()
    if ns.cmd == "pull":
        pull([
            "--folder-id",
            ns.folder_id,
            "--output-dir",
            ns.output_dir,
            "--service-account",
            ns.service_account,
            "--workers",
            str(ns.workers),
        ])
    elif ns.cmd == "split_test_train_val":
        split_test_train_val([
            "--images-dir",
            ns.images_dir,
            "--masks-dir",
            ns.masks_dir,
            "--out-dir",
            ns.out_dir,
            "--seed",
            str(ns.seed),
            "--link-method",
            ns.link_method,
        ])
