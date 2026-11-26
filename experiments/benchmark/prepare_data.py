"""Download the UCF-Crime and XD-Violence *test* videos and annotations.

Sources (public Hugging Face mirrors of the official releases):
  UCF-Crime    : datasets/jinmang2/ucf_crime  (official zips + official split files)
  XD-Violence  : datasets/jherng/xd-violence  (official test videos + test annotations)

Only the 290 UCF-Crime test videos (140 anomalous + 150 normal) and the 800
XD-Violence test videos are kept on disk; training videos are never extracted.

Usage (Colab):
  python experiments/benchmark/prepare_data.py --root /content/data --dataset ucf
  python experiments/benchmark/prepare_data.py --root /content/data --dataset xd
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

UCF_REPO = "jinmang2/ucf_crime"
XD_REPO = "jherng/xd-violence"
UCF_ZIPS = [
    "Anomaly-Videos-Part-1.zip",
    "Anomaly-Videos-Part-2.zip",
    "Anomaly-Videos-Part-3.zip",
    "Anomaly-Videos-Part-4.zip",
    "Testing_Normal_Videos.zip",
]
UCF_ANN = "UCF_Crimes-Train-Test-Split/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"


def prepare_ucf(root: Path):
    out = root / "ucf" / "videos"
    out.mkdir(parents=True, exist_ok=True)
    ann = hf_hub_download(UCF_REPO, UCF_ANN, repo_type="dataset", local_dir=root / "ucf" / "hf")
    shutil.copy(ann, root / "ucf" / "test_annotations.txt")
    wanted = set()
    for line in open(ann):
        parts = line.split()
        if parts:
            wanted.add(parts[0])
    print(f"UCF-Crime test videos wanted: {len(wanted)}")
    for name in UCF_ZIPS:
        missing = [w for w in wanted if not (out / w).exists()]
        if not missing:
            break
        print(f"Downloading {name} ...")
        path = hf_hub_download(UCF_REPO, name, repo_type="dataset", local_dir=root / "ucf" / "hf")
        with zipfile.ZipFile(path) as zf:
            for member in zf.namelist():
                base = os.path.basename(member)
                if base in wanted and not (out / base).exists():
                    with zf.open(member) as src, open(out / base, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        os.remove(path)  # free Colab disk
    have = sorted(p.name for p in out.glob("*.mp4"))
    print(f"UCF-Crime test videos on disk: {len(have)} / {len(wanted)}")
    miss = sorted(wanted - set(have))
    if miss:
        print("Missing:", miss[:20])


def prepare_xd(root: Path, workers=8):
    """Per-file download with retries (snapshot_download can die on a 416 resume)."""
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import HfApi

    out = root / "xd" / "videos"
    out.mkdir(parents=True, exist_ok=True)
    tmp = root / "xd" / "hf"
    for name in ("data/test_annotations.txt", "data/test_list.txt"):
        f = hf_hub_download(XD_REPO, name, repo_type="dataset", local_dir=tmp)
        shutil.copy(f, root / "xd" / os.path.basename(name))
    files = [x.path for x in HfApi().list_repo_tree(XD_REPO, path_in_repo="data/video/test_videos",
                                                     repo_type="dataset") if x.path.endswith(".mp4")]
    print("XD-Violence test videos listed:", len(files))

    def get(path):
        dst = out / os.path.basename(path)
        if dst.exists() and dst.stat().st_size > 0:
            return True
        for attempt in range(4):
            try:
                # a stale partial file triggers HTTP 416 on resume: retries re-download from scratch
                f = hf_hub_download(XD_REPO, path, repo_type="dataset", local_dir=tmp,
                                    force_download=attempt > 0)
                shutil.move(f, dst)
                return True
            except Exception as exc:
                if attempt == 3:
                    print("FAILED", path, str(exc)[:120])
        return False

    with ThreadPoolExecutor(workers) as ex:
        ok = sum(ex.map(get, files))
    print("XD-Violence test videos on disk:", ok, "/", len(files))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/data")
    ap.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    a = ap.parse_args()
    root = Path(a.root)
    (prepare_ucf if a.dataset == "ucf" else prepare_xd)(root)
