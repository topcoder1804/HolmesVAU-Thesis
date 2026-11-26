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


def prepare_xd(root: Path):
    local = root / "xd" / "hf"
    snapshot_download(
        XD_REPO,
        repo_type="dataset",
        local_dir=local,
        allow_patterns=["data/test_annotations.txt", "data/test_list.txt", "data/video/test_videos/*"],
    )
    vids = root / "xd" / "videos"
    if not vids.exists():
        os.symlink(local / "data" / "video" / "test_videos", vids)
    shutil.copy(local / "data" / "test_annotations.txt", root / "xd" / "test_annotations.txt")
    print("XD-Violence test videos on disk:", len(list(vids.glob("*.mp4"))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/data")
    ap.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    a = ap.parse_args()
    root = Path(a.root)
    (prepare_ucf if a.dataset == "ucf" else prepare_xd)(root)
