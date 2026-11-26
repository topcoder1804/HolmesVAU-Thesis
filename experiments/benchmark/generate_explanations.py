"""Stage 3 (GPU): HolmesVAU-2B explanations for the frames each sampler chose.

Uses the HIVAU-70k *video-level* test instructions (judgement + description
prompts, with reference answers) for UCF-Crime and XD-Violence test videos.
Frame selections come from replay.py (selections.json), so the language model,
prompt and frame budget are identical across methods; only the evidence differs.

Resumable: every response is appended to <out>.jsonl immediately.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from decord import VideoReader, cpu

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from holmesvau.holmesvau_utils import get_pixel_values  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instructions", default="HIVAU-70k/instruction/merge_instruction_test_final.jsonl")
    ap.add_argument("--selections", nargs="+", required=True, help="selections.json files from replay.py")
    ap.add_argument("--video-dirs", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", default=["Uniform", "ATS (Holmes-VAU)", "Fast-ATS"])
    ap.add_argument("--tasks", nargs="+", default=["judgement", "description"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-path", default="./ckpts/HolmesVAU-2B")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    a = ap.parse_args()

    sel = {}
    for p in a.selections:
        sel.update(json.load(open(p)))
    vpath = {}
    for d in a.video_dirs:
        for p in Path(d).glob("*.mp4"):
            vpath[p.stem] = p

    items = []
    for line in open(a.instructions):
        x = json.loads(line)
        if x["type"] != "video" or x["task"] not in a.tasks:
            continue
        stem = Path(x["video"]).stem
        if stem in sel and stem in vpath:
            items.append(x)
    print(f"{len(items)} video-level instructions with selections and videos")

    out = Path(a.out)
    done = set()
    if out.exists():
        for line in open(out):
            r = json.loads(line)
            done.add((r["id"], r["method"]))

    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(a.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                      use_flash_attn=False, trust_remote_code=True).eval().cuda()
    tok = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True, use_fast=False)
    cfg = dict(max_new_tokens=a.max_new_tokens, do_sample=False)

    f = open(out, "a")
    t_all = time.perf_counter()
    n = 0
    for x in items:
        stem = Path(x["video"]).stem
        vr = None
        for m in a.methods:
            if (x["id"], m) in done or m not in sel[stem]:
                continue
            vr = vr or VideoReader(str(vpath[stem]), ctx=cpu(0), num_threads=1)
            frames = [min(i, len(vr) - 1) for i in sel[stem][m]]
            px, patches = get_pixel_values(vr, frames)
            px = px.to(torch.bfloat16).cuda()
            q = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(patches))) + x["prompt"]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            resp = model.chat(tok, px, q, cfg, num_patches_list=patches, history=None, return_history=False)
            torch.cuda.synchronize()
            rec = dict(id=x["id"], video=stem, dataset=x["video"].split("/")[0], task=x["task"], method=m,
                       prompt=x["prompt"], reference=x["anwser"], response=resp, frames=frames,
                       gen_s=round(time.perf_counter() - t0, 3))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n += 1
            if n % 10 == 0:
                print(f"{n} done, {(time.perf_counter() - t_all) / 60:.1f} min", flush=True)
    print("finished", n)


if __name__ == "__main__":
    main()
