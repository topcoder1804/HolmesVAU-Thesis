"""Stage 1 (GPU): cache HolmesVAU vision-encoder CLS features on a dense grid.

For every test video we decode frames 0, s, 2s, ... (s = --stride, default 16,
the dense stride used by the original ATS), preprocess them exactly like
`holmesvau_utils.get_pixel_values` (448x448 bicubic, ImageNet normalisation,
max_num=1 tile), and store the InternViT CLS token of each frame.

Because the vision encoder processes frames independently, any sampler whose
candidates lie on this grid (uniform-grid ATS, Fast-ATS coarse/refine passes,
top-k, ...) can later be replayed *exactly* on CPU by running only the small
UR-DMU anomaly scorer on the cached features. Per-frame decode and encoder
times are also recorded so the replay can report real seconds, not just counts.

Output per video: <out>/<video_stem>.npz with
  feats  float16 [T, 1024]   CLS features
  idx    int32   [T]         frame indices
  fps, n_frames, decode_s, vit_s
"""

import argparse
import os as _os
_os.environ["MPLBACKEND"] = "Agg"
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from holmesvau.internvl_utils import build_transform  # noqa: E402


def load_vision(model_path, device, dtype):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
    ).eval()
    vision = model.vision_model.to(device)
    del model
    if os.getenv("HOLMES_SDPA", "1") == "1":
        patch_sdpa(vision)
    torch.cuda.empty_cache()
    return vision


def patch_sdpa(vision):
    """Swap InternViT's eager attention for torch SDPA (same maths, fused kernel)."""
    import types
    import torch.nn.functional as F

    def _sdpa_attn(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=self.scale)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

    n = 0
    for m in vision.modules():
        if type(m).__name__ == "InternAttention":
            m._naive_attn = types.MethodType(_sdpa_attn, m)
            n += 1
    print(f"SDPA attention patched into {n} layers")


def decode_worker(videos, stride, batch, q, transform):
    """Streams one preprocessed batch at a time so long videos never sit in RAM."""
    for path in videos:
        try:
            t0 = time.perf_counter()
            vr = VideoReader(str(path), ctx=cpu(0), num_threads=2)
            n, fps = len(vr), float(vr.get_avg_fps())
            idx = list(range(0, n, stride))
            q.put(("start", path, None))
            dec = 0.0
            for i in range(0, len(idx), batch):
                t1 = time.perf_counter()
                frames = vr.get_batch(idx[i:i + batch]).asnumpy()
                # identical to get_pixel_values with max_num=1: one 448x448 tile
                px = torch.stack([
                    transform(Image.fromarray(f).convert("RGB").resize((448, 448)))
                    for f in frames
                ]).half()
                dec += time.perf_counter() - t1
                q.put(("chunk", path, px))
            q.put(("end", path, (idx, n, fps, dec, time.perf_counter() - t0)))
        except Exception as exc:  # corrupted video: record and continue
            q.put(("error", path, str(exc)))
    q.put(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-path", default="./ckpts/HolmesVAU-2B")
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="fp16")  # T4 has no native bf16
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--list", default="", help="optional txt of video basenames to keep")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    videos = sorted(Path(a.video_dir).glob("*.mp4"))
    if a.list:
        keep = {Path(l.split()[0]).stem for l in open(a.list) if l.strip()}
        videos = [v for v in videos if v.stem in keep]
    videos = [v for v in videos if not (out / f"{v.stem}.npz").exists()]
    if a.limit:
        videos = videos[: a.limit]
    print(f"{len(videos)} videos to process -> {out}")

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float16
    vision = load_vision(a.model_path, device, dtype)
    transform = build_transform(448)

    q = queue.Queue(maxsize=3)
    threading.Thread(target=decode_worker, args=(videos, a.stride, a.batch, q, transform), daemon=True).start()

    log = open(out / "_log.jsonl", "a")
    done, t_start = 0, time.perf_counter()
    feats, vit_s, n_fp32 = [], 0.0, 0
    while True:
        item = q.get()
        if item is None:
            break
        kind, path, payload = item
        if kind == "start":
            feats, vit_s = [], 0.0
        elif kind == "chunk":
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                emb = vision(pixel_values=payload.to(device).to(dtype), output_hidden_states=False,
                             return_dict=True).last_hidden_state[:, 0, :].float()
                if not torch.isfinite(emb).all():  # fp16 overflow guard: redo this batch in fp32
                    vision.float()
                    emb = vision(pixel_values=payload.to(device).float(), output_hidden_states=False,
                                 return_dict=True).last_hidden_state[:, 0, :].float()
                    vision.to(dtype)
                    n_fp32 += 1
                feats.append(emb.cpu())
            torch.cuda.synchronize()
            vit_s += time.perf_counter() - t0
        elif kind == "error":
            log.write(json.dumps({"video": path.name, "error": payload}) + "\n")
            log.flush()
            print("ERROR", path.name, payload)
        else:
            idx, n, fps, decode_s, wall_s = payload
            arr = torch.cat(feats).numpy().astype(np.float16)
            np.savez(out / f"{path.stem}.npz", feats=arr, idx=np.asarray(idx, np.int32),
                     fps=fps, n_frames=n, decode_s=decode_s, vit_s=vit_s)
            rec = {"video": path.name, "T": len(idx), "n_frames": n, "fps": fps,
                   "decode_s": round(decode_s, 3), "vit_s": round(vit_s, 3), "fp32_batches": n_fp32}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            done += 1
            el = time.perf_counter() - t_start
            print(f"[{done}/{len(videos)}] {path.name} T={len(idx)} vit={vit_s:.1f}s "
                  f"dec={decode_s:.1f}s elapsed={el/60:.1f}m", flush=True)

if __name__ == "__main__":
    main()
