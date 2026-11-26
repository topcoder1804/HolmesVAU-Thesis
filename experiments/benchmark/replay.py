"""Stage 2 (CPU): replay every sampler on cached features and score it.

Metrics
-------
Detection (all test videos, official frame labels)
  UCF-Crime : frame-level ROC-AUC      XD-Violence : frame-level AP
  computed for the dense stride-16 scorer (Holmes-VAU ATS) and for cheaper
  scoring schedules (sparser grids, Fast-ATS coarse+refine), with the encoder
  cost of each.

Evidence (anomalous test videos, budget K frames handed to the LLM)
  event_hit : fraction of ground-truth anomaly events containing >=1 selected frame
  any_hit   : 1 if any selected frame falls inside an anomaly
  precision : fraction of selected frames that fall inside an anomaly
  n_vit     : encoder calls needed for selection;  sel_s : estimated seconds

Outputs: <out>/per_video.csv, detection.csv, summary.json, sweep.csv,
         selections.json (frames each method sends to HolmesVAU)
"""

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import samplers as S


# ----------------------------------------------------------------- ground truth
def load_gt(path, dataset):
    """Returns {video_stem: [(start_frame, end_frame), ...]} (empty list = normal)."""
    gt = {}
    for line in open(path):
        p = line.split()
        if not p:
            continue
        stem = Path(p[0]).stem
        nums = [int(x) for x in (p[2:] if dataset == "ucf" else p[1:])]
        ev = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2) if nums[i] >= 0]
        gt[stem] = ev
    return gt


def frame_labels(n, events):
    y = np.zeros(n, np.int8)
    for s, e in events:
        y[max(0, s):min(n, e + 1)] = 1
    return y


def frame_curve(n, frames, scores, hold=None):
    """Scores at sampled frames -> per-frame curve.

    hold=stride : repeat each score for `stride` frames (standard snippet protocol)
    hold=None   : linear interpolation between scored frames (non-uniform schedules)
    """
    frames = np.asarray(frames)
    scores = np.asarray(scores, np.float64)
    if hold:
        c = np.repeat(scores, hold)[:n]
        if len(c) < n:
            c = np.concatenate([c, np.full(n - len(c), scores[-1])])
        return c
    return np.interp(np.arange(n), frames, scores)


def evidence(sel, events):
    sel = np.asarray(sel)
    inside = np.zeros(len(sel), bool)
    hits = []
    for s, e in events:
        m = (sel >= s) & (sel <= e)
        inside |= m
        hits.append(m.any())
    return dict(event_hit=float(np.mean(hits)), any_hit=float(inside.any()),
                precision=float(inside.mean()) if len(sel) else 0.0)


# ----------------------------------------------------------------- method zoo
def methods(fast_kw):
    return {
        "Uniform": lambda v, k, sc: S.s_uniform(v, k),
        "Random": lambda v, k, sc: _avg_random(v, k),
        "Top-k": lambda v, k, sc: S.s_topk(v, k, sc),
        "Top-k + NMS": lambda v, k, sc: S.s_topk_diverse(v, k, sc),
        "ATS (Holmes-VAU)": lambda v, k, sc: S.s_ats(v, k, sc),
        "Coarse-ATS (s=48)": lambda v, k, sc: S.s_coarse_ats(v, k, sc, coarse=48),
        "Fast-ATS": lambda v, k, sc: S.s_fast_ats(v, k, sc, **fast_kw),
        "Fast-ATS (joint)": lambda v, k, sc: S.s_fast_ats(v, k, sc, joint=True, **fast_kw),
    }


def _avg_random(v, k, seeds=5):
    # returned as a list of selections, averaged in the metric step
    return [S.s_random(v, k, seed=s)[0] for s in range(seeds)], 0


def run(args):
    feat_dir = Path(args.features)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gt = load_gt(args.gt, args.dataset)
    files = sorted(p for p in feat_dir.glob("*.npz"))
    vids = [S.Video(p) for p in files if p.stem in gt]
    print(f"{len(vids)} videos with features and ground truth "
          f"({sum(1 for v in vids if gt[v.name])} anomalous)")

    sc = S.Scorer(args.ckpt, eval_mode=not args.train_mode_scorer)
    fast_kw = dict(coarse=48, regions=4, radius=8.0, context=4)
    zoo = methods(fast_kw)

    # ---------------- detection: AUC / AP vs encoder cost ----------------
    det_rows = []
    labels = np.concatenate([frame_labels(v.n, gt[v.name]) for v in vids])
    metric = roc_auc_score if args.dataset == "ucf" else average_precision_score
    mname = "AUC" if args.dataset == "ucf" else "AP"

    def add_det(name, curves, vit):
        det_rows.append({"schedule": name, mname: 100 * metric(labels, np.concatenate(curves)),
                         "vit_calls": int(vit), "vit_per_video": vit / len(vids)})
        print(f"  {name:32s} {mname}={det_rows[-1][mname]:.2f}  vit/video={vit/len(vids):.1f}")

    print("Detection:")
    add_det("Dense s=16 (ATS scorer)", [frame_curve(v.n, v.grid, v.dense_scores(sc), hold=v.stride) for v in vids],
            sum(len(v.grid) for v in vids))
    if args.train_mode_check:
        tm = [S.Scorer(args.ckpt, eval_mode=False, seed=s) for s in range(3)]
        for i, t in enumerate(tm):
            add_det(f"Dense s=16, released (dropout on) seed{i}",
                    [frame_curve(v.n, v.grid, t(v.feats), hold=v.stride) for v in vids],
                    sum(len(v.grid) for v in vids))
    for s in (32, 48, 64, 96, 128):
        step = s // 16
        curves, vit = [], 0
        for v in vids:
            p = np.arange(0, len(v.grid), step)
            curves.append(frame_curve(v.n, v.grid[p], sc(v.feats[p])))
            vit += len(p)
        add_det(f"Sparse grid s={s}", curves, vit)
    for joint in (False, True):
        curves, vit = [], 0
        for v in vids:
            info = {}
            _, nv = S.s_fast_ats(v, 12, sc, joint=joint, info=info, **fast_kw)
            curves.append(frame_curve(v.n, v.grid[info["scored_pos"]], info["scored_val"]))
            vit += nv
        add_det("Fast-ATS" + (" (joint)" if joint else ""), curves, vit)
    pd.DataFrame(det_rows).to_csv(out / "detection.csv", index=False)

    # ---------------- evidence at fixed budget ----------------
    rows, selections, stats = [], defaultdict(dict), {}
    for k in args.budgets:
        for v in vids:
            ev = gt[v.name]
            for m, fn in zoo.items():
                sel, nvit = fn(v, k, sc)
                if m == "Random":
                    mets = [evidence(s_, ev) for s_ in sel] if ev else [{}]
                    met = {kk: float(np.mean([d[kk] for d in mets])) for kk in mets[0]} if ev else {}
                    sel = sel[0]
                else:
                    met = evidence(sel, ev) if ev else {}
                rows.append(dict(video=v.name, anomalous=bool(ev), k=k, method=m, n_vit=nvit,
                                 sel_s=nvit * v.sec_per_vit, duration_s=v.n / v.fps,
                                 n_selected=len(sel), **met))
                if k == 12:
                    selections[v.name][m] = [int(x) for x in sel]
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_video.csv", index=False)
    (out / "selections.json").write_text(json.dumps(selections))

    summ = {}
    for k in args.budgets:
        d = df[df.k == k]
        a = d[d.anomalous]
        tab = a.groupby("method")[["event_hit", "any_hit", "precision"]].mean() * 100
        tab["vit_per_video"] = d.groupby("method").n_vit.mean()
        tab["sel_s_per_video"] = d.groupby("method").sel_s.mean()
        summ[k] = tab.round(3).reset_index().to_dict(orient="records")
        print(f"\nEvidence @K={k} (anomalous videos, n={a.video.nunique()}):")
        print(tab.sort_values("event_hit", ascending=False).round(2).to_string())

    # ---------------- Fast-ATS parameter sweep @K=12 ----------------
    sweep = []
    grid = itertools.product(args.sweep_coarse, (2, 4, 6), (4.0, 8.0, 12.0), (2, 4, 6), (False, True))
    for coarse, reg, rad, ctx, joint in grid:
        res = []
        for v in vids:
            sel, nvit = S.s_fast_ats(v, 12, sc, coarse=coarse, regions=reg, radius=rad, context=ctx, joint=joint)
            e = evidence(sel, gt[v.name]) if gt[v.name] else {}
            res.append(dict(n_vit=nvit, sel_s=nvit * v.sec_per_vit, **e))
        r = pd.DataFrame(res)
        sweep.append(dict(coarse=coarse, regions=reg, radius=rad, context=ctx, joint=joint,
                          event_hit=100 * r.event_hit.mean(), any_hit=100 * r.any_hit.mean(),
                          precision=100 * r.precision.mean(), vit_per_video=r.n_vit.mean(),
                          sel_s_per_video=r.sel_s.mean()))
    pd.DataFrame(sweep).to_csv(out / "sweep.csv", index=False)

    meta = dict(dataset=args.dataset, n_videos=len(vids), n_anomalous=int(sum(bool(gt[v.name]) for v in vids)),
                total_hours=sum(v.n / v.fps for v in vids) / 3600,
                mean_sec_per_vit=float(np.mean([v.sec_per_vit for v in vids])),
                ats_fallback=stats, scorer_eval_mode=sc.eval_mode)
    (out / "summary.json").write_text(json.dumps(dict(meta=meta, detection=det_rows, evidence=summ), indent=1))
    print("\nSaved to", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=str(Path(__file__).resolve().parents[2] / "holmesvau/ATS/anomaly_scorer.pth"))
    ap.add_argument("--budgets", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--sweep-coarse", type=int, nargs="+", default=[32, 48, 64, 96])
    ap.add_argument("--train-mode-scorer", action="store_true")
    ap.add_argument("--train-mode-check", action="store_true")
    run(ap.parse_args())
