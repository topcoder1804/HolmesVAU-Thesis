"""Frame samplers replayed exactly from cached vision-encoder features.

Every sampler returns (selected_frame_indices, n_vit) where n_vit is the number
of distinct frames that had to be passed through the InternViT encoder to make
the decision. That count is the dominant selection cost (see extraction logs).

The UR-DMU scorer is tiny, so it is re-run here on CPU on exactly the subset of
cached CLS features that each sampler would have scored online.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy import interpolate

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from holmesvau.ATS.anomaly_scorer import URDMU  # noqa: E402

TAU = 0.1  # ATS density smoothing constant (Temporal_Sampler.tau)


class Scorer:
    """UR-DMU anomaly scorer.

    eval_mode=False reproduces the released Temporal_Sampler, which never calls
    .eval(): Dropout(p=0.5) inside the transformer stays active at inference
    and scores become stochastic. eval_mode=True is the deterministic fix.
    """

    def __init__(self, ckpt, eval_mode=True, seed=0):
        self.model = URDMU()
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.eval_mode = eval_mode
        self.model.train(not eval_mode)
        self.seed = seed
        self.calls = 0

    @torch.no_grad()
    def __call__(self, feats):
        if len(feats) == 0:
            return np.zeros(0)
        if not self.eval_mode:
            torch.manual_seed(self.seed + self.calls)
        self.calls += 1
        x = torch.from_numpy(np.asarray(feats, np.float32)).unsqueeze(0)
        return self.model(x)["anomaly_scores"][0].numpy().astype(np.float64)


# ----------------------------------------------------------------- helpers
def uniform_frames(n_frames, k):
    """HolmesVAU / InternVL `get_index` with bound=None."""
    if k <= 0:
        return []
    seg = float(n_frames - 1) / k
    return sorted({int(seg / 2 + np.round(seg * i)) for i in range(k)})


def density_positions(scores, k, tau=TAU):
    """ATS density-aware sampling (Temporal_Sampler.density_aware_sample_from_scores).

    Released code returns *all* candidates when sum(scores) < 1; that would hand
    hundreds of frames to the LLM, so for a fixed-budget comparison we fall back
    to uniform positions over the candidates and report how often this happens.
    """
    t = len(scores)
    if t <= k:
        return list(range(t)), False
    if scores.sum() < 1:
        return sorted({int(round(p)) for p in np.linspace(0, t - 1, k)}), True
    s = scores + tau
    cum = np.concatenate([[0.0], np.cumsum(s)])
    f = interpolate.interp1d(cum, np.arange(t + 1), kind="linear", fill_value="extrapolate")
    pos = f(np.linspace(1, np.round(cum[-1]), k))
    return sorted({min(t - 1, max(0, int(p))) for p in pos}), False


def fill_to_budget(selected, n_frames, k):
    sel = sorted(set(selected))
    for f in uniform_frames(n_frames, k * 2):
        if len(sel) >= k:
            break
        if f not in sel:
            sel.append(f)
            sel.sort()
    return sel[:k] if len(sel) > k else sel


# ----------------------------------------------------------------- samplers
class Video:
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.name = Path(npz_path).stem
        self.feats = d["feats"].astype(np.float32)
        self.grid = d["idx"].astype(np.int64)  # frame index of each cached feature
        self.n = int(d["n_frames"])
        self.fps = float(d["fps"])
        T = max(1, len(self.grid))
        self.sec_per_vit = (float(d["vit_s"]) + float(d["decode_s"])) / T
        self.stride = int(self.grid[1] - self.grid[0]) if len(self.grid) > 1 else 16
        self._dense = {}

    def dense_scores(self, scorer):
        key = (scorer.eval_mode, scorer.seed)
        if key not in self._dense:
            self._dense[key] = scorer(self.feats)
        return self._dense[key]


def s_uniform(v, k, **_):
    return uniform_frames(v.n, k), 0


def s_random(v, k, seed=0, **_):
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(v.n, size=min(k, v.n), replace=False).tolist()), 0


def s_topk(v, k, scorer, **_):
    s = v.dense_scores(scorer)
    pos = np.argsort(s)[::-1][:k]
    return sorted(v.grid[pos].tolist()), len(v.grid)


def s_ats(v, k, scorer, stats=None, **_):
    s = v.dense_scores(scorer)
    pos, fb = density_positions(s, k)
    if stats is not None and fb:
        stats["ats_fallback"] = stats.get("ats_fallback", 0) + 1
    return fill_to_budget(v.grid[pos].tolist(), v.n, k), len(v.grid)


def s_topk_diverse(v, k, scorer, **_):
    """Top-k with temporal non-maximum suppression (min gap = duration / 2k)."""
    s = v.dense_scores(scorer)
    gap = v.n / (2 * k)
    chosen = []
    for p in np.argsort(s)[::-1]:
        f = int(v.grid[p])
        if all(abs(f - c) >= gap for c in chosen):
            chosen.append(f)
        if len(chosen) == k:
            break
    return fill_to_budget(chosen, v.n, k), len(v.grid)


def _choose_centers(frames, scores, count, min_gap):
    centers = []
    for p in np.argsort(scores)[::-1]:
        c = int(frames[p])
        if all(abs(c - o) >= min_gap for o in centers):
            centers.append(c)
        if len(centers) == count:
            break
    return sorted(centers)


def s_fast_ats(v, k, scorer, coarse=48, regions=4, radius=8.0, context=4,
               joint=False, early_exit=None, info=None, **_):
    """Coarse-to-fine Fast-ATS (experiments/fast_ats_comparison.py), on the cached grid.

    joint=False   : faithful to the original implementation (refinement frames are
                    scored as a separate sequence).
    joint=True    : coarse + refined frames are re-scored as one ordered sequence
                    (no extra encoder cost, only one extra UR-DMU pass).
    early_exit=th : if the max coarse score < th, skip refinement and use the
                    coarse density selection directly (saves encoder calls on
                    videos that look normal).
    """
    step = max(1, coarse // v.stride)
    cpos = np.arange(0, len(v.grid), step)
    cscore = scorer(v.feats[cpos])
    n_vit = len(cpos)

    if early_exit is not None and cscore.max() < early_exit:
        pos, _ = density_positions(cscore, k - context)
        sel = v.grid[cpos[pos]].tolist() + uniform_frames(v.n, context)
        if info is not None:
            info["exited"] = True
        return fill_to_budget(sel, v.n, k), n_vit

    r = int(radius * v.fps)
    centers = _choose_centers(v.grid[cpos], cscore, regions, max(1, r))
    refine = set(cpos.tolist())
    for c in centers:
        lo, hi = max(0, c - r), min(v.n - 1, c + r)
        refine.update(np.nonzero((v.grid >= lo) & (v.grid <= hi))[0].tolist())
    allpos = np.array(sorted(refine))
    newpos = np.array(sorted(set(allpos.tolist()) - set(cpos.tolist())), dtype=int)
    n_vit += len(newpos)

    if joint:
        allscore = scorer(v.feats[allpos])
    else:
        by = dict(zip(cpos.tolist(), cscore.tolist()))
        if len(newpos):
            by.update(zip(newpos.tolist(), scorer(v.feats[newpos]).tolist()))
        allscore = np.array([by[p] for p in allpos])

    pos, _ = density_positions(allscore, max(1, k - context))
    sel = v.grid[allpos[pos]].tolist() + uniform_frames(v.n, context)
    if info is not None:
        info["scored_pos"] = allpos
        info["scored_val"] = allscore
        info["centers"] = centers
    return fill_to_budget(sel, v.n, k), n_vit


def s_coarse_ats(v, k, scorer, coarse=48, **_):
    """ATS on a sparser uniform grid only (no refinement) – cost baseline."""
    step = max(1, coarse // v.stride)
    cpos = np.arange(0, len(v.grid), step)
    s = scorer(v.feats[cpos])
    pos, _ = density_positions(s, k)
    return fill_to_budget(v.grid[cpos[pos]].tolist(), v.n, k), len(cpos)


def s_fast_ats_d(v, k, scorer, coarse=48, regions=4, radius=8.0, context=4, gap_factor=2.0, **_):
    """Fast-ATS-D: Fast-ATS (joint re-scoring) with diversity-aware final selection.

    The k-context anomaly frames are the highest-scoring scored frames subject to a
    minimum temporal gap of n_frames / (gap_factor * k); `context` uniform frames are
    added for global context. Same encoder cost as Fast-ATS.
    """
    info = {}
    _, n_vit = s_fast_ats(v, k, scorer, coarse=coarse, regions=regions, radius=radius,
                          context=context, joint=True, info=info)
    frames, val = v.grid[info["scored_pos"]], info["scored_val"]
    gap = v.n / (gap_factor * k)
    chosen = []
    for i in np.argsort(val)[::-1]:
        f = int(frames[i])
        if all(abs(f - c) >= gap for c in chosen):
            chosen.append(f)
        if len(chosen) == k - context:
            break
    return fill_to_budget(chosen + uniform_frames(v.n, context), v.n, k), n_vit
