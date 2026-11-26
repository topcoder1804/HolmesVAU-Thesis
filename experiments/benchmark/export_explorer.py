"""Export per-video timelines (GT, dense scores, selections) for the interactive explorer."""
import json, sys
from pathlib import Path
import numpy as np
import samplers as S
import replay as R

ds, feat, gtp, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
gt = R.load_gt(gtp, ds)
sc = S.Scorer(str(Path(__file__).resolve().parents[2] / "holmesvau/ATS/anomaly_scorer.pth"))
meths = {"Uniform": lambda v: S.s_uniform(v, 12), "Top-k": lambda v: S.s_topk(v, 12, sc),
         "ATS": lambda v: S.s_ats(v, 12, sc),
         "Fast-ATS-D": lambda v: S.s_fast_ats_d(v, 12, sc)}
vids = []
for p in sorted(feat.glob("*.npz")):
    if p.stem not in gt:
        continue
    v = S.Video(p)
    s = v.dense_scores(sc)
    info = {}
    S.s_fast_ats(v, 12, sc, joint=True, info=info)
    rec = dict(id=v.name, n=v.n, fps=v.fps, events=gt[v.name],
               score=[round(float(x), 3) for x in s[:: max(1, len(s) // 200)]],
               fast_scored=[int(x) for x in v.grid[info["scored_pos"]]],
               centers=info["centers"],
               sel={m: [int(x) for x in f(v)[0]] for m, f in meths.items()})
    vids.append(rec)
out.write_text(json.dumps(vids, separators=(",", ":")))
print(len(vids), out.stat().st_size // 1024, "KB")
