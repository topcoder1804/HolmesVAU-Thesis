"""Stage 4 (CPU): score HolmesVAU explanations against HIVAU-70k references.

judgement   : anomaly/normal accuracy and anomaly-category accuracy
              (category keywords matched in the response vs. the reference)
description : BLEU-1..4, METEOR, ROUGE-L, CIDEr (pycocoevalcap; same family of
              metrics as the Holmes-VAU paper) + mean response length
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

CATS = {
    "Abuse": ["abuse", "abusing", "mistreat", "beating the dog", "cruel"],
    "Arrest": ["arrest", "handcuff", "detain", "apprehend"],
    "Arson": ["arson", "set fire", "setting fire", "ignite", "igniting"],
    "Assault": ["assault", "attack"],
    "Burglary": ["burglary", "burglar", "break-in", "breaking into", "broke into"],
    "Explosion": ["explosion", "explode", "blast", "detonat"],
    "Fighting": ["fight", "fighting", "brawl", "punch", "scuffle"],
    "RoadAccidents": ["accident", "collision", "crash", "collid", "hit by"],
    "Robbery": ["robbery", "robber", "rob ", "robbing", "armed"],
    "Shooting": ["shoot", "shot", "gunfire", "firing", "gun"],
    "Shoplifting": ["shoplift"],
    "Stealing": ["steal", "stole", "theft", "thief", "stolen"],
    "Vandalism": ["vandal", "damag", "destroy"],
    "Riot": ["riot", "protest", "unrest", "mob"],
}
NORMAL = ["no anomaly", "no anomalies", "not any anomal", "normal", "no abnormal"]


def cats_in(text):
    t = text.lower()
    return {c for c, kws in CATS.items() if any(k in t for k in kws)}


def is_normal(text):
    t = text.lower()
    return any(k in t for k in NORMAL) and not t.startswith("an anomaly exists")


def ref_category(ref):
    c = cats_in(ref)
    # references name the category explicitly; road accident / car accident -> RoadAccidents
    return c


def score_judgement(df):
    rows = []
    for _, r in df.iterrows():
        ref_norm = is_normal(r.reference)
        pred_norm = is_normal(r.response)
        rc, pc = ref_category(r.reference), cats_in(r.response)
        rows.append(dict(method=r.method, dataset=r.dataset, video=r.video,
                         binary_correct=int(ref_norm == pred_norm),
                         anomalous=not ref_norm,
                         category_correct=int((not ref_norm) and bool(rc & pc)) if not ref_norm else None))
    j = pd.DataFrame(rows)
    out = j.groupby(["dataset", "method"]).agg(binary_acc=("binary_correct", "mean"),
                                                  n=("binary_correct", "size")).reset_index()
    cat = j[j.anomalous].groupby(["dataset", "method"]).category_correct.mean().rename("category_acc").reset_index()
    out = out.merge(cat, on=["dataset", "method"], how="left")
    allj = j.groupby("method").agg(binary_acc=("binary_correct", "mean"), n=("binary_correct", "size")).reset_index()
    allj["category_acc"] = j[j.anomalous].groupby("method").category_correct.mean().values
    allj["dataset"] = "all"
    return pd.concat([out, allj])[["dataset", "method", "n", "binary_acc", "category_acc"]], j


def score_description(df):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge
    from nltk.translate.meteor_score import meteor_score

    def tok(s):
        return " ".join(re.findall(r"[a-z0-9']+", s.lower()))

    res = []
    for (ds, m), g in list(df.groupby(["dataset", "method"])) + [(("all", m), g) for m, g in df.groupby("method")]:
        gts = {i: [tok(r)] for i, r in enumerate(g.reference)}
        hyp = {i: [tok(r)] for i, r in enumerate(g.response)}
        b, _ = Bleu(4).compute_score(gts, hyp, verbose=0)
        c, _ = Cider().compute_score(gts, hyp)
        rl, _ = Rouge().compute_score(gts, hyp)
        me = sum(meteor_score([gts[i][0].split()], hyp[i][0].split()) for i in gts) / len(gts)
        res.append(dict(dataset=ds, method=m, n=len(g), BLEU1=b[0], BLEU4=b[3], METEOR=me, ROUGE_L=rl,
                        CIDEr=c, words=g.response.str.split().str.len().mean()))
    return pd.DataFrame(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import nltk
    for pkg in ("wordnet", "omw-1.4"):
        nltk.download(pkg, quiet=True)
    df = pd.read_json(a.responses, lines=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jt, jrows = score_judgement(df[df.task == "judgement"])
    jt.to_csv(out / "judgement.csv", index=False)
    jrows.to_csv(out / "judgement_rows.csv", index=False)
    print(jt.round(3).to_string())
    d = df[df.task == "description"]
    if len(d):
        dt = score_description(d)
        dt.to_csv(out / "description.csv", index=False)
        print(dt.round(4).to_string())
