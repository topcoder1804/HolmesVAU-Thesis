# Phase 1 and Phase 2 experiments

`phase1_phase2.py` runs the Holmes-VAU baseline and compares temporal sampling strategies using the same video, model, prompt, and frame budget.

## Strategies

- `ats`: Holmes-VAU Anomaly-focused Temporal Sampler.
- `uniform`: evenly spaced frames across the complete video.
- `random`: deterministic random frames using `--seed`.
- `fixed_rate`: fixed temporal stride, adjusted to the requested budget.
- `topk`: highest anomaly-score frames from the dense candidate set.

The script saves:

- `results.json`: complete metadata, frame indices, anomaly scores, responses, runtime, and GPU memory.
- `summary.csv`: compact comparison table.

## Colab usage

Run from the repository root:

```bash
MPLBACKEND=Agg /content/holmes-env/bin/python \
  experiments/phase1_phase2.py \
  --video examples/robbery.mp4 \
  --output-dir results/robbery_12frames
```

For a frame-budget ablation:

```bash
for budget in 4 8 12 16 24; do
  MPLBACKEND=Agg /content/holmes-env/bin/python \
    experiments/phase1_phase2.py \
    --video examples/robbery.mp4 \
    --frame-budget "$budget" \
    --output-dir "results/robbery_${budget}frames"
done
```

The default attention mode is standard attention so the code works on a T4 GPU. On an Ampere-or-newer GPU, FlashAttention can be enabled with:

```bash
HOLMESVAU_USE_FLASH_ATTN=1 MPLBACKEND=Agg \
  /content/holmes-env/bin/python experiments/phase1_phase2.py \
  --video examples/robbery.mp4
```

## Fast-ATS comparison

Run uniform sampling, the original ATS, and coarse-to-fine Fast-ATS on all five included example videos:

```bash
PYTHONPATH=. MPLBACKEND=Agg \
  /content/holmes-env/bin/python \
  experiments/fast_ats_comparison.py \
  --video-dir examples \
  --output-dir results/fast_ats
```

Fast-ATS uses a coarse scan, refines four high-scoring regions, and reserves four of the twelve final frames for uniform context. It records separate selection, generation, and total runtimes in `results.json` and `summary.csv`.

Default Fast-ATS settings:

```text
final frame budget: 12
coarse stride: 48
refinement stride: 16
refinement regions: 4
region radius: 8 seconds
uniform context frames: 4
```
