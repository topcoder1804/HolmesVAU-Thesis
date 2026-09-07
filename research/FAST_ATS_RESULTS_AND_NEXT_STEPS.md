# Fast-ATS Experiment Report

## 1. Purpose

This report records the first Fast-ATS experiment for Holmes-VAU. The goal was to reduce the expensive temporal-selection time of the original Anomaly-focused Temporal Sampler (ATS) while keeping the final HolmesVAU-2B explanation comparable.

Raw results are stored in:

```text
research/results/fast_ats/results.json
research/results/fast_ats/summary.csv
```

## 2. Experimental setup

- Videos: all five repository examples.
- Model: HolmesVAU-2B.
- GPU: Colab `cuda:0`, Tesla T4.
- Final frame budget: 12.
- Prompt: `Could you specify the anomaly events present in the video?`
- Compared methods: uniform sampling, original ATS, and Fast-ATS.
- Original ATS dense stride: 16 frames.
- Fast-ATS coarse stride: 48 frames.
- Fast-ATS refinement stride: 16 frames.
- Fast-ATS refined regions: 4.
- Fast-ATS region radius: 8 seconds.
- Fast-ATS uniform context frames: 4.

## 3. How Fast-ATS was implemented

The original ATS evaluates a dense candidate sequence across the full video. Each candidate is passed through the Holmes vision encoder, and the anomaly scorer produces a score. This is effective but expensive.

Fast-ATS uses a coarse-to-fine strategy:

1. Sample the full video sparsely using a stride of 48 frames.
2. Run the anomaly scorer on this coarse candidate set.
3. Select up to four high-scoring, temporally separated region centers.
4. Expand each center by 8 seconds.
5. Add frames inside those regions using a stride of 16 frames.
6. Score the new candidates.
7. Use density-aware selection for the anomaly-focused portion.
8. Reserve four final frames for uniform global context.
9. Send the final 12 frames to HolmesVAU-2B.

This reduces the number of candidates sent through the expensive vision encoder while preserving global context.

## 4. Timing results

| Video | Uniform | Original ATS | Fast-ATS |
|---|---:|---:|---:|
| `american_fire.mp4` | 13.11 s | 210.84 s | 107.18 s |
| `car_accident.mp4` | 13.06 s | 16.17 s | 15.56 s |
| `earthquake.mp4` | 12.43 s | 22.01 s | 22.33 s |
| `robbery.mp4` | 13.14 s | 78.60 s | 68.82 s |
| `superwind.mp4` | 13.37 s | 184.42 s | 98.06 s |

Average total runtime:

| Method | Average runtime |
|---|---:|
| Uniform | 13.02 s |
| Fast-ATS | 62.39 s |
| Original ATS | 102.41 s |

Fast-ATS reduced average ATS runtime by approximately **39%**:

```text
(102.41 - 62.39) / 102.41 ≈ 39.1%
```

The improvement was especially large for:

- `american_fire.mp4`: approximately 49% faster.
- `superwind.mp4`: approximately 47% faster.
- `robbery.mp4`: approximately 12% faster.

For short videos, the original ATS and Fast-ATS were already close to uniform sampling because there were very few candidate frames to score.

## 5. Candidate-count reduction

The number of anomaly-scored candidates was:

| Video | Original ATS | Fast-ATS | Reduction |
|---|---:|---:|---:|
| `american_fire.mp4` | 505 | 229 | 54.7% |
| `car_accident.mp4` | 8 | 8 | 0% |
| `earthquake.mp4` | 26 | 26 | 0% |
| `robbery.mp4` | 171 | 141 | 17.5% |
| `superwind.mp4` | 440 | 207 | 52.9% |

The candidate reduction explains the runtime improvement on long videos. On short videos, the coarse scan already covers nearly all available frames, so Fast-ATS has little opportunity to save work.

## 6. Explanation comparison

### American fire

All three methods identified destructive fire and smoke. Fast-ATS described the event as a wildfire, while ATS and uniform sampling used more general catastrophic-fire wording. The broad event interpretation was consistent.

### Car accident

Uniform sampling identified an explosion, smoke, debris, and people running. ATS emphasized people running from a vehicle. Fast-ATS included a motorbike, explosion, and people running. The explanations were related but not identical, showing that frame selection changes event detail.

### Earthquake

Uniform sampling described people running. ATS and Fast-ATS described a car accident. This demonstrates that the language model's interpretation can vary with the selected evidence and that the example label or event context should be verified before claiming accuracy.

### Robbery

All methods identified suspicious entry or burglary. ATS and Fast-ATS both described a potential burglary or theft, while uniform sampling was less specific. Fast-ATS preserved the correct broad event despite selecting fewer scored candidates than ATS.

### Superwind

All methods consistently identified a tornado. This is the strongest qualitative agreement case.

## 7. Main inferences

### 7.1 Fast-ATS provides a useful efficiency-quality tradeoff

Fast-ATS was substantially faster than original ATS on long videos and generally produced the same broad anomaly interpretation. This supports the hypothesis that coarse-to-fine sampling can reduce computation without immediately destroying explanation quality.

### 7.2 Uniform sampling remains an important baseline

Uniform sampling was approximately 4.8 times faster than Fast-ATS on average. Therefore, Fast-ATS should not be presented as the fastest method overall. Its value is that it adds anomaly-focused selection while retaining more efficiency than original ATS.

The correct thesis question is:

> Does Fast-ATS provide better anomaly evidence or explanation quality than uniform sampling at an acceptable additional cost?

### 7.3 The gain depends on video length

Fast-ATS helps most when videos are long enough for the original dense candidate set to be expensive. It provides little benefit on very short videos.

### 7.4 Runtime was dominated by selection

Generation took roughly 12–14 seconds for all methods. The major difference was anomaly-selection time:

- Uniform selection was effectively free.
- Original ATS spent much longer scoring dense candidates.
- Fast-ATS reduced but did not eliminate that cost.

This confirms that optimizing candidate scoring is the main route to further speedups.

### 7.5 The qualitative results are not formal accuracy measurements

The experiment measured generated descriptions but did not use ground-truth temporal annotations or human ratings. Therefore, it demonstrates feasibility and qualitative consistency, not superiority or statistical accuracy.

## 8. Important implementation issue found

The first run exposed a boundary case: when a video has fewer candidate frames than the requested frame budget, the original density-aware fallback used `linspace` and could return duplicate frame indices.

For example, a short video could produce repeated indices such as:

```text
[0, 16, 16, 32, 48, 48, ...]
```

The sampler fallback has now been corrected to return each available candidate once when the candidate count is less than or equal to the requested budget. This avoids duplicate visual evidence.

The existing uploaded results preserve the original run and should remain as an experiment record. Future runs should use the corrected implementation.

## 9. Limitations

1. Only five videos were tested.
2. No temporal ground truth was used.
3. No human explanation-quality ratings were collected.
4. The same model and prompt were used for all methods.
5. The T4 environment and standard attention affect absolute timing.
6. The current Fast-ATS parameters were chosen heuristically.
7. The experiment does not yet measure anomaly recall.
8. Runtime includes model inference and may vary with GPU allocation.

## 10. Recommended next experiments

### A. Re-run after the duplicate-index fix

Repeat the five-video comparison and preserve the new output separately, for example:

```text
research/results/fast_ats_corrected/
```

### B. Add a diversity-aware baseline

Pure top-k scoring can cluster frames in one short region. Add a method that selects high-score frames subject to a minimum temporal distance.

### C. Tune Fast-ATS parameters

Test:

- Coarse strides: 32, 48, 64.
- Region counts: 2, 4, 6.
- Region radii: 4, 8, 12 seconds.
- Context budgets: 2, 4, 6.

Use the same videos and final 12-frame budget.

### D. Create ground-truth annotations

For each video, record:

```text
anomaly category
start time
end time
evidence description
```

Then evaluate selected-frame recall and temporal IoU.

### E. Separate sampler quality from language quality

Measure whether selected frames overlap the annotated anomaly before asking HolmesVAU to generate a response. This distinguishes a sampler failure from a language-model hallucination.

### F. Test on a larger dataset

Move from repository examples to a controlled subset of UCF-Crime, XD-Violence, or HIVAU-70k.

## 11. Proposed thesis contribution

A defensible thesis contribution is:

> A coarse-to-fine, context-preserving temporal sampler that reduces anomaly-selection cost relative to ATS while maintaining comparable evidence coverage and video-anomaly explanation quality.

The current results support this as a promising hypothesis, not as a final claim.

## 12. Current repository organization

```text
research/
├── FAST_ATS_RESULTS_AND_NEXT_STEPS.md
└── results/
    └── fast_ats/
        ├── results.json
        └── summary.csv
```

This keeps experimental outputs and interpretation separate from the model source code and scripts.

