"""Run reproducible Holmes-VAU baseline and temporal-sampling comparisons."""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu

# Allow the script to be launched directly from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holmesvau.holmesvau_utils import (
    generate,
    get_index,
    get_pixel_values,
    load_model,
)


DEFAULT_PROMPT = "Could you specify the anomaly events present in the video?"
SUPPORTED_STRATEGIES = ("ats", "uniform", "random", "fixed_rate", "topk")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def select_uniform(vr, frame_budget):
    indices = get_index(
        bound=None,
        fps=float(vr.get_avg_fps()),
        max_frame=len(vr) - 1,
        first_idx=0,
        num_segments=frame_budget,
    )
    return [int(index) for index in indices]


def select_random(vr, frame_budget, seed):
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(len(vr), size=frame_budget, replace=False).tolist())


def select_fixed_rate(vr, frame_budget):
    stride = max(1, len(vr) // frame_budget)
    indices = list(range(0, len(vr), stride))[:frame_budget]
    if len(indices) < frame_budget:
        indices = select_uniform(vr, frame_budget)
    return indices


def dense_candidates(vr, dense_sample_freq):
    return list(range(len(vr)))[::dense_sample_freq]


def anomaly_candidates(vr, model, sampler, dense_sample_freq, frame_budget):
    candidates = dense_candidates(vr, dense_sample_freq)
    pixel_values, num_patches = get_pixel_values(vr, candidates)
    scores = sampler.get_anomaly_scores(pixel_values, model)
    return candidates, pixel_values, num_patches, np.asarray(scores)


def select_topk(candidates, scores, frame_budget):
    order = np.argsort(scores)[-frame_budget:]
    return sorted(int(candidates[index]) for index in order)


def generate_from_indices(video_path, vr, indices, model, tokenizer, generation_config):
    pixel_values, num_patches = get_pixel_values(vr, indices)
    pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
    question = "".join(
        f"Frame{i + 1}: <image>\n" for i in range(len(num_patches))
    ) + DEFAULT_PROMPT
    response, history = model.chat(
        tokenizer,
        pixel_values,
        question,
        generation_config,
        num_patches_list=num_patches,
        history=None,
        return_history=True,
    )
    return response, history


def run_strategy(
    strategy,
    video_path,
    vr,
    model,
    tokenizer,
    generation_config,
    sampler,
    frame_budget,
    dense_sample_freq,
    seed,
    device,
):
    start = time.perf_counter()
    anomaly_scores = None
    candidate_indices = None

    if strategy == "ats":
        candidate_indices, pixel_values, num_patches, anomaly_scores = (
            anomaly_candidates(
                vr, model, sampler, dense_sample_freq, frame_budget
            )
        )
        sampled = sampler.density_aware_sample_from_scores(
            anomaly_scores, frame_budget
        )
        sampled = [int(index) for index in sampled]
        frame_indices = sorted(
            int(candidate_indices[index]) for index in sampled
        )
        sparse_pixel_values = pixel_values[sampled]
        sparse_num_patches = [num_patches[index] for index in sampled]
        sparse_pixel_values = sparse_pixel_values.to(torch.bfloat16).to(
            model.device
        )
        question = "".join(
            f"Frame{i + 1}: <image>\n"
            for i in range(len(sparse_num_patches))
        ) + DEFAULT_PROMPT
        response, history = model.chat(
            tokenizer,
            sparse_pixel_values,
            question,
            generation_config,
            num_patches_list=sparse_num_patches,
            history=None,
            return_history=True,
        )
    elif strategy == "topk":
        candidate_indices, _, _, anomaly_scores = anomaly_candidates(
            vr, model, sampler, dense_sample_freq, frame_budget
        )
        frame_indices = select_topk(
            candidate_indices, anomaly_scores, frame_budget
        )
        response, history = generate_from_indices(
            video_path, vr, frame_indices, model, tokenizer, generation_config
        )
    elif strategy == "uniform":
        frame_indices = select_uniform(vr, frame_budget)
        response, history = generate_from_indices(
            video_path, vr, frame_indices, model, tokenizer, generation_config
        )
    elif strategy == "random":
        frame_indices = select_random(vr, frame_budget, seed)
        response, history = generate_from_indices(
            video_path, vr, frame_indices, model, tokenizer, generation_config
        )
    elif strategy == "fixed_rate":
        frame_indices = select_fixed_rate(vr, frame_budget)
        response, history = generate_from_indices(
            video_path, vr, frame_indices, model, tokenizer, generation_config
        )
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    synchronize(device)
    elapsed = time.perf_counter() - start
    result = {
        "strategy": strategy,
        "frame_indices": frame_indices,
        "selected_frame_count": len(frame_indices),
        "response": response,
        "elapsed_seconds": round(elapsed, 4),
    }
    if candidate_indices is not None:
        result["candidate_frame_indices"] = candidate_indices
    if anomaly_scores is not None:
        result["anomaly_scores"] = anomaly_scores.tolist()
    if device.type == "cuda":
        result["max_cuda_memory_mb"] = round(
            torch.cuda.max_memory_allocated(device) / 1024**2, 2
        )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model-path", default="./ckpts/HolmesVAU-2B")
    parser.add_argument(
        "--sampler-path",
        default="./holmesvau/ATS/anomaly_scorer.pth",
    )
    parser.add_argument("--output-dir", default="./results/phase1_phase2")
    parser.add_argument("--frame-budget", type=int, default=12)
    parser.add_argument("--dense-sample-freq", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=SUPPORTED_STRATEGIES,
        default=list(SUPPORTED_STRATEGIES),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    video_path = str(Path(args.video))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    model, tokenizer, generation_config, sampler = load_model(
        args.model_path, args.sampler_path, device
    )

    metadata = {
        "video": video_path,
        "frame_count": len(vr),
        "fps": float(vr.get_avg_fps()),
        "duration_seconds": round(len(vr) / float(vr.get_avg_fps()), 4),
        "frame_budget": args.frame_budget,
        "dense_sample_freq": args.dense_sample_freq,
        "seed": args.seed,
        "device": str(device),
        "strategies": args.strategies,
    }
    results = []
    for strategy in args.strategies:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        print(f"Running {strategy}...")
        result = run_strategy(
            strategy,
            video_path,
            vr,
            model,
            tokenizer,
            generation_config,
            sampler,
            args.frame_budget,
            args.dense_sample_freq,
            args.seed,
            device,
        )
        results.append(result)
        print(f"{strategy}: {result['elapsed_seconds']:.2f}s")

    payload = {"metadata": metadata, "results": results}
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "strategy",
                "selected_frame_count",
                "elapsed_seconds",
                "max_cuda_memory_mb",
                "frame_indices",
                "response",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        for result in results:
            row = dict(result)
            row["frame_indices"] = json.dumps(row["frame_indices"])
            writer.writerow(row)
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
