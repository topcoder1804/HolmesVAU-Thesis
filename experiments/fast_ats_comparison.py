"""Compare uniform sampling, ATS, and coarse-to-fine Fast-ATS."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holmesvau.holmesvau_utils import get_index, get_pixel_values, load_model


PROMPT = "Could you specify the anomaly events present in the video?"
EXAMPLE_VIDEOS = (
    "american_fire.mp4",
    "car_accident.mp4",
    "earthquake.mp4",
    "robbery.mp4",
    "superwind.mp4",
)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def uniform_indices(video_reader, count):
    values = get_index(
        bound=None,
        fps=float(video_reader.get_avg_fps()),
        max_frame=len(video_reader) - 1,
        first_idx=0,
        num_segments=count,
    )
    return sorted({int(value) for value in values})


def score_candidates(video_reader, candidates, model, sampler):
    pixel_values, patches = get_pixel_values(video_reader, candidates)
    scores = sampler.get_anomaly_scores(pixel_values, model)
    return np.asarray(scores), pixel_values, patches


def generate_response(
    video_reader, frame_indices, model, tokenizer, generation_config
):
    pixel_values, patches = get_pixel_values(video_reader, frame_indices)
    pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
    question = "".join(
        f"Frame{i + 1}: <image>\n" for i in range(len(patches))
    ) + PROMPT
    response, _ = model.chat(
        tokenizer,
        pixel_values,
        question,
        generation_config,
        num_patches_list=patches,
        history=None,
        return_history=True,
    )
    return response


def density_select(indices, scores, sampler, count):
    positions = sampler.density_aware_sample_from_scores(scores, count)
    return sorted(int(indices[int(position)]) for position in positions)


def select_ats(video_reader, model, sampler, frame_budget, stride):
    candidates = list(range(0, len(video_reader), stride))
    scores, _, _ = score_candidates(video_reader, candidates, model, sampler)
    selected = density_select(candidates, scores, sampler, frame_budget)
    return {
        "selected_indices": selected,
        "candidate_count": len(candidates),
        "coarse_candidate_count": len(candidates),
        "refinement_candidate_count": 0,
        "region_centers": [],
        "anomaly_scores": scores.tolist(),
    }


def choose_centers(candidates, scores, region_count, minimum_gap):
    centers = []
    for position in np.argsort(scores)[::-1]:
        center = int(candidates[int(position)])
        if all(abs(center - previous) >= minimum_gap for previous in centers):
            centers.append(center)
        if len(centers) == region_count:
            break
    return sorted(centers)


def select_fast_ats(
    video_reader,
    model,
    sampler,
    frame_budget,
    coarse_stride,
    refine_stride,
    region_count,
    region_radius_seconds,
    context_budget,
):
    fps = float(video_reader.get_avg_fps())
    coarse_candidates = list(range(0, len(video_reader), coarse_stride))
    coarse_scores, _, _ = score_candidates(
        video_reader, coarse_candidates, model, sampler
    )
    minimum_gap = max(1, int(region_radius_seconds * fps))
    centers = choose_centers(
        coarse_candidates, coarse_scores, region_count, minimum_gap
    )

    radius = int(region_radius_seconds * fps)
    refinement_candidates = set(coarse_candidates)
    for center in centers:
        start = max(0, center - radius)
        end = min(len(video_reader) - 1, center + radius)
        refinement_candidates.update(range(start, end + 1, refine_stride))
    refinement_candidates = sorted(refinement_candidates)

    scores_by_index = dict(zip(coarse_candidates, coarse_scores.tolist()))
    new_candidates = [
        index for index in refinement_candidates if index not in scores_by_index
    ]
    if new_candidates:
        new_scores, _, _ = score_candidates(
            video_reader, new_candidates, model, sampler
        )
        scores_by_index.update(zip(new_candidates, new_scores.tolist()))

    scored_indices = sorted(scores_by_index)
    scored_values = np.asarray(
        [scores_by_index[index] for index in scored_indices]
    )
    anomaly_budget = max(1, frame_budget - context_budget)
    anomaly_indices = density_select(
        scored_indices, scored_values, sampler, anomaly_budget
    )
    context_indices = uniform_indices(video_reader, context_budget)
    selected = sorted(set(anomaly_indices + context_indices))

    for index in uniform_indices(video_reader, frame_budget):
        if len(selected) >= frame_budget:
            break
        if index not in selected:
            selected.append(index)

    return {
        "selected_indices": sorted(selected[:frame_budget]),
        "candidate_count": len(scored_indices),
        "coarse_candidate_count": len(coarse_candidates),
        "refinement_candidate_count": len(new_candidates),
        "region_centers": centers,
        "anomaly_scores": scored_values.tolist(),
    }


def run_method(
    method,
    video_reader,
    model,
    tokenizer,
    generation_config,
    sampler,
    args,
    device,
):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    total_start = time.perf_counter()
    selection_start = time.perf_counter()

    if method == "uniform":
        selection = {
            "selected_indices": uniform_indices(video_reader, args.frame_budget),
            "candidate_count": 0,
            "coarse_candidate_count": 0,
            "refinement_candidate_count": 0,
            "region_centers": [],
            "anomaly_scores": [],
        }
    elif method == "ats":
        selection = select_ats(
            video_reader,
            model,
            sampler,
            args.frame_budget,
            args.dense_stride,
        )
    else:
        selection = select_fast_ats(
            video_reader,
            model,
            sampler,
            args.frame_budget,
            args.fast_coarse_stride,
            args.fast_refine_stride,
            args.fast_region_count,
            args.fast_region_radius,
            args.fast_context_budget,
        )

    synchronize(device)
    selection_seconds = time.perf_counter() - selection_start
    generation_start = time.perf_counter()
    response = generate_response(
        video_reader,
        selection["selected_indices"],
        model,
        tokenizer,
        generation_config,
    )
    synchronize(device)
    generation_seconds = time.perf_counter() - generation_start
    result = {
        "method": method,
        "selected_indices": selection["selected_indices"],
        "selected_timestamps": [
            round(index / float(video_reader.get_avg_fps()), 4)
            for index in selection["selected_indices"]
        ],
        "candidate_count": selection["candidate_count"],
        "coarse_candidate_count": selection["coarse_candidate_count"],
        "refinement_candidate_count": selection[
            "refinement_candidate_count"
        ],
        "region_centers": selection["region_centers"],
        "selection_seconds": round(selection_seconds, 4),
        "generation_seconds": round(generation_seconds, 4),
        "total_seconds": round(time.perf_counter() - total_start, 4),
        "response": response,
    }
    if selection["anomaly_scores"]:
        result["anomaly_score_min"] = float(min(selection["anomaly_scores"]))
        result["anomaly_score_max"] = float(max(selection["anomaly_scores"]))
    if device.type == "cuda":
        result["max_cuda_memory_mb"] = round(
            torch.cuda.max_memory_allocated(device) / 1024**2, 2
        )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default="./examples")
    parser.add_argument("--output-dir", default="./results/fast_ats")
    parser.add_argument("--model-path", default="./ckpts/HolmesVAU-2B")
    parser.add_argument(
        "--sampler-path",
        default="./holmesvau/ATS/anomaly_scorer.pth",
    )
    parser.add_argument("--frame-budget", type=int, default=12)
    parser.add_argument("--dense-stride", type=int, default=16)
    parser.add_argument("--fast-coarse-stride", type=int, default=48)
    parser.add_argument("--fast-refine-stride", type=int, default=16)
    parser.add_argument("--fast-region-count", type=int, default=4)
    parser.add_argument("--fast-region-radius", type=float, default=8.0)
    parser.add_argument("--fast-context-budget", type=int, default=4)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("uniform", "ats", "fast_ats"),
        default=("uniform", "ats", "fast_ats"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = Path(args.video_dir)
    model, tokenizer, generation_config, sampler = load_model(
        args.model_path, args.sampler_path, device
    )

    results = []
    for name in EXAMPLE_VIDEOS:
        video_path = video_dir / name
        if not video_path.exists():
            print(f"Skipping missing video: {video_path}")
            continue
        print(f"\nProcessing {name}")
        video_reader = VideoReader(
            str(video_path), ctx=cpu(0), num_threads=1
        )
        metadata = {
            "video": str(video_path),
            "frame_count": len(video_reader),
            "fps": float(video_reader.get_avg_fps()),
            "duration_seconds": round(
                len(video_reader) / float(video_reader.get_avg_fps()), 4
            ),
        }
        for method in args.methods:
            print(f"  {method}...")
            result = run_method(
                method,
                video_reader,
                model,
                tokenizer,
                generation_config,
                sampler,
                args,
                device,
            )
            result.update(metadata)
            results.append(result)
            print(f"    total: {result['total_seconds']:.2f}s")

    payload = {
        "configuration": vars(args),
        "device": str(device),
        "results": results,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        fields = [
            "video",
            "method",
            "frame_count",
            "duration_seconds",
            "candidate_count",
            "selected_indices",
            "selection_seconds",
            "generation_seconds",
            "total_seconds",
            "max_cuda_memory_mb",
            "response",
        ]
        writer = csv.DictWriter(
            csv_file, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        for result in results:
            row = dict(result)
            row["selected_indices"] = json.dumps(row["selected_indices"])
            writer.writerow(row)
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
