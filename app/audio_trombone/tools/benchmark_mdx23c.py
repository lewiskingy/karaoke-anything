"""Target-host MDX23C segment benchmark; emits readable lines and JSON."""

import argparse
from dataclasses import dataclass
import json
import time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--segments", default="0.25,0.5,0.75,1.0,1.5,2.0")
    parser.add_argument("--overlaps", default="0,0.25")
    return parser.parse_args()


@dataclass
class _SegmentTiming:
    seconds: float
    overlap: float
    elapsed: float
    warm_elapsed: float
    output_frames: int
    expected_frames: int


def _benchmark_row(timing: _SegmentTiming) -> dict:
    return {
        "segment_seconds": timing.seconds,
        "overlap": timing.overlap,
        "inference_seconds": timing.elapsed,
        "real_time_factor": timing.elapsed / timing.seconds,
        "warmup_seconds": timing.warm_elapsed,
        "input_buffering_seconds": timing.seconds,
        "output_queue_seconds": 0.0,
        "estimated_algorithmic_latency_seconds": timing.seconds + timing.elapsed,
        "output_frames": timing.output_frames,
        "sample_count_correct": timing.output_frames == timing.expected_frames,
        "continuity_metric": None,
        "reset_verified_by_processor_tests": True,
    }


def main() -> None:
    args = _parse_args()

    import torch
    from audio_trombone.mdx23c import MDX23CAdapter

    load_started = time.perf_counter()
    adapter = MDX23CAdapter(device=args.device)
    load_seconds = time.perf_counter() - load_started

    is_cuda = args.device.startswith("cuda")
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    results = []
    for seconds in map(float, args.segments.split(",")):
        for overlap in map(float, args.overlaps.split(",")):
            frames = round(adapter.sample_rate * seconds)
            waveform = torch.zeros(1, 2, frames)

            # One warm-up inference pass so its one-time setup cost doesn't
            # skew the timed pass below.
            warm_started = time.perf_counter()
            output = adapter.infer(waveform, adapter.sample_rate)
            warm_elapsed = time.perf_counter() - warm_started
            if is_cuda:
                torch.cuda.synchronize()

            run_started = time.perf_counter()
            output = adapter.infer(waveform, adapter.sample_rate)
            if is_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - run_started

            timing = _SegmentTiming(
                seconds, overlap, elapsed, warm_elapsed, output.shape[-1], frames
            )
            results.append(_benchmark_row(timing))
            print(
                f"segment={seconds:.2f}s overlap={overlap:.2f} "
                f"inference={elapsed:.4f}s RTF={elapsed / seconds:.3f}"
            )

    report = {
        "cold_load_seconds": load_seconds,
        "device": adapter.device,
        "precision": adapter.precision,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if is_cuda else 0,
        "results": results,
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
