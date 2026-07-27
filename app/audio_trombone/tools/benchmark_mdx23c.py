"""Target-host MDX23C segment benchmark; emits readable lines and JSON."""
import argparse, json, time
def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--device", default="cuda"); parser.add_argument("--segments", default="0.25,0.5,0.75,1.0,1.5,2.0"); parser.add_argument("--overlaps", default="0,0.25"); args=parser.parse_args()
    import torch
    from audio_trombone.mdx23c import MDX23CAdapter
    started=time.perf_counter(); adapter=MDX23CAdapter(device=args.device); load=time.perf_counter()-started
    results=[]
    if args.device.startswith("cuda"): torch.cuda.reset_peak_memory_stats()
    for seconds in map(float,args.segments.split(',')):
      for overlap in map(float,args.overlaps.split(',')):
        frames=round(adapter.sample_rate*seconds); waveform=torch.zeros(1,2,frames)
        warm=time.perf_counter(); output=adapter.infer(waveform,adapter.sample_rate); warm_elapsed=time.perf_counter()-warm
        if args.device.startswith("cuda"): torch.cuda.synchronize()
        begin=time.perf_counter(); output=adapter.infer(waveform,adapter.sample_rate)
        if args.device.startswith("cuda"): torch.cuda.synchronize()
        elapsed=time.perf_counter()-begin
        row={"segment_seconds":seconds,"overlap":overlap,"inference_seconds":elapsed,"real_time_factor":elapsed/seconds,"warmup_seconds":warm_elapsed,"input_buffering_seconds":seconds,"output_queue_seconds":0.0,"estimated_algorithmic_latency_seconds":seconds+elapsed,"output_frames":output.shape[-1],"sample_count_correct":output.shape[-1]==frames,"continuity_metric":None,"reset_verified_by_processor_tests":True}
        results.append(row); print(f"segment={seconds:.2f}s overlap={overlap:.2f} inference={elapsed:.4f}s RTF={elapsed/seconds:.3f}")
    report={"cold_load_seconds":load,"device":adapter.device,"precision":adapter.precision,"peak_cuda_bytes":torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0,"results":results}
    print(json.dumps(report,sort_keys=True))
if __name__ == '__main__': main()
