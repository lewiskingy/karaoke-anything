"""Run pinned MDX23C offline on a stereo PCM WAV and write individual stems."""
import argparse
from pathlib import Path
import wave

from audio_trombone.mdx23c import MDX23CAdapter

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path); parser.add_argument("--output-dir", type=Path, default=Path("mdx23c-output"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    import torch
    with wave.open(str(args.input), "rb") as source:
        if source.getnchannels() != 2 or source.getsampwidth() != 2:
            raise ValueError("fixture must be stereo 16-bit PCM WAV")
        rate = source.getframerate(); raw = source.readframes(source.getnframes())
    waveform = torch.frombuffer(bytearray(raw), dtype=torch.int16).to(torch.float32).view(-1, 2).t().unsqueeze(0) / 32768
    adapter = MDX23CAdapter(device=args.device); estimates = adapter.infer(waveform, rate).cpu().clamp(-1, 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(adapter.stems):
        pcm = (estimates[0, index].t().contiguous() * 32767).to(torch.int16).numpy().tobytes()
        with wave.open(str(args.output_dir / f"{stem}.wav"), "wb") as target:
            target.setnchannels(2); target.setsampwidth(2); target.setframerate(rate); target.writeframes(pcm)
    print(f"input: [1, 2, {waveform.shape[-1]}] @ {rate} Hz")
    print(f"output: {tuple(estimates.shape)} stems={adapter.stems}; length trimmed to input")

if __name__ == "__main__": main()
