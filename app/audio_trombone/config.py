import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    listen_host: str = "0.0.0.0"
    input_port: int = 5004
    output_port: int = 5006
    return_host: str | None = None

    http_host: str = "0.0.0.0"
    http_port: int = 8080
    log_level: str = "INFO"

    processor: str = "passthrough"
    input_queue_size: int = 512
    max_packet_size: int = 65535

    demucs_model: str = "htdemucs"
    demucs_device: str = "auto"
    demucs_segment_seconds: float = 6.0
    demucs_overlap: float = 0.25
    demucs_shifts: int = 0
    demucs_vocal_reduction: float = 1.0

    convtasnet_model_path: str = "/models/convtasnet-lyrics-causal"
    convtasnet_device: str = "auto"
    convtasnet_segment_seconds: float = 1.0
    convtasnet_vocal_reduction: float = 1.0
    convtasnet_vocal_source_index: int = 0
    convtasnet_accompaniment_source_index: int = 1

    mdx23c_config_path: str = "/models/mdx23c/model_2_stem_full_band_8k.yaml"
    mdx23c_checkpoint_path: str = "/models/mdx23c/MDX23C-8KFFT-InstVoc_HQ.ckpt"
    mdx23c_device: str = "auto"
    mdx23c_segment_seconds: float = 1.0
    mdx23c_overlap: float = 0.25
    mdx23c_batch_size: int = 1
    mdx23c_vocal_reduction: float = 1.0
    mdx23c_precision: str = "float32"

    centre_reduction: float = 0.7

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            listen_host=os.getenv("LISTEN_HOST", "0.0.0.0"),
            input_port=int(os.getenv("INPUT_PORT", "5004")),
            output_port=int(os.getenv("OUTPUT_PORT", "5006")),
            return_host=os.getenv("RETURN_HOST") or None,
            http_host=os.getenv("HTTP_HOST", "0.0.0.0"),
            http_port=int(os.getenv("HTTP_PORT", "8080")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            processor=os.getenv("PROCESSOR", "passthrough"),
            input_queue_size=int(os.getenv("INPUT_QUEUE_SIZE", "512")),
            max_packet_size=int(os.getenv("MAX_PACKET_SIZE", "65535")),
            demucs_model=os.getenv("DEMUCS_MODEL", "htdemucs"),
            demucs_device=os.getenv("DEMUCS_DEVICE", "auto"),
            demucs_segment_seconds=float(os.getenv("DEMUCS_SEGMENT_SECONDS", "6.0")),
            demucs_overlap=float(os.getenv("DEMUCS_OVERLAP", "0.25")),
            demucs_shifts=int(os.getenv("DEMUCS_SHIFTS", "0")),
            demucs_vocal_reduction=float(os.getenv("DEMUCS_VOCAL_REDUCTION", "1.0")),
            convtasnet_model_path=os.getenv(
                "CONVTASNET_MODEL_PATH", "/models/convtasnet-lyrics-causal"
            ),
            convtasnet_device=os.getenv("CONVTASNET_DEVICE", "auto"),
            convtasnet_segment_seconds=float(
                os.getenv("CONVTASNET_SEGMENT_SECONDS", "1.0")
            ),
            convtasnet_vocal_reduction=float(
                os.getenv("CONVTASNET_VOCAL_REDUCTION", "1.0")
            ),
            convtasnet_vocal_source_index=int(
                os.getenv("CONVTASNET_VOCAL_SOURCE_INDEX", "0")
            ),
            convtasnet_accompaniment_source_index=int(
                os.getenv("CONVTASNET_ACCOMPANIMENT_SOURCE_INDEX", "1")
            ),
            mdx23c_config_path=os.getenv(
                "MDX23C_CONFIG_PATH", "/models/mdx23c/model_2_stem_full_band_8k.yaml"
            ),
            mdx23c_checkpoint_path=os.getenv(
                "MDX23C_CHECKPOINT_PATH", "/models/mdx23c/MDX23C-8KFFT-InstVoc_HQ.ckpt"
            ),
            mdx23c_device=os.getenv("MDX23C_DEVICE", "auto"),
            mdx23c_segment_seconds=float(os.getenv("MDX23C_SEGMENT_SECONDS", "1.0")),
            mdx23c_overlap=float(os.getenv("MDX23C_OVERLAP", "0.25")),
            mdx23c_batch_size=int(os.getenv("MDX23C_BATCH_SIZE", "1")),
            mdx23c_vocal_reduction=float(os.getenv("MDX23C_VOCAL_REDUCTION", "1.0")),
            mdx23c_precision=os.getenv("MDX23C_PRECISION", "float32"),
            centre_reduction=float(os.getenv("CENTRE_REDUCTION", "0.7")),
        )
