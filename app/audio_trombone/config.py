from dataclasses import dataclass
import os


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
        )