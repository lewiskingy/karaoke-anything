from array import array

import pytest

from audio_trombone.processors.base import ProcessorCapabilities
from audio_trombone.processors.segmented_inference import SegmentedInferenceProcessor


class ConcreteSegmentedProcessor(SegmentedInferenceProcessor):
    _model_label = "Concrete"
    name = "concrete-segmented"
    description = "minimal concrete processor for exercising base defaults"
    capabilities = ProcessorCapabilities(
        passthrough=False,
        stateful=True,
        can_buffer=True,
        changes_payload=True,
    )


def test_decode_and_validate_is_not_implemented() -> None:
    processor = ConcreteSegmentedProcessor(segment_seconds=1.0, inference_fn=None)
    with pytest.raises(NotImplementedError):
        processor._decode_and_validate(b"abc")


def test_run_inference_is_not_implemented() -> None:
    processor = ConcreteSegmentedProcessor(segment_seconds=1.0, inference_fn=None)
    with pytest.raises(NotImplementedError):
        processor._run_inference(array("f"), 1_000, 2)
