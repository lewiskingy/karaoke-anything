from array import array
import asyncio
import pytest
from audio_trombone.kany import HEADER_SIZE, KanyPacket
from audio_trombone.models import MediaPacket
from audio_trombone.processors.mdx23c_vocals import MDX23CVocalsProcessor

def packet(sequence: int, value: float = 1.0) -> MediaPacket:
    header=bytearray(HEADER_SIZE); header[:4]=b'KANY'; header[4]=1; header[6]=2; header[7]=1
    header[8:12]=(100).to_bytes(4,'big'); header[12:16]=sequence.to_bytes(4,'big'); header[24:26]=(25).to_bytes(2,'big')
    return MediaPacket.received(bytes(header)+array('f',[value]*50).tobytes(),'127.0.0.1',40000)

@pytest.mark.parametrize('seconds',[0.1,0.3,3.0])
def test_rejects_unvalidated_segment_sizes(seconds):
    with pytest.raises(ValueError, match='one of'): MDX23CVocalsProcessor(segment_seconds=seconds, inference_fn=lambda s,r,c:s)

@pytest.mark.asyncio
async def test_exact_frames_bounded_buffer_and_reset():
    processor=MDX23CVocalsProcessor(segment_seconds=.25, max_buffered_segments=1, inference_fn=lambda s,r,c:array('f',[0]*len(s)))
    await processor.start(); assert [x async for x in processor.process(packet(0))] == []
    for _ in range(30):
        await asyncio.sleep(.002)
        outputs=[x async for x in processor.process(packet(1))]
        if outputs: break
    decoded=KanyPacket.decode(outputs[0].payload); assert decoded.sequence == 0; assert decoded.frames == 25
    await processor.reset(); diagnostics=processor.diagnostics()
    assert diagnostics['buffered_input_packets']==0 and diagnostics['ready_output_packets']==0
    assert processor._previous_tail is None

def test_overlap_crossfade():
    processor=MDX23CVocalsProcessor(segment_seconds=.25, overlap=.25, inference_fn=lambda s,r,c:s)
    processor._previous_tail=array('f',[0.0]*4); output=array('f',[1.0]*8); processor._crossfade(output)
    assert 0 < output[0] < 1 and output[-1] == 1
