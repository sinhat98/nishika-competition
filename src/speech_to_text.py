import torchaudio
from dataclasses import dataclass, field
from asr import load_model as load_asr_model
from vad import load_model as load_vad_model
import numpy as np
from pyannote.core import Segment, Timeline
from tqdm import tqdm
import logging

# ロガーの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class STTSegment:
    """A segment of transcription with timestamps"""
    start: float
    end: float
    text: str

@dataclass
class TranscribeResult:
    text: int
    segments: list[STTSegment]

# Hyper parameters
WINDOW_SECONDS = 20
PADDING = (16000, 8000)

SAMPLE_RATE = 16000

@dataclass
class TranscribeConfig:
    # ASR model configuration
    asr_model_dir: str
    asr_device: str = None
    asr_ctc_weight: float = 0.3
    asr_lm_weight: float = 0.1
    asr_maxlenratio: float = 1.0
    asr_beam_size: int = 20
    asr_nbest: int = 1
    asr_additional_kwargs: dict = field(default_factory=dict)

    # VAD model configuration
    vad_device: str = None
    vad_min_duration_on: float = 0.5
    vad_min_duration_off: float = 1.0
    merge_segments: bool = True

class SpeechToText:
    def __init__(self, config: TranscribeConfig):
        self.config = config
        self.asr_model = self._load_asr_model()
        self.vad_model = self._load_vad_model()
        
    def _load_asr_model(self):
        return load_asr_model(
            self.config.asr_model_dir,
            device=self.config.asr_device,
            ctc_weight=self.config.asr_ctc_weight,
            lm_weight=self.config.asr_lm_weight,
            maxlenratio=self.config.asr_maxlenratio,
            beam_size=self.config.asr_beam_size,
            nbest=self.config.asr_nbest,
            **self.config.asr_additional_kwargs
        )

    def _load_vad_model(self):
        return load_vad_model(
            device=self.config.vad_device,
            min_duration_on=self.config.vad_min_duration_on,
            min_duration_off=self.config.vad_min_duration_off
        )

    def load_audio(self, audio_file):
        #  'waveform' must be provided as a (channel, time) torch Tensor.
        # do resampling
        waveform, sample_rate = torchaudio.load(audio_file)
        if sample_rate != SAMPLE_RATE:
            logger.info(f"Resampling audio from {sample_rate} to {SAMPLE_RATE}")
            waveform = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)(waveform)
        return waveform

    def vad(self, waveform):
        return self.vad_model({'waveform': waveform, 'sample_rate': SAMPLE_RATE})
    
    
    def asr(self, samples):
        return self.asr_model(np.pad(samples, PADDING, mode="constant"))[0][0]

    def merge_segments(self, segments_list, min_duration=10, max_duration=30):
        merged_timeline = Timeline()
        current_segment = None

        for segment in segments_list:
            if current_segment is None:
                current_segment = segment
            else:
                merged_duration = current_segment.end - current_segment.start + segment.duration
                if merged_duration <= max_duration:
                    current_segment = Segment(current_segment.start, segment.end)
                else:
                    while current_segment.duration > max_duration:
                        # Split current segment into max_duration length parts
                        split_segment = Segment(current_segment.start, current_segment.start + max_duration)
                        merged_timeline.add(split_segment)
                        current_segment = Segment(current_segment.start + max_duration, current_segment.end)

                    if current_segment.duration >= min_duration:
                        merged_timeline.add(current_segment)
                    current_segment = segment

        # Add the last segment if it exists
        if current_segment is not None:
            while current_segment.duration > max_duration:
                split_segment = Segment(current_segment.start, current_segment.start + max_duration)
                merged_timeline.add(split_segment)
                current_segment = Segment(current_segment.start + max_duration, current_segment.end)

            merged_timeline.add(current_segment)

        return merged_timeline

    def transcribe(self, audio_file):
        """Interface function to transcribe audio data

        Args:
            audio_file: path to audio file

        Returns:
            TranscribeResult
        """
        
        waveform = self.load_audio(audio_file)
        vad_result = self.vad(waveform)
        
        segments_list = [seg for seg, _ in vad_result.itertracks(yield_label=False)]

        # Merge VAD segments
        if self.config.merge_segments:    
            segments_list = self.merge_segments(segments_list)
        
        logger.info(f"Total number of segments to process: {len(segments_list)}")
        
        # セグメント長の統計情報を計算
        segment_lengths = [seg.end - seg.start for seg in segments_list]
        avg_length = sum(segment_lengths) / len(segment_lengths)
        max_length = max(segment_lengths)
        min_length = min(segment_lengths)
        # 統計情報をログに出力
        logger.info(f"Merged segment length statistics: Avg: {avg_length:.2f}s, Max: {max_length:.2f}s, Min: {min_length:.2f}s")


        fulltext = ""
        segments = []

        # tqdmを使用してプログレスバーを作成
        pbar = tqdm(total=len(segments_list), desc="Transcribing")

        waveform = waveform.squeeze(0).numpy()
        for i, segment in enumerate(segments_list, 1):
            start_sample = int(segment.start * SAMPLE_RATE)
            end_sample = int(segment.end * SAMPLE_RATE)
            audio_segment = waveform[start_sample:end_sample]
            
            text = self.asr(audio_segment)
            
            fulltext += text + " "
            new_segment = STTSegment(
                start=segment.start,
                end=segment.end,
                text=text
            )
            segments.append(new_segment)

            # プログレスバーを更新し、最新のセグメント情報を表示
            pbar.set_postfix_str(f"Latest: {new_segment.start:.2f}s - {new_segment.end:.2f}s")
            pbar.update(1)

            # 10セグメントごとに進捗をログに記録
            if i % 10 == 0:
                logger.info(f"Processed {i}/{len(segments_list)} segments")

        pbar.close()

        logger.info(f"Transcription completed. Total segments processed: {len(segments)}")

        return TranscribeResult(fulltext.strip(), segments)