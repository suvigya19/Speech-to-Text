import os
import json
import torch
import time
from math import ceil, floor
from pyannote.audio import Pipeline, Audio
import whisper
from whisper.utils import WriteSRT
from whisper import Whisper


def format_timestamp(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"


def generate_srt_and_json(segments, output_prefix):
    srt_path = f"{output_prefix}.srt"
    json_path = f"{output_prefix}.json"

    with open(srt_path, "w", encoding="utf-8") as srt_file:
        for idx, seg in enumerate(segments, 1):
            srt_file.write(f"{idx}\n")
            srt_file.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            srt_file.write(f"[{seg['speaker'].upper()}] {seg['text'].strip()}\n\n")

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(segments, json_file, ensure_ascii=False, indent=2)

    return srt_path, json_path


def diarize_audio(HF_AUTH_TOKEN, AUDIO_FILE):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_AUTH_TOKEN
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))

    io = Audio(mono='downmix', sample_rate=16000)
    waveform, sample_rate = io(AUDIO_FILE)
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    return diarization


class AppendResultsMixin:
    first_call = True
    output_path = ''

    def get_path_and_open_mode(self, *, audio_path: str, dir: str, ext: str, base_name: str):
        if self.first_call:
            self.output_path = os.path.join(dir, base_name + "." + ext)
            self.first_call = False
            mode = 'w'
        else:
            mode = 'a'
        return self.output_path, mode

class WriteSRTIncremental(AppendResultsMixin, WriteSRT):
    srt_index = 1

    def __init__(self, output_dir='.'):
        super().__init__(output_dir=output_dir)


    def __call__(self, result, audio_path, options=None, base_name="output", **kwargs):
        path, mode = self.get_path_and_open_mode(audio_path=audio_path, dir=self.output_dir, ext=self.extension, base_name=base_name)
        with open(path, mode, encoding="utf-8") as f:
            self.write_result(result, file=f, options=options, **kwargs)

    def write_result(self, result, file, options=None, **kwargs):
        speaker = kwargs.get('speaker', 'Speaker')
        for (start, end, text) in self.iterate_result(result, options):
            print_text = f"{self.srt_index}\n{start} --> {end}\n[{speaker}] {text}\n"
            file.write(print_text)
            self.srt_index += 1


class WhisperFacade:
    def __init__(self, model_name: str, quantize=False):
        self.wmodel = whisper.load_model(model_name)
        if quantize:
            DTYPE = torch.qint8
            self.wmodel = torch.quantization.quantize_dynamic(self.wmodel, {torch.nn.Linear}, dtype=DTYPE)

    def _set_timing_for(self, segment, offset):
        segment['start'] += offset
        segment['end'] += offset
        if 'words' in segment:
            for w in segment['words']:
                w['start'] += offset
                w['end'] += offset

    def load_audio(self, file_path):
        self.audio = whisper.load_audio(file_path)

    def transcribe(self, start, end, options):
        SAMPLE_RATE = 16000
        start_index = floor(start * SAMPLE_RATE)
        end_index = ceil(end * SAMPLE_RATE)
        audio_segment = self.audio[start_index:end_index]
        result = whisper.transcribe(self.wmodel, audio_segment, **options)
        for s in result['segments']:
            self._set_timing_for(s, start)
        return result


def run(audio_path, progress_callback=None, base_name="output"):
    HF_AUTH_TOKEN = "YOUR_HF_TOKEN_HERE"

    diarization = diarize_audio(HF_AUTH_TOKEN, audio_path)
    model = WhisperFacade(model_name='large-v3', quantize=False)
    model.load_audio(audio_path)

    writer = WriteSRTIncremental(output_dir='.')

    whisper_options = {
        "verbose": False,
        "word_timestamps": True,
        "task": "transcribe",
        "suppress_tokens": ""
    }

    writer_options = {
        "max_line_width": 55,
        "max_line_count": 2,
        "highlight_words": False
    }

    all_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        all_segments.append((turn, speaker))

    all_segments.sort(key=lambda x: x[0].start)
    total_segments = len(all_segments)
    transcript_segments = []
    start_time = time.time()

    for idx, (turn, speaker) in enumerate(all_segments, 1):
        if turn.end - turn.start < 0.5:
            continue

        result = model.transcribe(start=turn.start, end=turn.end, options=whisper_options)

        writer(result, audio_path, writer_options, base_name=base_name, speaker=speaker)

        for seg in result.get("segments", []):
            transcript_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": speaker,
                "text": seg["text"]
            })

        if progress_callback:
            elapsed = time.time() - start_time
            avg_time_per_segment = elapsed / idx
            remaining_time = avg_time_per_segment * (total_segments - idx)
            progress_callback(idx / total_segments, remaining_time)

    srt_path, json_path = generate_srt_and_json(transcript_segments, base_name)
    return transcript_segments, srt_path, json_path
