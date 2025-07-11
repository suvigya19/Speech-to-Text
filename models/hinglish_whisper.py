import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pyannote.audio import Pipeline as DiarizationPipeline
import torchaudio
import torchaudio.transforms as T
import os
import time
import json
from tempfile import NamedTemporaryFile
from huggingface_hub import login
login(" TOKEN HERE")

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

import time

def run(audio_path, progress_callback=None, base_name="output"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "Oriserve/Whisper-Hindi2Hinglish-Prime"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=0 if torch.cuda.is_available() else -1,
        generate_kwargs={
            "task": "transcribe",
            "language": "en"
        }
    )

    diarization_pipeline = DiarizationPipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    diarization_result = diarization_pipeline(audio_path)

    waveform, sample_rate = torchaudio.load(audio_path)
    target_sample_rate = 16000
    resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate) if sample_rate != target_sample_rate else None

    all_segments = []
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        all_segments.append((turn, speaker))

    all_segments.sort(key=lambda x: x[0].start)
    total_segments = len(all_segments)
    transcript_segments = []

    start_time = time.time()

    for idx, (segment, speaker) in enumerate(all_segments, 1):
        start_sample = int(segment.start * sample_rate)
        end_sample = int(segment.end * sample_rate)
        chunk = waveform[:, start_sample:end_sample]

        if resampler:
            chunk = resampler(chunk)

        with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_chunk:
            torchaudio.save(tmp_chunk.name, chunk, target_sample_rate)
            temp_wav_path = tmp_chunk.name

        result = pipe(temp_wav_path)
        os.remove(temp_wav_path)

        text = result.get("text", "").strip()
        transcript_segments.append({
            "start": segment.start,
            "end": segment.end,
            "speaker": speaker,
            "text": text
        })

        elapsed = time.time() - start_time
        avg_time_per_segment = elapsed / idx
        remaining_time = avg_time_per_segment * (total_segments - idx)

        if progress_callback:
            progress_callback(idx / total_segments, remaining_time)

    #  DO NOT override base_name — use it as-is
    srt_path, json_path = generate_srt_and_json(transcript_segments, base_name)

    return transcript_segments, srt_path, json_path

