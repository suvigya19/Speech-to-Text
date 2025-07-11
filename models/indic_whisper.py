import os
import json
import streamlit as st
import torch
from transformers import pipeline, AutoProcessor, WhisperForConditionalGeneration, GenerationConfig
from pyannote.audio import Pipeline as DiarizationPipeline
import torchaudio
import torchaudio.transforms as T
from datetime import timedelta

# Helper function to format timestamps in SRT style
def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

# Load diarization pipeline (replace with your HF token if required)
@st.cache_resource(show_spinner=False)
def load_diarization_pipeline():
    # If your pipeline needs a HuggingFace token, pass `use_auth_token="hf_xxx"`
    return DiarizationPipeline.from_pretrained("pyannote/speaker-diarization-3.1")

# Load Whisper model and processor for Indic model
@st.cache_resource(show_spinner=False)
def load_whisper_model_and_processor(model_path, lang_code):
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)

    generation_config = GenerationConfig.from_pretrained(model_path)
    if not hasattr(generation_config, "no_timestamps_token_id"):
        generation_config.no_timestamps_token_id = 50363
    model.generation_config = generation_config

    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=0 if torch.cuda.is_available() else -1,
    )

    if lang_code == 'or':
        forced_ids = processor.get_decoder_prompt_ids(language=None, task="transcribe")
    else:
        forced_ids = processor.get_decoder_prompt_ids(language=lang_code, task="transcribe")
    asr_pipe.model.config.forced_decoder_ids = forced_ids

    return asr_pipe

def main():
    st.title("Indic Whisper + Pyannote Speaker Diarization ASR")

    uploaded_file = st.file_uploader("Upload WAV audio file", type=["wav"])
    if not uploaded_file:
        st.info("Please upload a WAV file to start transcription.")
        return

    model_path = "hindi_models/whisper-large-hi-noldcil"  # Adjust path as needed
    lang_code = "hi"

    diarization_pipeline = load_diarization_pipeline()
    asr_pipeline = load_whisper_model_and_processor(model_path, lang_code)

    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("Running speaker diarization... (this may take a while)")
    diarization_result = diarization_pipeline("temp_audio.wav")

    waveform, sample_rate = torchaudio.load("temp_audio.wav")
    target_sample_rate = 16000
    resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate) if sample_rate != target_sample_rate else None

    segments = []
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        if turn.end - turn.start < 0.5:
            continue
        segments.append((turn, speaker))
    segments.sort(key=lambda x: x[0].start)

    transcriptions = []
    srt_entries = []

    st.info(f"Processing {len(segments)} speaker segments for ASR...")

    for i, (segment, speaker) in enumerate(segments, 1):
        start_sample = int(segment.start * sample_rate)
        end_sample = int(segment.end * sample_rate)
        chunk = waveform[:, start_sample:end_sample]
        if resampler:
            chunk = resampler(chunk)

        # Save segment to temp file
        temp_segment_file = f"segment_{i}.wav"
        torchaudio.save(temp_segment_file, chunk, target_sample_rate)

        # ASR inference
        result = asr_pipeline(temp_segment_file, return_timestamps=False)

        # Clean up temp segment file
        os.remove(temp_segment_file)

        text = result.get("text", "").strip()
        transcriptions.append({
            "start": segment.start,
            "end": segment.end,
            "speaker": speaker,
            "text": text
        })

        srt_entries.append(f"{i}")
        srt_entries.append(f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}")
        srt_entries.append(f"[{speaker.upper()}] {text}")
        srt_entries.append("")  # Blank line after each entry

    # Save SRT and JSON outputs
    base_name = os.path.splitext(uploaded_file.name)[0]
    srt_path = f"{base_name}.srt"
    json_path = f"{base_name}_segments.json"

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_entries))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(transcriptions, f, ensure_ascii=False, indent=2)

    st.success("Transcription completed!")

    st.download_button("Download SRT File", srt_path, file_name=srt_path, mime="text/plain")
    st.download_button("Download JSON File", json_path, file_name=json_path, mime="application/json")

    st.text_area("Transcript JSON", value=json.dumps(transcriptions, indent=2), height=300)

if __name__ == "__main__":
    main()
