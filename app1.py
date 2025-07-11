import streamlit as st
from models import hinglish_whisper, whisper_largev3, indic_whisper
import subprocess
import datetime
import os

def query_llm_with_gemma(transcript_segments):
    prompt = "Identify the names of people from the transcript.\n\n"
    for seg in transcript_segments:
        prompt += f"[{seg['speaker']}] {seg['text']}\n"

    try:
        result = subprocess.run(
            ["ollama", "run", "gemma"],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600
        )

        if result.returncode != 0:
            error_msg = result.stderr.decode("utf-8")
            st.error(f"LLM Error:\n{error_msg}")
            return f"LLM Error: {error_msg}"

        return result.stdout.decode("utf-8").strip()

    except subprocess.TimeoutExpired:
        return "LLM timed out while generating a response."
    except Exception as e:
        return f"Unexpected error while querying LLM: {e}"

st.title("Multi-Model Speech Transcription")

model_choice = st.selectbox("Choose transcription model", [
    "Hinglish-Whisper",
    "Whisper Large-v3",
    "Indic-whisper"
])

uploaded_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])
if uploaded_file:
    st.audio(uploaded_file)

    if st.button("Run transcription"):
        with st.spinner("Processing..."):
            from tempfile import NamedTemporaryFile
            with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.read())
                audio_path = tmp.name

            original_filename = uploaded_file.name
            base_name = os.path.splitext(original_filename)[0]

            if model_choice == "Hinglish-Whisper":
                progress_bar = st.progress(0)
                eta_placeholder = st.empty()

                def update_progress(progress, remaining_seconds):
                    progress_bar.progress(min(progress, 1.0))
                    eta = str(datetime.timedelta(seconds=int(remaining_seconds)))
                    eta_placeholder.text(f"Estimated time remaining: {eta}")

                transcript_segments, srt_path, json_path = hinglish_whisper.run(
                    audio_path, progress_callback=update_progress, base_name=base_name
                )
                progress_bar.empty()
                eta_placeholder.empty()

            elif model_choice == "Whisper Large-v3":
                transcript_segments, srt_path, json_path = whisper_largev3.run(
                    audio_path, base_name=base_name
                )
            elif model_choice == "Indic-whisper":
                transcript_segments, srt_path, json_path = indic_whisper.run(
                    audio_path, base_name=base_name
                )
            else:
                st.error("Unknown model selected")
                st.stop()

        st.success("Done!")
        st.dataframe(transcript_segments)

        with st.spinner("Generating summary with LLM..."):
            llm_response = query_llm_with_gemma(transcript_segments)

        st.subheader("LLM Summary / Response")
        st.write(llm_response)

        with open(srt_path, "rb") as f:
            st.download_button("Download SRT", f, file_name=f"{base_name}.srt")
        with open(json_path, "rb") as f:
            st.download_button("Download JSON", f, file_name=f"{base_name}.json")
