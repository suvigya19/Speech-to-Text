import streamlit as st
from models import hinglish_whisper, whisper_largev3, indic_whisper
import subprocess
import datetime
import os
import pandas as pd
from io import BytesIO
from pydub import AudioSegment

st.set_page_config(layout="wide")

# CSS with adjusted styling for text size, position, and dropdown alignment
st.markdown("""
    <style>
        body, .stApp, .block-container {
            background-color: white;
            color: black;
        }
        .css-1d391kg, .css-18e3th9, .stMarkdown, label, .stSelectbox label, .stFileUploader label {
            color: black !important;
        }
        div.stButton > button:first-child,
        div.stDownloadButton > button {
            color: white !important;
            background-color: #000000 !important;
            margin: 10px 0 !important;
            width: 200px !important;
            height: 40px !important;
            border-radius: 5px !important;
        }
        /* Style for file uploader and select box containers */
        .stFileUploader, .stSelectbox {
            width: 22rem !important;
            min-height: 6rem !important;
            margin-bottom: 20px !important;
        }
        /* Ensure file uploader and selectbox align properly */
        .stFileUploader > div, .stSelectbox > div {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100%;
        }
        /* Prevent overlap of results */
        .results-container {
            margin-top: 20px !important;
            clear: both !important;
        }
        /* Ensure download buttons are spaced */
        .stDownloadButton {
            margin-right: 10px !important;
            display: inline-block !important;
        }
        /* Adjust title position and size */
        .title-container h1 {
            text-align: center;
            color: black;
            font-size: 2.5em !important;
            margin-top: 50px !important;
        }
        .title-container h3 {
            text-align: center;
            color: black;
            font-size: 1.5em !important;
            margin-top: 10px !important;
        }
        /* Shift dropdown towards right */
        .stSelectbox {
            margin-left: auto !important;
            margin-right: 0 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title Layout with adjusted positioning
col1, col2, col3 = st.columns([2, 6, 2])
with col1:
    st.image("logo1.jpeg", width=300)
with col2:
    st.markdown('<div class="title-container">', unsafe_allow_html=True)
    #st.markdown("<h1 style='text-align: center; color: black;'>LexiVo</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: black; font-size: 50px;'>Speech to Text</h3>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.image("logo2.png", width=300)

# Input Widgets in a Container
with st.container():
    upload_col, model_col = st.columns([1, 1])
    with upload_col:
        uploaded_files = st.file_uploader("**Upload Audio File(s)**", type=["wav"], accept_multiple_files=True)
    with model_col:
        model_choice = st.selectbox("**Select Transcription Model**", ["Hinglish-Whisper (Hinglish)", "Whisper Large-v3 (Hindi+English)", "Indic-whisper (Devnagri)"])

# Run Transcription Button displayed only after upload
if uploaded_files:
    with st.container():
        st.button("Run transcription", key="run_transcription")

# Session state to store results
if "results" not in st.session_state:
    st.session_state.results = []

# Helpers
def query_llm_with_gemma(transcript_segments):
    prompt = '''
You are an expert in analyzing customer-agent phone call transcripts. Given the conversation below, perform the following tasks:

 1. Extract insights in a structured table format with the following fields:

| Field                             | Description |
|----------------------------------|-------------|
| Customer Sentiment               | "positive", "neutral", or "negative" |
| Agent Sentiment                  | "positive", "neutral", or "negative" |
| Sentiment Justification          | Reasoning for both sentiments based on the tone, words, and context of the conversation |
| Summary                          | A detailed summary of 7-8 lines that captures all important points discussed in the call without missing any key information |
| Key Customer Pain Points         | Specific concerns, complaints, or requests raised by the customer |
| Competitor Mentioned             | "Yes" or "No" — If "Yes", SPECIFY the name of the competitor mentioned by the customer |
| Competitor Justification         | Explain why the competitor was mentioned, and include all relevant statements made by the customer about the competitor |
| Agent Self-Introduction/Disclaimer | Whether the agent introduced themselves, shared the company name, gave a disclaimer, and mentioned any ID or role |
| Greetings                        | Mention whether the agent greeted the customer at the beginning of the call |
---
    '''
    for seg in transcript_segments:
        prompt += f"[{seg['speaker']}] {seg['text']}\n"

    try:
        result = subprocess.run(
            ["ollama", "run", "gemma3:27b"],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600
        )
        if result.returncode != 0:
            return f"LLM Error: {result.stderr.decode('utf-8')}"
        return result.stdout.decode("utf-8").strip()
    except subprocess.TimeoutExpired:
        return "LLM timed out while generating a response."
    except Exception as e:
        return f"Unexpected error: {e}"

def format_transcript_as_srt(transcript_segments):
    lines = []
    for idx, seg in enumerate(transcript_segments, start=1):
        def format_time(t):
            td = datetime.timedelta(seconds=float(t))
            return str(td)[:-3].replace('.', ',').zfill(12)
        block = f"{idx}\n{format_time(seg['start'])} --> {format_time(seg['end'])}\n[{seg['speaker']}] {seg['text']}\n"
        lines.append(block)
    return "\n".join(lines)

def get_audio_duration(uploaded_file):
    try:
        audio = AudioSegment.from_file(uploaded_file)
        duration_ms = len(audio)
        duration = datetime.timedelta(milliseconds=duration_ms)
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except Exception as e:
        return "Unknown"

def process_audio(uploaded_file, model_choice):
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    base_name = os.path.splitext(uploaded_file.name)[0]

    if model_choice == "Hinglish-Whisper (Hinglish)":
        progress_bar = st.progress(0)
        eta_placeholder = st.empty()
        def update_progress(progress, remaining_seconds):
            progress_bar.progress(min(progress, 1.0))
            eta = str(datetime.timedelta(seconds=int(remaining_seconds)))
            eta_placeholder.markdown(
                f"<span style='color:black; font-weight:bold;'>Estimated time remaining: {eta}</span>",
                unsafe_allow_html=True
            )

        transcript_segments, srt_path, json_path = hinglish_whisper.run(
            audio_path, progress_callback=update_progress, base_name=base_name
        )
        progress_bar.empty()
        eta_placeholder.empty()
    elif model_choice == "Whisper Large-v3 (Hindi+English)":
        transcript_segments, srt_path, json_path = whisper_largev3.run(
            audio_path, base_name=base_name
        )
    elif model_choice == "Indic-whisper (Devnagri)":
        transcript_segments, srt_path, json_path = indic_whisper.run(
            audio_path, base_name=base_name
        )
    else:
        st.error("Unknown model selected")
        return None, None, None

    try:
        os.remove(audio_path)
    except OSError:
        pass

    return transcript_segments, srt_path, json_path

# Transcribe and Analyze
if uploaded_files and st.session_state.get("run_transcription"):
    with st.container():
        for uploaded_file in uploaded_files:
            st.markdown(f"### Processing: {uploaded_file.name}")
            st.markdown("<br>", unsafe_allow_html=True)

            with st.spinner("Transcribing..."):
                transcript_segments, srt_path, json_path = process_audio(uploaded_file, model_choice)

            if not transcript_segments:
                st.error(f"Transcription failed for: {uploaded_file.name}")
                continue

            st.dataframe(pd.DataFrame(transcript_segments))

            with st.spinner("Generating summary with LLM..."):
                llm_response = query_llm_with_gemma(transcript_segments)

            st.subheader("LLM Summary / Response")
            st.write(llm_response)

            # Calculate audio duration
            uploaded_file.seek(0)  # Reset file pointer
            call_duration = get_audio_duration(uploaded_file)

            st.session_state.results.append({
                "Audio file name": uploaded_file.name,
                "Call Duration": call_duration,
                "Transcript segments": transcript_segments,
                "LLM Summary": llm_response,
            })

            # Download buttons in a row
            col_srt, col_json = st.columns([1, 1])
            with col_srt:
                with open(srt_path, "rb") as f:
                    st.download_button("Download SRT", f, file_name=os.path.basename(srt_path), key=uploaded_file.name+"_srt")
            with col_json:
                with open(json_path, "rb") as f:
                    st.download_button("Download JSON", f, file_name=os.path.basename(json_path), key=uploaded_file.name+"_json")

# Excel download section
if st.session_state.results:
    import re
    def parse_llm_response(text):
        fields = {
            "Customer Sentiment": "", "Agent Sentiment": "", "Sentiment Justification": "",
            "Summary": "", "Key Customer Pain Points": "", "Competitor Mentioned": "",
            "Competitor Justification": "", "Agent Self-Introduction/Disclaimer": "", "Greetings": ""
        }
        for field in fields:
            pattern = re.compile(rf"{field}[:\s|]*([\s\S]*?)(?=\n\S|$)", re.IGNORECASE)
            match = pattern.search(text)
            if match:
                # Remove leading **, |, spaces
                cleaned = match.group(1).strip().lstrip('*| ').rstrip('| ')
                fields[field] = cleaned
        return fields


    rows = []
    for res in st.session_state.results:
        parsed = parse_llm_response(res["LLM Summary"])
        transcript_str = format_transcript_as_srt(res["Transcript segments"])
        row = {
            "Audio file name": res["Audio file name"],
            "Call Duration": res["Call Duration"],
            "Transcription": transcript_str
        }
        row.update(parsed)
        rows.append(row)

    df_results = pd.DataFrame(rows)
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_results.to_excel(writer, index=False, sheet_name="Summary")
    output.seek(0)

    with st.container():
        st.download_button(
            label="Download Excel Summary",
            data=output,
            file_name="transcription_summary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="excel_download"
        )
