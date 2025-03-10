import asyncio
import streamlit as st
import os
import librosa
import numpy as np
from transformers import pipeline
from ftplib import FTP

# Fix asyncio issue in Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))

# Streamlit UI
st.title("🎵 Sentiment Analysis from FTP Audio Files")

# User inputs for FTP connection
st.sidebar.header("📡 FTP Login")
host = st.sidebar.text_input("Host", "cph.v4one.co.uk")
username = st.sidebar.text_input("Username", "your_username")
password = st.sidebar.text_input("Password", type="password")
remote_path = "/path/to/audio/folders"  # Change based on server

# Function to list folders from FTP
def list_ftp_folders(host, username, password):
    try:
        with FTP(host, timeout=120) as ftp:
            ftp.login(user=username, passwd=password)
            folders = []
            ftp.retrlines("LIST", lambda x: folders.append(x.split()[-1]))
            return [folder for folder in folders if folder.startswith("2025")]
    except Exception as e:
        st.error(f"FTP Connection Error: {e}")
        return []

# Connect and List Available Folders
if st.sidebar.button("🔄 Connect & List Folders"):
    available_dates = list_ftp_folders(host, username, password)
    if available_dates:
        st.session_state["available_dates"] = available_dates
        st.success("✅ Connected! Select a date below.")
    else:
        st.error("No available folders found.")

# Dropdown for Date Selection
if "available_dates" in st.session_state:
    selected_date = st.selectbox("📅 Select a Date", st.session_state["available_dates"])

    if st.button("📥 Download & Analyze"):
        try:
            local_folder = f"temp_audio/{selected_date}"
            os.makedirs(local_folder, exist_ok=True)

            with FTP(host) as ftp:
                ftp.login(user=username, passwd=password)
                remote_folder = f"{remote_path}/{selected_date}"
                ftp.cwd(remote_folder)

                audio_files = []
                ftp.retrlines("LIST", lambda x: audio_files.append(x.split()[-1]))

                if not audio_files:
                    st.warning("⚠ No audio files found in the selected folder.")
                else:
                    st.info(f"📥 Downloading {len(audio_files)} audio files...")

                    # Download files
                    for file in audio_files:
                        local_file_path = os.path.join(local_folder, file)
                        with open(local_file_path, "wb") as f:
                            ftp.retrbinary(f"RETR {file}", f.write)

                    st.success(f"✅ Downloaded {len(audio_files)} files from {selected_date}")

                    # Sentiment Analysis
                    sentiment_model = pipeline("sentiment-analysis")
                    results = []

                    st.info("🧠 Running Sentiment Analysis...")

                    for file in os.listdir(local_folder):
                        file_path = os.path.join(local_folder, file)
                        y, sr = librosa.load(file_path, sr=16000)
                        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)

                        # Mock transcription (Replace with real ASR model)
                        text = "This is a sample transcription"
                        sentiment = sentiment_model(text)

                        results.append({
                            "File": file,
                            "Sentiment": sentiment[0]["label"],
                            "Confidence": round(sentiment[0]["score"], 2)
                        })

                    st.write("### Sentiment Analysis Results")
                    st.table(results)

        except Exception as e:
            st.error(f"❌ Download or analysis failed: {e}")
