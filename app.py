import streamlit as st
import os
import librosa
import numpy as np
from transformers import pipeline
from ftplib import FTP

# Streamlit UI
st.title("Sentiment Analysis from FTP Audio Files 🎵")

# User inputs for FTP connection
st.sidebar.header("📡 FTP Login")
host = st.sidebar.text_input("Host", "cph.v4one.co.uk")  
username = st.sidebar.text_input("Username", "your_username")
password = st.sidebar.text_input("Password", type="password")

# Connect and List Available Folders
if st.sidebar.button("🔄 Connect & List Folders"):
    try:
        ftp = FTP(host, timeout=120)
        ftp.login(user=username, passwd=password)

        # List available folders
        folders = []
        ftp.retrlines("LIST", lambda x: folders.append(x.split()[-1]))  
        available_dates = [folder for folder in folders if folder.startswith("2025")]

        ftp.quit()

        # Debug: Print available folders
        st.write("Available folders:", available_dates)

        st.session_state["available_dates"] = available_dates
        st.success("✅ Connected! Select a date below.")
    except Exception as e:
        st.error(f"Connection failed: {e}")

# Dropdown for Date Selection
if "available_dates" in st.session_state:
    selected_date = st.selectbox("📅 Select a Date", st.session_state["available_dates"])

    if st.button("📥 Download & Analyze"):
        try:
            ftp = FTP(host)
            ftp.login(user=username, passwd=password)
            
            remote_folder = selected_date  # FIX: Removed remote_path
            ftp.cwd(remote_folder)

            local_folder = f"temp_audio/{selected_date}"
            os.makedirs(local_folder, exist_ok=True)

            audio_files = []
            ftp.retrlines("LIST", lambda x: audio_files.append(x.split()[-1]))  

            # Download files
            for file in audio_files:
                local_file_path = os.path.join(local_folder, file)
                with open(local_file_path, "wb") as f:
                    ftp.retrbinary(f"RETR {file}", f.write)

            ftp.quit()
            st.success(f"✅ Downloaded {len(audio_files)} files from {selected_date}")

            # Sentiment Analysis
            sentiment_model = pipeline("sentiment-analysis")
            results = []

            for file in os.listdir(local_folder):
                file_path = os.path.join(local_folder, file)
                y, sr = librosa.load(file_path, sr=16000)
                mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)

                # Mock transcription (Replace with real ASR model)
                text = "This is a sample transcription"
                sentiment = sentiment_model(text)

                results.append({"File": file, "Sentiment": sentiment[0]["label"], "Confidence": sentiment[0]["score"]})

            st.write("### Sentiment Analysis Results")
            st.table(results)

        except Exception as e:
            st.error(f"Download failed: {e}")
