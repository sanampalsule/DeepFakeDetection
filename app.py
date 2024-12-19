import os
import logging
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from cvfinal import (
    extract_frames, process_frames, apply_augmentations, extract_features, aggregate_features
)
import yt_dlp
from urllib.parse import urlparse

# Initialize Flask and Model
app = Flask(__name__, static_folder="static")
CORS(app)  # Enable CORS for frontend-backend communication
lstm_model = load_model('lstm_model.h5')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Validate YouTube URLs
def is_valid_youtube_url(url):
    try:
        parsed = urlparse(url)
        return parsed.netloc in ["www.youtube.com", "youtube.com", "youtu.be"]
    except Exception:
        return False

# Download Video
def download_video(url, output_path="new_video.mp4"):
    try:
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': output_path,
            'socket_timeout': 30,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        if os.path.exists(output_path):
            logging.info(f"Downloaded video to {output_path}")
            return output_path
        else:
            raise FileNotFoundError("Video file not found after download.")
    except Exception as e:
        logging.error(f"Error downloading video: {e}")
        return None

@app.route('/')
def home():
    """Serve the HTML frontend."""
    return send_from_directory("static", "index.html")

@app.route('/classify', methods=['POST'])
def classify():
    """Flask API endpoint to classify a YouTube video."""
    try:
        video_url = request.json.get('video_url')
        if not video_url or not is_valid_youtube_url(video_url):
            return jsonify({"error": "Invalid YouTube URL"}), 400

        # Create fixed directories
        temp_frames = "temp_frames"
        processed_frames = "processed_frames"
        augmented_frames = "augmented_frames"
        features_dir = "features"

        os.makedirs(temp_frames, exist_ok=True)
        os.makedirs(processed_frames, exist_ok=True)
        os.makedirs(augmented_frames, exist_ok=True)
        os.makedirs(features_dir, exist_ok=True)

        # Log directory creation
        logging.info(f"Frames directory: {temp_frames}")
        logging.info(f"Processed frames directory: {processed_frames}")
        logging.info(f"Augmented frames directory: {augmented_frames}")
        logging.info(f"Features directory: {features_dir}")

        # Step 1: Download video
        logging.info("Downloading video...")
        video_path = download_video(video_url)
        if not video_path:
            return jsonify({"error": "Failed to download video"}), 500

        # Steps 2-6: Preprocess and extract features
        extract_frames(video_path, temp_frames)
        process_frames(temp_frames, processed_frames)
        apply_augmentations(processed_frames, augmented_frames)
        extract_features(augmented_frames, features_dir)
        aggregate_features(features_dir, "aggregated_features.npy")

        # Step 7: Classify video
        features = np.load("aggregated_features.npy")
        features = np.expand_dims(features, axis=0)
        result = lstm_model.predict(features)
        label = "real" if result[0] > 0.5 else "fake"
        confidence = float(result[0])
        logging.info(f"Classification result: {label} (Confidence: {confidence})")

        # Return result
        return jsonify({"result": label, "confidence": confidence})
    except Exception as e:
        logging.error(f"Error during classification: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)