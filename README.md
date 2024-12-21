### **README: Deepfake Video Classifier**

---

## **Project Overview**
The **Deepfake Video Classifier** is a web-based application designed to classify YouTube videos as either real or fake using advanced deep learning techniques. The system leverages a multi-stage pipeline that includes preprocessing, feature extraction, data augmentation, and temporal modeling. The final classification is performed using state-of-the-art models like LSTMs, Transformers, and CViT.

---

## **Features**
- **YouTube Video Input**: Accepts YouTube video URLs for analysis.
- **Preprocessing**: Extracts, resizes, and detects faces in video frames.
- **Data Augmentation**: Enhances data variability with flipping, rotation, brightness adjustment, and Gaussian blur.
- **Feature Extraction**: Uses ResNet-50 to generate frame-level features.
- **Temporal Modeling**: Classifies videos using LSTM, Temporal Transformer, and CViT models.
- **Real-Time Classification**: Provides real-time results with confidence scores via a user-friendly interface.

---

## **Technologies Used**
- **Backend**: Flask, TensorFlow, PyTorch
- **Frontend**: HTML, CSS, JavaScript
- **Preprocessing**: OpenCV, MTCNN
- **Models**: LSTM, Temporal Transformer, CViT, GAN
- **Other Tools**: ResNet-50, yt-dlp, Flask-CORS

---

## **Setup Instructions**

### **1. Prerequisites**
- Python 3.8 or above
- Node.js (optional for frontend)
- Create a virtual environment
- Libraries: Install required Python libraries using:
  ```bash
  pip install -r requirements.txt
  ```

### **2. Clone the Repository**
```bash
git clone https://github.com/your-repo/deepfake-classifier.git
cd deepfake-classifier
```

### **3. Download Pretrained Models**
- Place pretrained models (e.g., `lstm_model.h5`, `civit_model.pth`) in the project root directory.

### **4. Run the Application**
- Start the Flask server:
  ```bash
  python app.py
  ```
- Access the frontend at `http://127.0.0.1:5000`.

---

## **Usage**

1. **Input a YouTube Video**: Enter a valid YouTube URL in the provided input field.
2. **Processing**: The system downloads the video, extracts frames, processes them, and classifies the video.
3. **Output**: The result (real or fake) and confidence score are displayed on the screen.

---

## **File Structure**

```
Here's a revised directory structure in the same format as the example:


deepfake-classifier/
├── __pycache__/                      # Compiled Python bytecode
├── augmented_frames/                 # Augmented frame data
├── features/                         # Extracted features from video
├── processed_frames/                 # Processed video frames
├── static/                           # Frontend resources (HTML, CSS, JS)
│   └── index.html                    # Web application entry point
├── temp_frames/                      # Temporary frame storage
├── venv/                             # Virtual environment for dependencies
├── aggregated_features...            # Aggregated feature data
├── app.py                            # Flask backend API
├── cv_DeepFake_detection_...         # Deepfake detection script
├── CV_Project.pptx                   # Project presentation
├── cvfinal.py                        # Final script for computer vision
├── lstm_model.h5                     # Pretrained LSTM model
├── new_video.mp4                     # Test video
├── Project_Proposal.pdf              # Proposal document
├── README.md                         # Project documentation
└── requirements.txt                  # Python dependencies

```

---

## **Pipeline Workflow**

1. **Input**: YouTube video URL.
2. **Preprocessing**: Frame extraction, resizing, face detection, and augmentation.
3. **Feature Extraction**: Frame features are extracted using ResNet-50.
4. **Modeling**:
   - **LSTM**: For temporal analysis of sequences.
   - **Temporal Transformer**: For long-range temporal dependencies.
   - **CViT**: Combines vision and transformer techniques.
5. **Output**: Final classification (real or fake) with confidence score.

---
## **Workflow walkthrough**
https://drive.google.com/file/d/1lTy5NC1_TWaiBNgZhWtUOzWWKxZzStQs/view?usp=sharing

## **Future Improvements**
- Add more advanced deepfake detection models.
- Extend to other video formats and platforms.
- Improve processing speed with multi-threading or distributed systems.

---

