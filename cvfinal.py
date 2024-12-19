import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from facenet_pytorch import MTCNN
import torch
import torchvision.models as models
from torchvision import transforms

# Initialize necessary components
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
feature_model = models.resnet50(pretrained=True)
feature_model = torch.nn.Sequential(*list(feature_model.children())[:-1])  # Remove classification layer
feature_model.eval()
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to extract frames
def extract_frames(video_path, output_dir, frame_rate=1, resize_dim=(224, 224)):
    print(f"Extracting frames from video: {video_path}")
    print(f"Saving frames to: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    interval = max(1, fps // frame_rate)
    frame_count = 0
    success, frame = cap.read()
    while success:
        if frame_count % interval == 0:
            resized_frame = cv2.resize(frame, resize_dim)
            frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_path, resized_frame)
        success, frame = cap.read()
        frame_count += 1
    cap.release()

# Function to process frames (face detection)
def process_frames(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for frame_file in os.listdir(input_dir):
        frame_path = os.path.join(input_dir, frame_file)
        try:
            frame = Image.open(frame_path)
            faces, _ = mtcnn.detect(frame)
            if faces is not None:
                for i, box in enumerate(faces):
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    cropped_face = frame.crop((x1, y1, x2, y2)).resize((224, 224))
                    face_filename = os.path.join(output_dir, f"{frame_file}_face_{i}.jpg")
                    cropped_face.save(face_filename)
        except Exception as e:
            print(f"Error processing frame {frame_file}: {e}")

# Augmentation logic
def augment_image(image):
    augmentations = []
    augmentations.append(ImageOps.mirror(image))  # Horizontal flip
    augmentations.append(image.rotate(np.random.uniform(-15, 15)))  # Random rotation
    enhancer = ImageEnhance.Brightness(image)
    augmentations.append(enhancer.enhance(np.random.uniform(0.7, 1.3)))  # Brightness adjustment
    augmentations.append(image.filter(ImageFilter.GaussianBlur(radius=2)))  # Gaussian blur
    return augmentations

def apply_augmentations(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for frame_file in os.listdir(input_dir):
        frame_path = os.path.join(input_dir, frame_file)
        try:
            image = Image.open(frame_path)
            augmented_images = augment_image(image)
            for i, aug_img in enumerate(augmented_images):
                aug_file_name = os.path.join(output_dir, f"{frame_file}_aug_{i}.jpg")
                aug_img.save(aug_file_name)
        except Exception as e:
            print(f"Error augmenting frame {frame_file}: {e}")

# Feature extraction and aggregation
def extract_features(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for frame_file in os.listdir(input_dir):
        frame_path = os.path.join(input_dir, frame_file)
        try:
            frame = Image.open(frame_path).convert("RGB")
            input_tensor = preprocess(frame).unsqueeze(0)
            with torch.no_grad():
                features = feature_model(input_tensor).squeeze().numpy()
            np.save(os.path.join(output_dir, f"{frame_file}.npy"), features)
        except Exception as e:
            print(f"Error extracting features from {frame_file}: {e}")

def aggregate_features(input_dir, output_file):
    features = [np.load(os.path.join(input_dir, f)) for f in sorted(os.listdir(input_dir))]
    np.save(output_file, np.array(features))
