import os
import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import random
import joblib

# Initialize FaceNet and MTCNN
facenet = FaceNet()
detector = MTCNN()

# Paths
base_dir = "C:/Users/patry/.cache/kagglehub/datasets/podgorskip01/face-recognition/versions/1/VGGFace2"

# 25, 10, 8, 5, 3, 2
div = 1

# Functions
def extract_face(image_path, required_size=(224, 224)):
    """Detects a face in an image and resizes it."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(image)
    if results:
        x, y, width, height = results[0]['box']
        x, y = abs(x), abs(y)
        face = image[y:y + height, x:x + width]
        face = cv2.resize(face, required_size)
        return face
    return None

def get_embedding(model, face_pixels):
    """Generates a feature vector for a face image using FaceNet."""
    face_pixels = face_pixels.astype('float32')
    face_pixels = np.expand_dims(face_pixels, axis=0)  # Add batch dimension
    embedding = model.embeddings(face_pixels)
    return embedding.flatten()

def balance_images(people_images, masked_images):
    """Ensure both lists have the same number of images."""
    num_images = min(len(people_images), len(masked_images)) // div  # Use 1/4 of the images
    return random.sample(people_images, num_images), random.sample(masked_images, num_images)

def get_balanced_embeddings(base_dir, label_mapping):
    """Extract embeddings and balance datasets."""
    embeddings = []
    labels = []
    used_folders = []  # Track successfully processed folders
    
    for label, folder in enumerate(label_mapping):
        people_path = os.path.join(base_dir, folder, "people")
        masked_path = os.path.join(base_dir, folder, "people_masked")

        # Check if directories exist
        if not os.path.exists(people_path):
            print(f"Warning: 'people' directory missing for folder {folder}. Skipping.")
            continue
        if not os.path.exists(masked_path):
            print(f"Warning: 'people_masked' directory missing for folder {folder}. Skipping.")
            continue

        # Load image paths
        people_images = [os.path.join(people_path, img) for img in os.listdir(people_path) if img.endswith(('.jpg', '.png'))]
        masked_images = [os.path.join(masked_path, img) for img in os.listdir(masked_path) if img.endswith(('.jpg', '.png'))]

        # Skip if no valid images
        if not people_images or not masked_images:
            print(f"Warning: No valid images in 'people' or 'people_masked' for folder {folder}. Skipping.")
            continue

        # Balance the datasets
        people_images, masked_images = balance_images(people_images, masked_images)

        # Extract embeddings
        for img_path in people_images + masked_images:
            face = extract_face(img_path)
            if face is not None:
                embedding = get_embedding(facenet, face)
                embeddings.append(embedding)
                labels.append(label)
            else:
                print(f"Skipping {img_path}, no face detected.")
        
        used_folders.append(folder)  # Add folder to successfully processed list

    return np.array(embeddings), np.array(labels), used_folders



# Main Code
folders = os.listdir(base_dir)
print(f"Detected folders: {folders}")

if not folders:
    print("Error: No folders detected in base directory. Check your path.")
else:
    print(f"Total folders detected: {len(folders)}")

label_mapping = folders[:50]  # Limit to first 50 classes for simplicity

# Extract balanced embeddings
print("Extracting embeddings...")
embeddings, labels, used_folders = get_balanced_embeddings(base_dir, label_mapping)

# Split into train/test sets
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Initialize an incremental learning model
incremental_model = make_pipeline(
    StandardScaler(),
    SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
)

# Train the model
print("Training SVM (SGDClassifier)...")
incremental_model.fit(X_train, y_train)

# Save the model
model_path = "svm_incremental_model_2.joblib"
joblib.dump(incremental_model, model_path)
print(f"Model saved to: {model_path}")

# Test the classifier
print("Testing classifier...")
y_pred = incremental_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Divider: {div}")
print(f"SVM Classifier Accuracy: {accuracy:.2f}")

# Function to recognize a face
def recognize_face(image_path, model, label_mapping):
    face = extract_face(image_path)
    if face is not None:
        embedding = get_embedding(facenet, face)
        label = model.predict([embedding])
        return label_mapping[label[0]]  # Map back to folder name
    return None

# Example usage for recognizing a face
example_image = os.path.join(base_dir, "n000002", "people", "0001_01.jpg")  # Change to a valid test image path
predicted_label = recognize_face(example_image, incremental_model, label_mapping)
if predicted_label:
    print(f"Predicted Identity: {predicted_label}")
else:
    print("No face detected or unable to recognize.")
