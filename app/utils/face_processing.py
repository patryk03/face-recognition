import cv2
from keras_facenet import FaceNet
from mtcnn import MTCNN
import numpy as np
from sklearn.preprocessing import normalize


# Initialize FaceNet and MTCNN
facenet = FaceNet()
detector = MTCNN()

def capture_frame():
    """Capture a frame from the webcam."""
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to capture frame.")
        return None
    return frame

def process_face(frame, return_boxes=False):
    """Detect and extract face embedding from a frame."""
    results = detector.detect_faces(frame)
    print("Detection Results:", results)

    if not results:
        print("No face detected.")
        return [] if return_boxes else None

    processed_faces = []
    for result in results:
        confidence = result.get('confidence', 0)
        if confidence < 0.8:  # Adjust the threshold as needed
            print(f"Low confidence detection skipped: {confidence}")
            continue

        x, y, w, h = result['box']
        face = frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (224, 224))

        # Generate embedding
        embedding = facenet.embeddings(np.expand_dims(face_resized, axis=0)).flatten()
        embedding = normalize([embedding])[0]  # L2 normalize
        if return_boxes:
            processed_faces.append((embedding, (x, y, w, h)))
        else:
            return embedding

    return processed_faces if return_boxes else None

