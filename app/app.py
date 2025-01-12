import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import KNeighborsClassifier
import joblib
import pickle
from utils.face_processing import process_face

# Paths for model and mapping
MODEL_PATH = "model/classifier.joblib"
MAPPING_PATH = "model/label_mapping.pkl"
DATASET_X_PATH = "model/X_dataset.npy"
DATASET_Y_PATH = "model/y_dataset.npy"

# Global variables for the classifier and dataset
X_dataset = []
y_dataset = []

app = Flask(__name__, template_folder='./templates')

def load_model_and_mapping():
    """Load the KNN classifier, label mapping, and dataset."""
    global X_dataset, y_dataset
    try:
        # Load classifier and label mapping
        classifier = joblib.load(MODEL_PATH)
        with open(MAPPING_PATH, "rb") as f:
            label_mapping = pickle.load(f)

        # Load dataset
        X_dataset = np.load(DATASET_X_PATH, allow_pickle=True).tolist()
        y_dataset = np.load(DATASET_Y_PATH, allow_pickle=True).tolist()

        # Ensure classifier is trained
        if not hasattr(classifier.named_steps['kneighborsclassifier'], 'classes_'):
            print("Classifier not trained. Retraining...")
            X_train = np.array(X_dataset)
            y_train = np.array(y_dataset)
            classifier.named_steps['kneighborsclassifier'].fit(X_train, y_train)
            print("Classifier retrained successfully.")

        print("Model, label mapping, and dataset loaded successfully.")
        return classifier, label_mapping

    except FileNotFoundError:
        # Initialize new model and label mapping if not found
        print("Model or dataset not found. Initializing new model.")
        classifier = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5)
        )
        label_mapping = {}
        X_dataset = []
        y_dataset = []
        return classifier, label_mapping


def save_model_and_mapping(classifier, label_mapping):
    """Save the classifier, label mapping, and dataset."""
    global X_dataset, y_dataset
    joblib.dump(classifier, MODEL_PATH)
    with open(MAPPING_PATH, "wb") as f:
        pickle.dump(label_mapping, f)

    np.save(DATASET_X_PATH, np.array(X_dataset, dtype=object))
    np.save(DATASET_Y_PATH, np.array(y_dataset, dtype=object))
    print("Model, dataset, and label mapping saved successfully.")


def add_new_person(name, embeddings, classifier, label_mapping):
    """Add a new person to the classifier."""
    if not embeddings:
        print("No valid embeddings collected. Skipping addition.")
        return classifier, label_mapping

    global X_dataset, y_dataset

    # Assign a label for the new person
    new_label = len(label_mapping)
    label_mapping[new_label] = name

    # Add embeddings and labels to the dataset
    X_dataset.extend(embeddings)
    y_dataset.extend([new_label] * len(embeddings))

    # Retrain the classifier
    X_train = np.array(X_dataset)
    y_train = np.array(y_dataset)
    classifier.named_steps['kneighborsclassifier'].fit(X_train, y_train)
    print(f"Added {name} to the classifier.")
    return classifier, label_mapping


def detect_face_and_predict(frame, classifier, label_mapping):
    """Detect faces and predict labels."""
    results = process_face(frame, return_boxes=True)
    predictions = []

    for embedding, (x, y, w, h) in results:
        if embedding is not None and embedding.shape == (512,):
            embedding = normalize([embedding])[0]
            try:
                probabilities = classifier.named_steps['kneighborsclassifier'].predict_proba([embedding])[0]
                predicted_class = np.argmax(probabilities)
                label = label_mapping.get(predicted_class, "Unknown")
                confidence = probabilities[predicted_class]

                # Handle low-confidence predictions
                if confidence < 0.8:
                    label = "Unknown"

                predictions.append((label, confidence, (x, y, w, h)))
            except Exception as e:
                print(f"Prediction error: {e}")
                predictions.append(("Unknown", 0.0, (x, y, w, h)))
        else:
            predictions.append(("No face detected", 0.0, (0, 0, 0, 0)))

    return predictions


def gen_frames(classifier, label_mapping):
    """Generate video frames for the web app."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Skipping...")
            continue  # Skip the current iteration and continue with the next frame

        # Detect and predict faces
        predictions = detect_face_and_predict(frame, classifier, label_mapping)
        for label, confidence, (x, y, w, h) in predictions:
            if label != "No face detected":
                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Provide the video feed."""
    return Response(gen_frames(classifier, label_mapping), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/add_person', methods=['GET', 'POST'])
def add_person():
    """Add a new person to the face recognition system."""
    if request.method == 'POST':
        name = request.form['name']
        return render_template('capture_person.html', name=name)
    return render_template('add_person.html')

@app.route('/video_feed_add/<name>')
def video_feed_add(name):
    """Provide the video feed for adding a person."""
    return Response(gen_add_person_frames(name), mimetype='multipart/x-mixed-replace; boundary=frame')


def gen_add_person_frames(name):
    """Generate frames for capturing a new person's data."""
    global classifier, label_mapping, X_dataset, y_dataset
    embeddings = []
    frame_count = 0

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while frame_count < 50:
        ret, frame = cap.read()
        if not ret:
            break

        # Process face and collect embeddings
        embedding = process_face(frame)
        if embedding is not None:
            embeddings.append(embedding)
            frame_count += 1
            cv2.putText(frame, f"Adding {name} ({frame_count}/50)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected. Adjust position.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

    # Once all frames are captured, save the person data
    classifier, label_mapping = add_new_person(name, embeddings, classifier, label_mapping)
    save_model_and_mapping(classifier, label_mapping)

if __name__ == "__main__":
    # Load the model and mappings
    classifier, label_mapping = load_model_and_mapping()

    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
