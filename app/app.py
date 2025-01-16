import cv2
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import KNeighborsClassifier
import joblib
import pickle
from utils.face_processing import process_face

MODEL_PATH = "model/classifier.joblib"
MAPPING_PATH = "model/label_mapping.pkl"
DATASET_X_PATH = "model/X_dataset.npy"
DATASET_Y_PATH = "model/y_dataset.npy"

X_dataset = []
y_dataset = []

live_recognition = False
adding_new_person = False
app = Flask(__name__, template_folder='./templates')

def load_model_and_mapping():
    """Load the KNN classifier, label mapping, and dataset."""
    global X_dataset, y_dataset
    try:
        classifier = joblib.load(MODEL_PATH)
        with open(MAPPING_PATH, "rb") as f:
            label_mapping = pickle.load(f)

        X_dataset = np.load(DATASET_X_PATH, allow_pickle=True).tolist()
        y_dataset = np.load(DATASET_Y_PATH, allow_pickle=True).tolist()

        if not hasattr(classifier.named_steps['kneighborsclassifier'], 'classes_'):
            print("Classifier not trained. Retraining...")
            X_train = np.array(X_dataset)
            y_train = np.array(y_dataset)
            classifier.named_steps['kneighborsclassifier'].fit(X_train, y_train)
            print("Classifier retrained successfully.")

        print("Model, label mapping, and dataset loaded successfully.")
        return classifier, label_mapping

    except FileNotFoundError:
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

    new_label = len(label_mapping)
    label_mapping[new_label] = name

    X_dataset.extend(embeddings)
    y_dataset.extend([new_label] * len(embeddings))

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

                if confidence < 0.8:
                    label = "Unknown"

                predictions.append((label, confidence, (x, y, w, h)))
            except Exception as e:
                print(f"Prediction error: {e}")
                predictions.append(("Unknown", 0.0, (x, y, w, h)))
        else:
            predictions.append(("No face detected", 0.0, (0, 0, 0, 0)))

    return predictions

def stop_video_stream():
    global video_streaming
    video_streaming = False

def gen_frames(classifier, label_mapping):
    """Generate video frames for the web app."""
    global live_recognition

    if not live_recognition:
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        return

    while live_recognition:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Skipping...")
            continue

        predictions = detect_face_and_predict(frame, classifier, label_mapping)
        for label, confidence, (x, y, w, h) in predictions:
            if label != "No face detected":
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    """Render the homepage."""
    global live_recognition, adding_new_person

    live_recognition = False
    adding_new_person = False

    return render_template('index.html')

@app.route('/live-recognition', methods=['GET'])
def live_recognition():
    """Render the live recognition page."""
    global live_recognition
    live_recognition = True
    return redirect('/video_feed')

@app.route('/video_feed')
def video_feed():
    """Provide the video feed."""
    return Response(gen_frames(classifier, label_mapping), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video_feed')
def stop_video_feed():
    global live_recognition, adding_new_person

    live_recognition = False
    adding_new_person = False

    return "Video Feed Stopped"

@app.route('/add_person', methods=['GET', 'POST'])
def add_person():
    """Add a new person to the face recognition system."""
    global adding_new_person

    if request.method == 'POST':
        name = request.form['name']
        adding_new_person = True
        return redirect(f'/video_feed_add/{name}')

    return render_template('add_person.html')

@app.route('/video_feed_add/<name>')
def video_feed_add(name):
    """Provide the video feed for adding a person."""
    return Response(gen_add_person_frames(name), mimetype='multipart/x-mixed-replace; boundary=frame')


def gen_add_person_frames(name):
    """Generate frames for capturing a new person's data."""
    global classifier, label_mapping, X_dataset, y_dataset, adding_new_person
    embeddings = []
    frame_count = 0

    if not adding_new_person:
        return

    cap = cv2.VideoCapture(0)
    while frame_count < 50 and adding_new_person:
        ret, frame = cap.read()
        if not ret:
            break

        embedding = process_face(frame)
        if embedding is not None:
            embeddings.append(embedding)
            frame_count += 1
            cv2.putText(frame, f"Adding {name} ({frame_count}/50)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected. Adjust position.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

    classifier, label_mapping = add_new_person(name, embeddings, classifier, label_mapping)
    save_model_and_mapping(classifier, label_mapping)

if __name__ == "__main__":
    classifier, label_mapping = load_model_and_mapping()
    app.run(host='0.0.0.0', port=5000, debug=True)
