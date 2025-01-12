import cv2
import os
import numpy as np
from utils.face_processing import process_face
# from utils.model_utils import add_new_face
import time
import joblib
import pickle
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import shutil
import random
from sklearn.preprocessing import normalize


MODEL_PATH = "model/classifier.joblib"
MAPPING_PATH = "model/label_mapping.pkl"

X_dataset = []
y_dataset = []

def add_new_person(name, embeddings, classifier, label_mapping):
    """Add a new person to the KNN classifier."""
    if not embeddings:
        print("No valid embeddings collected. Skipping addition of the new person.")
        return classifier, label_mapping

    # Assign a new label for the person
    if name not in label_mapping.values():
        new_label = len(label_mapping)
        label_mapping[new_label] = name
    else:
        new_label = [key for key, value in label_mapping.items() if value == name][0]

    print(f"Adding {name} with {len(embeddings)} embeddings to the dataset.")

    # Update the global dataset
    global X_dataset, y_dataset
    X_dataset.extend(embeddings)
    y_dataset.extend([new_label] * len(embeddings))

    # Refit the classifier with all data
    X_train = np.array(X_dataset)
    y_train = np.array(y_dataset)

    try:
        classifier.named_steps['kneighborsclassifier'].fit(X_train, y_train)
        print(f"Successfully retrained the classifier with {len(np.unique(y_train))} classes.")
    except ValueError as e:
        print(f"Error during classifier fitting: {e}")
        raise

    return classifier, label_mapping




def load_model_and_mapping(model_path, mapping_path):
    """Load the KNN classifier and label mapping."""
    global X_dataset, y_dataset
    try:
        classifier = joblib.load(model_path)
        with open(mapping_path, "rb") as f:
            label_mapping = pickle.load(f)
        # Load the dataset
        X_dataset = np.load("model/X_dataset.npy", allow_pickle=True).tolist()
        y_dataset = np.load("model/y_dataset.npy", allow_pickle=True).tolist()
        print("Model, dataset, and label mapping loaded successfully.")
    except FileNotFoundError:
        print("Model or label mapping not found. Initializing new model.")
        classifier = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5)
        )
        label_mapping = {}
        X_dataset = []
        y_dataset = []
    return classifier, label_mapping


def save_model_and_mapping(classifier, label_mapping, model_path, mapping_path):
    """Save the KNN classifier, dataset, and label mapping."""
    global X_dataset, y_dataset
    joblib.dump(classifier, model_path)
    with open(mapping_path, "wb") as f:
        pickle.dump(label_mapping, f)
    # Save the dataset
    np.save("model/X_dataset.npy", np.array(X_dataset, dtype=object))
    np.save("model/y_dataset.npy", np.array(y_dataset, dtype=object))
    print("Model, dataset, and label mapping saved.")


def capture_and_save_frames_with_gaps(output_dir, num_frames=20, gap_seconds=1):
    """
    Capture and save frames to a specified directory with gaps between captures.
    
    Args:
        output_dir (str): Directory to save frames.
        num_frames (int): Number of frames to capture.
        gap_seconds (float): Time gap between each frame capture.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        return
    
    print(f"Capturing {num_frames} frames with a gap of {gap_seconds} seconds. Press 'q' to exit early.")
    frame_count = 0

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Save the current frame
        file_path = os.path.join(output_dir, f"frame_{frame_count + 1}.jpg")
        cv2.imwrite(file_path, frame)
        frame_count += 1
        print(f"Saved frame {frame_count} to {file_path}")

        # Display the current frame
        cv2.imshow("Capturing Frames", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting capture early.")
            break

        # Wait for the specified gap duration
        time.sleep(gap_seconds)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {frame_count} frames in {output_dir}")



def train_from_saved_images(name, image_dir, classifier, label_mapping):
    """Train the KNN classifier with images from a directory."""
    classifier, label_mapping = add_new_person(name, image_dir, classifier, label_mapping)
    return classifier, label_mapping



def test_model(image_dir, classifier, label_mapping):
    """Test the KNN classifier on saved images."""
    for file_name in os.listdir(image_dir):
        file_path = os.path.join(image_dir, file_name)
        frame = cv2.imread(file_path)
        embedding = process_face(frame)
        if embedding is not None:
            probabilities = classifier.named_steps['kneighborsclassifier'].predict_proba([embedding])[0]
            for idx, prob in enumerate(probabilities):
                label = label_mapping.get(idx, "Unknown")
                print(f"{label}: {prob:.2f}")
            predicted_class = np.argmax(probabilities)
            print(f"Predicted: {label_mapping.get(predicted_class, 'Unknown')}, Confidence: {probabilities[predicted_class]:.2f}")
        else:
            print(f"No face detected in {file_name}.")


def split_train_test(input_dir, train_dir, test_dir, test_size=0.2):
    """
    Split the images in the input directory into training and testing sets.
    """
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    all_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    train_files, test_files = train_test_split(all_files, test_size=test_size, random_state=42)

    for file_name in train_files:
        shutil.move(os.path.join(input_dir, file_name), os.path.join(train_dir, file_name))

    for file_name in test_files:
        shutil.move(os.path.join(input_dir, file_name), os.path.join(test_dir, file_name))

    print(f"Split completed: {len(train_files)} training and {len(test_files)} testing files.")


def add_face_mode(cap, classifier, label_mapping, num_frames=50):
    """Mode to add a new person by capturing frames and saving embeddings."""
    print("Add New Face Mode. Enter a label for the new face:")
    new_name = input("Label: ")

    print(f"Capturing {num_frames} frames for {new_name}...")
    embeddings = []
    frame_count = 0

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Process the face and collect embedding
        embedding = process_face(frame)
        if embedding is not None:
            embeddings.append(embedding)
            frame_count += 1
            cv2.putText(frame, f"Adding face ({frame_count}/{num_frames})...",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(f"Frame {frame_count}: Detection successful")
        else:
            cv2.putText(frame, "No face detected. Please adjust position.",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print(f"Frame {frame_count + 1}: Detection failed")

        cv2.imshow("Add New Face", frame)

        # Exit early if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting face capture early.")
            break

    print("Frames captured. Processing embeddings...")

    # Train the model with collected embeddings
    if embeddings:
        classifier, label_mapping = add_new_person(new_name, embeddings, classifier, label_mapping)
        save_model_and_mapping(classifier, label_mapping, MODEL_PATH, MAPPING_PATH)
        print(f"Successfully added {new_name}.")
    else:
        print("No embeddings collected. Could not add the new face.")

    return classifier, label_mapping



def recognition_mode(cap, classifier, label_mapping):
    """Real-time face recognition mode."""
    print("Face Recognition Mode. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Convert frame to RGB as `process_face` works with RGB images
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect and process faces (including bounding boxes)
        results = process_face(frame_rgb, return_boxes=True)
        if not results:
            print("No faces detected.")
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Iterate through each detected face
        for embedding, (x, y, w, h) in results:
            if embedding is None or len(embedding) == 0 or embedding.shape != (512,):
                print("Invalid embedding. Skipping this face.")
                continue

            # Normalize the embedding
            embedding = normalize([embedding])[0]

            # Predict with the classifier
            try:
                probabilities = classifier.named_steps['kneighborsclassifier'].predict_proba([embedding])[0]
                predicted_class = np.argmax(probabilities)
                name = label_mapping.get(predicted_class, "Unknown")
                confidence = probabilities[predicted_class]

                # Draw red box around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # Add the name and confidence score above the box
                cv2.putText(frame, f"{name} ({confidence:.2f})",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                print(f"Recognized: {name}, Confidence: {confidence:.2f}")
            except ValueError as e:
                print(f"Error during recognition: {e}")

        # Display the frame
        cv2.imshow("Face Recognition", frame)

        # Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    # Initialize webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        exit()

    # Load model and mappings
    classifier, label_mapping = load_model_and_mapping(MODEL_PATH, MAPPING_PATH)

    # Main menu
    print("Choose mode:")
    print("1. Add New Face")
    print("2. Recognize Faces")
    choice = input("Enter 1 or 2: ")

    if choice == "1":
        classifier, label_mapping = add_face_mode(cap, classifier, label_mapping, num_frames=50)
    elif choice == "2":
        recognition_mode(cap, classifier, label_mapping)
    else:
        print("Invalid choice. Exiting.")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

