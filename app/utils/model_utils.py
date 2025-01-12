import joblib
import pickle
import numpy as np
from utils.face_processing import process_face
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


X_dataset = []
y_dataset = []


def load_model_and_mapping(model_path, mapping_path):
    """Load the KNN classifier and label mapping."""
    global X_dataset, y_dataset
    try:
        # Load the classifier and label mapping
        classifier = joblib.load(model_path)
        with open(mapping_path, "rb") as f:
            label_mapping = pickle.load(f)

        # Load the dataset
        X_dataset = np.load("model/X_dataset.npy", allow_pickle=True).tolist()
        y_dataset = np.load("model/y_dataset.npy", allow_pickle=True).tolist()

        print("Model, dataset, and label mapping loaded successfully.")
    except FileNotFoundError:
        print("Model or label mapping not found. Initializing new model.")
        # Initialize KNN classifier
        classifier = KNeighborsClassifier(n_neighbors=5)
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


def add_new_face(name, embeddings, classifier, label_mapping):
    """Add a new face to the KNN classifier."""
    global X_dataset, y_dataset

    if not embeddings:
        print("No embeddings provided. Skipping addition of new face.")
        return classifier, label_mapping

    # Assign a new label for the person
    if name not in label_mapping.values():
        new_label = len(label_mapping)
        label_mapping[new_label] = name
    else:
        new_label = [key for key, value in label_mapping.items() if value == name][0]

    print(f"Adding {name} with {len(embeddings)} embeddings to the dataset.")

    # Update the dataset
    X_dataset.extend(embeddings)
    y_dataset.extend([new_label] * len(embeddings))

    # Fit the KNN classifier with the updated dataset
    try:
        classifier.fit(X_dataset, y_dataset)
        print(f"Successfully retrained the classifier with {len(np.unique(y_dataset))} classes.")
    except ValueError as e:
        print(f"Error during classifier fitting: {e}")
        raise

    return classifier, label_mapping
