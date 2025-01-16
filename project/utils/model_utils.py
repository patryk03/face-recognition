from river.neighbors import KNNClassifier
import pickle
import numpy as np

label_mapping = {}  # Initialize an empty label mapping


def load_model_and_mapping(model_path, mapping_path):
    """Load the online KNN classifier and label mapping."""
    global label_mapping
    try:
        # Load the classifier
        with open(model_path, "rb") as f:
            classifier = pickle.load(f)

        # Load the label mapping
        with open(mapping_path, "rb") as f:
            label_mapping = pickle.load(f)

        print("Model and label mapping loaded successfully.")
    except FileNotFoundError:
        print("Model or label mapping not found. Initializing new model.")
        # Initialize the online KNN classifier
        classifier = KNNClassifier(n_neighbors=5)
        label_mapping = {}

    return classifier, label_mapping


def save_model_and_mapping(classifier, label_mapping, model_path, mapping_path):
    """Save the online KNN classifier and label mapping."""
    # Save the classifier
    with open(model_path, "wb") as f:
        pickle.dump(classifier, f)

    # Save the label mapping
    with open(mapping_path, "wb") as f:
        pickle.dump(label_mapping, f)

    print("Model and label mapping saved.")


def add_new_face(name, embeddings, classifier, label_mapping):
    """Add a new face to the online KNN classifier."""
    if not embeddings:
        print("No embeddings provided. Skipping addition of new face.")
        return classifier, label_mapping

    # Assign a label to the person
    if name not in label_mapping.values():
        new_label = len(label_mapping)
        label_mapping[new_label] = name
    else:
        new_label = [key for key, value in label_mapping.items() if value == name][0]

    print(f"Adding {name} with {len(embeddings)} embeddings to the dataset.")

    # Incrementally add each embedding to the classifier
    for embedding in embeddings:
        # Convert embedding to a dictionary for River compatibility
        embedding_dict = dict(enumerate(embedding))
        classifier = classifier.learn_one(embedding_dict, new_label)

    print(f"Successfully added {name} to the classifier.")
    return classifier, label_mapping
