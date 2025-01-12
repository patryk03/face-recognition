import joblib

# Load the model
model_path = "model/classifier.joblib"
classifier = joblib.load(model_path)

# Extract known classes
known_classes = classifier.named_steps['sgdclassifier'].classes_
print("Known classes:", known_classes)

# Recreate label mapping (if names are unknown, use generic names)
label_mapping = {label: f"Person_{label}" for label in known_classes}
print("Recreated label mapping:", label_mapping)

# Save the recreated label mapping for future use
import pickle
mapping_path = "model/label_mapping.pkl"
with open(mapping_path, "wb") as f:
    pickle.dump(label_mapping, f)

print(f"Label mapping saved to {mapping_path}.")
