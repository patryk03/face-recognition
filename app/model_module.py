import kagglehub

import numpy as np
from PIL import Image
from keras_facenet import FaceNet
from mtcnn import MTCNN
import os
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import random

path = kagglehub.dataset_download("podgorskip01/face-recognition")
facenet = FaceNet()
detector = MTCNN()

model_file_path = 'face_recognition_triplet_model.h5'

base_dir = os.path.join(path, "VGGFace2")

embeddings = []
labels = []


def detect_face(image_path):
    img = np.array(Image.open(image_path))
    results = detector.detect_faces(img)
    if results:
        x, y, width, height = results[0]['box']
        x, y = max(0, x), max(0, y)
        face = img[y:y + height, x:x + width]
        face = Image.fromarray(face).resize((160, 160))
        face_array = np.array(face) / 255.0
        return np.expand_dims(face_array, axis=0)
    return None

def extract_embeddings(directory, max_dirs=10):
    global embeddings, labels
    count = 0
    for entity_name in os.listdir(directory):
        if count >= max_dirs:
            break
        person_dir = os.path.join(directory, entity_name)
        if os.path.isdir(person_dir):
            print(f"Processing directory: {person_dir}")
            count += 1
            for sub_dir in ['people', 'people_masked']:
                sub_dir_path = os.path.join(person_dir, sub_dir)
                if os.path.isdir(sub_dir_path):
                    for img_name in os.listdir(sub_dir_path):
                        if img_name.endswith(('.jpg', '.png')):
                            img_path = os.path.join(sub_dir_path, img_name)
                            face = detect_face(img_path)
                            if face is not None:
                                embedding = facenet.embeddings(face)
                                embeddings.append(embedding.flatten())
                                labels.append(entity_name)


def generate_triplets(data, batch_size=32):
    while True:
        triplets = []
        for _ in range(batch_size):
            anchor, label = random.choice(data)

            positives = [img for img, lbl in data if lbl == label]
            positive = random.choice(positives)

            negatives = [img for img, lbl in data if lbl != label]
            negative = random.choice(negatives)

            triplets.append((anchor, positive, negative))

        anchors = np.array([t[0] for t in triplets])
        positives = np.array([t[1] for t in triplets])
        negatives = np.array([t[2] for t in triplets])
        yield [anchors, positives, negatives], None

def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[:, :128], y_pred[:, 128:256], y_pred[:, 256:]

    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

    loss = tf.maximum(pos_dist - neg_dist + alpha, 0.0)
    return tf.reduce_mean(loss)


def create_triplet_model(base_model):
    input_anchor = tf.keras.layers.Input(shape=(160, 160, 3))
    input_positive = tf.keras.layers.Input(shape=(160, 160, 3))
    input_negative = tf.keras.layers.Input(shape=(160, 160, 3))

    anchor_embedding = tf.keras.layers.Flatten()(base_model(input_anchor))
    positive_embedding = tf.keras.layers.Flatten()(base_model(input_positive))
    negative_embedding = tf.keras.layers.Flatten()(base_model(input_negative))

    merged_output = tf.keras.layers.concatenate([anchor_embedding, positive_embedding, negative_embedding], axis=1)
    model = tf.keras.models.Model(inputs=[input_anchor, input_positive, input_negative], outputs=merged_output)
    model.compile(optimizer='adam', loss=triplet_loss)
    return model


def train_triplet_model():
    data = [(embedding, label) for embedding, label in zip(embeddings, labels)]

    triplet_model = create_triplet_model(facenet.model)

    batch_size = 32
    steps_per_epoch = len(data) // batch_size

    triplet_model.fit(
        generate_triplets(data, batch_size=batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=10
    )

    triplet_model.save(model_file_path)

def predict_person_triplet(image_path):
    face = detect_face(image_path)
    if face is None:
        return "No face detected in the image."

    input_embedding = facenet.embeddings(face).flatten()

    min_dist = float('inf')
    predicted_label = None

    for known_embedding, label in zip(embeddings, labels):
        dist = np.linalg.norm(input_embedding - known_embedding)
        if dist < min_dist:
            min_dist = dist
            predicted_label = label

    threshold = 0.8
    if min_dist < threshold:
        return f"Predicted Person: {predicted_label}, Confidence: {1 - min_dist:.2f}"
    else:
        return "Unknown or not a face"

if os.path.exists(model_file_path):
    print("Loading pre-trained triplet model...")
    triplet_model = tf.keras.models.load_model(model_file_path, custom_objects={'triplet_loss': triplet_loss})
else:
    extract_embeddings(base_dir, max_dirs=100)
    train_triplet_model()
label_file_path = 'model/label_encoder.pkl'

label_encoder = LabelEncoder()
label_encoder.fit(labels)
with open(label_file_path, 'wb') as label_file:
    pickle.dump(label_encoder, label_file)