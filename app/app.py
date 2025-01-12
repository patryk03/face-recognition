import os
import cv2
import pickle
from flask import Flask, render_template, Response
from keras_facenet import FaceNet
from mtcnn import MTCNN

MODEL_FILE_PATH = 'model/face_recognition_model.pkl'
LABEL_FILE_PATH = 'model/label_encoder.pkl'
face = 0
prediction_label = ""

mtcnn_detector = MTCNN()
facenet_model = FaceNet()

if os.path.exists(MODEL_FILE_PATH) and os.path.exists(LABEL_FILE_PATH):
    with open(MODEL_FILE_PATH, 'rb') as model_file:
        classifier = pickle.load(model_file)
    with open(LABEL_FILE_PATH, 'rb') as label_file:
        label_encoder = pickle.load(label_file)
else:
    raise FileNotFoundError("Model or label encoder not found! Train the model first.")

app = Flask(__name__, template_folder='./templates')

camera = cv2.VideoCapture(0)

def detect_face_and_predict(frame):
    global prediction_label

    results = mtcnn_detector.detect_faces(frame)
    if results:
        for result in results:
            x, y, width, height = result['box']
            x, y = max(0, x), max(0, y)

            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)

            face_region = frame[y:y + height, x:x + width]
            face_resized = cv2.resize(face_region, (160, 160)) / 255.0
            embedding = facenet_model.embeddings([face_resized]).flatten()

            probabilities = classifier.predict_proba([embedding])[0]
            max_prob = max(probabilities)
            predicted_label = classifier.predict([embedding])[0]

            if max_prob < 0.8:
                prediction_label = "Unknown"
            else:
                prediction_label = label_encoder.inverse_transform([predicted_label])[0]

            return prediction_label, x, y
    else:
        return "No face detected", 0, 0


def gen_frames():
    global prediction_label
    while True:
        success, frame = camera.read()
        if success:
            if face:
                label, x, y = detect_face_and_predict(frame)

                flipped_frame = cv2.flip(frame, 1)

                flipped_frame = cv2.putText(flipped_frame, prediction_label, (x, y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                ret, buffer = cv2.imencode('.jpg', flipped_frame)
                flipped_frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + flipped_frame + b'\r\n')
        else:
            pass


@app.route('/')
def index():
    global camera, face
    camera = cv2.VideoCapture(0)  # Reinitialize camera
    face = 1  # Start face detection
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()

camera.release()
cv2.destroyAllWindows()
