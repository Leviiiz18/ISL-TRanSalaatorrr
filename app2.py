import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response
from keras.models import load_model
import json

app = Flask(__name__)

# Load student model
model = load_model('student_model.h5')

# Load class indices to label mapping (digits 0-9 and letters A-Z)
with open('class_indices.json') as f:
    class_indices = json.load(f)

# Create inverse mapping list by index order for quick lookup
labels = [None] * len(class_indices)
for label, idx in class_indices.items():
    labels[idx] = label

print("Labels loaded:", labels)  # Debug: check labels

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils


def gen_frames():
    cap = cv2.VideoCapture(0)
    pad = 20  # padding around detected hand bbox

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                img_h, img_w, _ = frame.shape
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                x_min = int(min(x_coords) * img_w)
                x_max = int(max(x_coords) * img_w)
                y_min = int(min(y_coords) * img_h)
                y_max = int(max(y_coords) * img_h)

                x1 = max(x_min - pad, 0)
                y1 = max(y_min - pad, 0)
                x2 = min(x_max + pad, img_w)
                y2 = min(y_max + pad, img_h)

                hand_img = frame[y1:y2, x1:x2]

                if hand_img.size == 0:
                    continue

                # Preprocess for model
                hand_img = cv2.resize(hand_img, (64, 64))
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                hand_img = hand_img.astype('float32') / 255.0
                hand_img = np.expand_dims(hand_img, axis=0)

                preds = model.predict(hand_img)
                pred_class = np.argmax(preds)
                label = labels[pred_class]

                # Draw red bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Draw landmarks on hand
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')  # Your HTML page with <img src="/video_feed">


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
