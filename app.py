import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response
from keras.models import load_model
import tensorflow as tf
import json

app = Flask(__name__)

# Load student model (multi-class)
student_model = load_model('student_model1.h5')

# Load expert model for 'L' only
expert_L_model = tf.keras.models.load_model('expert_I_model.keras')

# Load class indices
with open('class_indices.json') as f:
    class_indices = json.load(f)
inv_class_indices = {v: k for k, v in class_indices.items()}

# MediaPipe setup
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
    pad = 20

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

                hand_img_resized = cv2.resize(hand_img, (64, 64))
                hand_img_rgb = cv2.cvtColor(hand_img_resized, cv2.COLOR_BGR2RGB)
                hand_img_norm = hand_img_rgb.astype('float32') / 255.0
                hand_img_input = np.expand_dims(hand_img_norm, axis=0)

                # Step 1: Run student model
                preds = student_model.predict(hand_img_input)
                pred_class = np.argmax(preds)
                pred_label = inv_class_indices.get(pred_class, 'Unknown')

                # Step 2: Trigger expert model only if student predicts 'L'
                if pred_label == 'L':
                    print("🔍 Student predicted 'L' – triggering expert model...")
                    expert_l_prob = expert_L_model.predict(hand_img_input)[0][0]
                    print(f"👉 Expert L model probability: {expert_l_prob:.4f}")
                    if expert_l_prob > 0.95:
                        label = "L (Expert)"
                    else:
                        label = "Not L (Refuted by Expert)"
                else:
                    label = pred_label

                # Draw box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Encode and yield frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
