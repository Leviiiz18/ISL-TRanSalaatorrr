from keras.models import model_from_json
import cv2
import numpy as np
import mediapipe as mp
import time

# Load the model
with open("s124.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("S224.h5")

labels = ['A', 'M', 'N', 'S', 'T', 'blank']

def extract_features(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (48, 48))
    image = image.reshape(1, 48, 48, 1)
    return image / 255.0

# MediaPipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

output_text = ""
prev_label = ""
label_stable_since = time.time()
MIN_STABLE_DURATION = 1.2  # seconds to hold a gesture

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror view
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box around the hand
            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]
            xmin = int(min(x_list) * w) - 20
            ymin = int(min(y_list) * h) - 20
            xmax = int(max(x_list) * w) + 20
            ymax = int(max(y_list) * h) + 20

            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)

            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 165, 255), 2)

            # Crop the hand ROI and predict
            hand_img = frame[ymin:ymax, xmin:xmax]
            if hand_img.size == 0:
                continue

            processed = extract_features(hand_img)
            pred = model.predict(processed)
            confidence = np.max(pred)
            label = labels[pred.argmax()]

            # Add label only if not blank
            if label != 'blank':
                if label == prev_label:
                    if time.time() - label_stable_since > MIN_STABLE_DURATION:
                        output_text += label
                        label_stable_since = time.time()
                        prev_label = ""
                else:
                    prev_label = label
                    label_stable_since = time.time()

                # Draw label
                cv2.rectangle(frame, (xmin, ymin - 40), (xmin + 200, ymin - 10), (0, 165, 255), -1)
                cv2.putText(frame, f"{label} {confidence * 100:.2f}%", (xmin + 5, ymin - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                prev_label = ""

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Text area
    cv2.rectangle(frame, (10, 400), (630, 470), (50, 50, 50), -1)
    cv2.putText(frame, "Typed: " + output_text, (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Sign Language to Text", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break
    elif key == ord('c'):  # Press 'c' to clear
        output_text = ""

cap.release()
cv2.destroyAllWindows()
