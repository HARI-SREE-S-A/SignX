import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                            CONFIGURATION                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Mode selection: 'collect', 'train', or 'recognize'
MODE = 'collect'  # Change this to switch between modes

# Data directories
DATA_DIR = './sign_language_data'
MODEL_FILE = 'sign_language_model.pkl'
SAMPLES_PER_GESTURE = 100

# Sign language labels
LABELS = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'
}

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         MEDIAPIPE SETUP                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                      DATA COLLECTION FUNCTIONS                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def collect_training_data():
    print('=' * 70)
    print('DATA COLLECTION MODE')
    print('=' * 70)

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f'Created directory: {DATA_DIR}')

    data, labels = [], []
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print('ERROR: Cannot open camera!')
        return

    print('Camera opened successfully! Press 0-9 to collect, Q to quit.')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        cv2.putText(frame, 'Press 0-9 to collect data, Q to quit', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        cv2.imshow('Sign Language Data Collection', frame)
        key = cv2.waitKey(1) & 0xFF

        if key >= ord('0') and key <= ord('9'):
            if results.multi_hand_landmarks:
                label = key - ord('0')
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks_array = extract_landmarks(hand_landmarks)
                data.append(landmarks_array)
                labels.append(label)
                print(f'Collected sample for label {label} ({LABELS[label]}) | Total: {len(data)}')
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(data) > 0:
        with open(os.path.join(DATA_DIR, 'training_data.pkl'), 'wb') as f:
            pickle.dump({'data': data, 'labels': labels}, f)
        print(f'Saved {len(data)} samples to ./sign_language_data/training_data.pkl')
    else:
        print('No data collected!')

def extract_landmarks(hand_landmarks):
    landmarks = []
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    for landmark in hand_landmarks.landmark:
        x_norm = (landmark.x - x_min) / (x_max - x_min + 1e-6)
        y_norm = (landmark.y - y_min) / (y_max - y_min + 1e-6)
        landmarks.extend([x_norm, y_norm])
    return np.array(landmarks)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       MODEL TRAINING FUNCTIONS                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def train_model():
    print('=' * 70)
    print('MODEL TRAINING MODE')
    print('=' * 70)

    data_path = os.path.join(DATA_DIR, 'training_data.pkl')
    if not os.path.exists(data_path):
        print('ERROR: No training data found! Please collect samples first.')
        return

    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)

    data = np.array(dataset['data'])
    labels = np.array(dataset['labels'])

    if len(data) < 5:
        print('Not enough data to train. Please collect more samples!')
        return

    print(f'Loaded {len(data)} samples.')

    # FIX APPLIED HERE: removed stratify=labels
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'MODEL ACCURACY: {acc * 100:.2f}%')

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({'model': model, 'labels': LABELS}, f)
    print(f'Model saved to {MODEL_FILE}')

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       RECOGNITION FUNCTION                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def recognize_sign_language():
    if not os.path.exists(MODEL_FILE):
        print('Error: Model not found! Train it first.')
        return

    with open(MODEL_FILE, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    labels_dict = model_data['labels']

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('ERROR: Cannot open camera!')
        return

    recognized_text = ''

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())
            try:
                landmarks_array = extract_landmarks(hand_landmarks).reshape(1, -1)
                prediction = model.predict(landmarks_array)[0]
                confidence = max(model.predict_proba(landmarks_array)[0])
                predicted_label = labels_dict.get(prediction, 'Unknown')
                cv2.putText(frame, f'{predicted_label} ({confidence*100:.1f}%)', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                print('Prediction error:', e)

        cv2.imshow('Sign Language Recognition', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                             MAIN PROGRAM                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    print('Sign Language Recognition System\nMode:', MODE)
    if MODE == 'collect':
        collect_training_data()
    elif MODE == 'train':
        train_model()
    elif MODE == 'recognize':
        recognize_sign_language()
    else:
        print('Invalid mode! Choose collect, train, or recognize.')

if __name__ == '__main__':
    main()

