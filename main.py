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
MODE = 'train'  # ← CHANGE THIS to switch between modes

# Data directories
DATA_DIR = './sign_language_data'
MODEL_FILE = 'sign_language_model.pkl'

# Number of samples to collect per gesture (recommendation: 100-200)
SAMPLES_PER_GESTURE = 100

# Sign language labels - Customize as needed
# Keys 0-9 map to these labels during data collection
LABELS = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'
}

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         MEDIAPIPE SETUP                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe Hands model
# This uses Google's pre-trained hand detection model (bundled with mediapipe)
hands = mp_hands.Hands(
    static_image_mode=False,  # False = video stream, True = static images
    max_num_hands=1,  # Detect only 1 hand (change to 2 for both hands)
    min_detection_confidence=0.5,  # Minimum confidence for hand detection
    min_tracking_confidence=0.5  # Minimum confidence for hand tracking
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                      DATA COLLECTION FUNCTIONS                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def collect_training_data():
    """
    Collect training data for sign language gestures using webcam.

    Controls:
        Keys 0-9: Collect samples for corresponding gesture
        Q: Save data and quit

    Returns:
        None (saves data to file)
    """
    print("=" * 70)
    print(" " * 20 + "DATA COLLECTION MODE")
    print("=" * 70)
    print("Instructions:")
    print("  - Press keys 0-9 to collect samples for different letters")
    print("  - Hold the key while making the gesture")
    print("  - Move your hand around to capture different angles")
    print("  - Collect at least 100 samples per gesture for good accuracy")
    print("  - Press 'q' to save and quit")
    print("=" * 70)
    print()

    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    # Initialize data storage
    data = []
    labels = []

    # Open webcam
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        print("Solutions:")
        print("  1. Check if camera is connected")
        print("  2. Try changing VideoCapture(0) to VideoCapture(1)")
        print("  3. Close other applications using the camera")
        print("  4. Check camera permissions")
        return

    print("Camera opened successfully!")
    print("Collecting data... Press 0-9 to start collecting samples.")
    print()

    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to capture frame")
            break

        # Flip frame horizontally for mirror view (easier for user)
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB (MediaPipe uses RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with MediaPipe to detect hands
        results = hands.process(frame_rgb)

        # Display instructions on frame
        cv2.putText(frame, 'Press 0-9 to collect data, Q to quit',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display sample count
        cv2.putText(frame, f'Total samples: {len(data)}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw hand landmarks if hand is detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Display frame
        cv2.imshow('Sign Language Data Collection', frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        # Check for data collection keys (0-9)
        if key >= ord('0') and key <= ord('9'):
            label = key - ord('0')

            if results.multi_hand_landmarks:
                # Extract landmarks from detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks_array = extract_landmarks(hand_landmarks)

                # Store data and label
                data.append(landmarks_array)
                labels.append(label)

                # Print progress
                label_name = LABELS.get(label, 'Unknown')
                print(f"✓ Collected sample for label {label} ({label_name}). Total: {len(data)}")
            else:
                print("✗ No hand detected! Please show your hand to the camera.")

        # Quit on 'q' key
        elif key == ord('q'):
            print("\nQuitting data collection...")
            break

    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()

    # Save collected data
    if len(data) > 0:
        save_data = {'data': data, 'labels': labels}
        save_path = os.path.join(DATA_DIR, 'training_data.pkl')

        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)

        print()
        print("=" * 70)
        print(f"SUCCESS! Saved {len(data)} samples to {save_path}")
        print("=" * 70)
        print("\nData collection summary:")

        # Show distribution of collected samples
        from collections import Counter
        label_counts = Counter(labels)
        for label_id, count in sorted(label_counts.items()):
            label_name = LABELS.get(label_id, 'Unknown')
            print(f"  Label {label_id} ({label_name}): {count} samples")

        print()
        print("Next step: Set MODE = 'train' and run the program to train the model.")
        print("=" * 70)
    else:
        print()
        print("=" * 70)
        print("WARNING: No data collected!")
        print("=" * 70)


def extract_landmarks(hand_landmarks):
    """
    Extract and normalize landmark coordinates from MediaPipe hand landmarks.

    Args:
        hand_landmarks: MediaPipe hand landmarks object (21 landmarks)

    Returns:
        numpy array: 1D array of 42 features (21 landmarks × 2 coordinates)
    """
    landmarks = []

    # Get x and y coordinates of all 21 landmarks
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]

    # Find bounding box of hand
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Normalize each landmark relative to bounding box
    for landmark in hand_landmarks.landmark:
        # Normalize coordinates to [0, 1] range
        # This makes recognition scale-invariant (works at different distances)
        x_norm = (landmark.x - x_min) / (x_max - x_min + 1e-6)
        y_norm = (landmark.y - y_min) / (y_max - y_min + 1e-6)

        landmarks.extend([x_norm, y_norm])

    return np.array(landmarks)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       MODEL TRAINING FUNCTIONS                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def train_model():
    """
    Train a Random Forest classifier on collected training data.

    Process:
        1. Load training data from file
        2. Split into train/test sets (80/20)
        3. Train Random Forest classifier
        4. Evaluate on test set
        5. Save trained model to file

    Returns:
        None (saves model to file)
    """
    print("=" * 70)
    print(" " * 25 + "MODEL TRAINING MODE")
    print("=" * 70)
    print()

    # Load training data
    data_path = os.path.join(DATA_DIR, 'training_data.pkl')

    if not os.path.exists(data_path):
        print("ERROR: Training data not found!")
        print(f"Expected location: {data_path}")
        print()
        print("Solution: Run data collection first (set MODE = 'collect')")
        print("=" * 70)
        return

    print(f"Loading training data from {data_path}...")
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)

    data = np.array(dataset['data'])
    labels = np.array(dataset['labels'])

    print(f"✓ Loaded {len(data)} samples")
    print(f"  Feature shape: {data.shape}")
    print(f"  Unique labels: {np.unique(labels)}")
    print()

    # Validate data
    if len(data) < 10:
        print("ERROR: Not enough training data!")
        print(f"  Current samples: {len(data)}")
        print(f"  Minimum required: 10")
        print(f"  Recommended: 100+ per gesture")
        print()
        print("Solution: Collect more samples using data collection mode")
        print("=" * 70)
        return

    # Show data distribution
    from collections import Counter
    label_counts = Counter(labels)
    print("Data distribution:")
    for label_id, count in sorted(label_counts.items()):
        label_name = LABELS.get(label_id, 'Unknown')
        print(f"  Label {label_id} ({label_name}): {count} samples")
    print()

    # Split data into training and testing sets
    print("Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels,
        test_size=0.2,
        shuffle=True,
        stratify=labels,
        random_state=42
    )

    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    print()

    # Train Random Forest classifier
    print("Training Random Forest classifier...")
    print("  Parameters:")
    print("    - n_estimators: 100 (number of decision trees)")
    print("    - max_depth: 20 (maximum tree depth)")
    print("    - n_jobs: -1 (use all CPU cores)")
    print()

    model = RandomForestClassifier(
        n_estimators=100,  # Number of trees in the forest
        max_depth=20,  # Maximum depth of each tree
        random_state=42,  # For reproducibility
        n_jobs=-1  # Use all CPU cores for faster training
    )

    model.fit(X_train, y_train)
    print("✓ Training completed!")
    print()

    # Evaluate model on test set
    print("Evaluating model on test set...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print()
    print("=" * 70)
    print(f"MODEL ACCURACY: {accuracy * 100:.2f}%")
    print("=" * 70)
    print()

    # Provide feedback based on accuracy
    if accuracy >= 0.95:
        print("✓ EXCELLENT! Your model has very high accuracy.")
    elif accuracy >= 0.85:
        print("✓ GOOD! Your model has good accuracy.")
    elif accuracy >= 0.70:
        print("⚠ FAIR. Consider collecting more training samples.")
    else:
        print("⚠ LOW ACCURACY. Collect more samples and ensure distinct gestures.")
    print()

    # Save model
    model_data = {'model': model, 'labels': LABELS}
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"✓ Model saved to {MODEL_FILE}")
    print()
    print("Next step: Set MODE = 'recognize' and run the program to test recognition.")
    print("=" * 70)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       RECOGNITION FUNCTIONS                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def recognize_sign_language():
    """
    Real-time sign language recognition using webcam and trained model.

    Controls:
        SPACE: Add current prediction to text
        C: Clear accumulated text
        Q: Quit

    Returns:
        None
    """
    print("=" * 70)
    print(" " * 20 + "SIGN LANGUAGE RECOGNITION MODE")
    print("=" * 70)
    print()

    # Load trained model
    if not os.path.exists(MODEL_FILE):
        print("ERROR: Trained model not found!")
        print(f"Expected location: {MODEL_FILE}")
        print()
        print("Solution: Train the model first (set MODE = 'train')")
        print("=" * 70)
        return

    print(f"Loading model from {MODEL_FILE}...")
    with open(MODEL_FILE, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    labels_dict = model_data['labels']

    print("✓ Model loaded successfully!")
    print()
    print("Controls:")
    print("  SPACE - Add current prediction to text")
    print("  C     - Clear accumulated text")
    print("  Q     - Quit")
    print()
    print("Starting recognition...")
    print("=" * 70)
    print()

    # Open webcam
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        print("Solutions:")
        print("  1. Check if camera is connected")
        print("  2. Try changing VideoCapture(0) to VideoCapture(1)")
        print("  3. Close other applications using the camera")
        print("  4. Check camera permissions")
        return

    # Text accumulation
    recognized_text = ""

    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to capture frame")
            break

        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with MediaPipe
        results = hands.process(frame_rgb)

        prediction_text = ""

        # If hand is detected
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract features and make prediction
            try:
                landmarks_array = extract_landmarks(hand_landmarks)
                landmarks_array = landmarks_array.reshape(1, -1)

                # Predict gesture
                prediction = model.predict(landmarks_array)[0]
                confidence = max(model.predict_proba(landmarks_array)[0])

                predicted_label = labels_dict.get(prediction, 'Unknown')
                prediction_text = f"{predicted_label} ({confidence * 100:.1f}%)"

                # Draw bounding box around hand
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                x1 = int(min(x_coords) * w) - 20
                y1 = int(min(y_coords) * h) - 20
                x2 = int(max(x_coords) * w) + 20
                y2 = int(max(y_coords) * h) + 20

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Display prediction above bounding box
                cv2.putText(frame, prediction_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            except Exception as e:
                print(f"Prediction error: {e}")

        # Display accumulated text at bottom of frame
        cv2.putText(frame, f"Text: {recognized_text}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display instructions at top of frame
        cv2.putText(frame, "Press 'q' to quit, SPACE to add letter, 'c' to clear",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Show frame
        cv2.imshow('Sign Language Recognition', frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # Quit
            print("\nQuitting recognition...")
            break
        elif key == ord(' ') and prediction_text:
            # Add letter to text (SPACE key)
            letter = prediction_text.split()[0]
            recognized_text += letter
            print(f"Added: {letter} | Text: {recognized_text}")
        elif key == ord('c'):
            # Clear text (C key)
            recognized_text = ""
            print("Text cleared")

    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()

    # Display final text
    if recognized_text:
        print()
        print("=" * 70)
        print(f"FINAL RECOGNIZED TEXT: {recognized_text}")
        print("=" * 70)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          MAIN PROGRAM                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    """
    Main program entry point.
    Routes to appropriate function based on MODE setting.
    """
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 18 + "SIGN LANGUAGE RECOGNITION SYSTEM" + " " * 18 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print(f"Current mode: {MODE}")
    print()

    # Route to appropriate function based on mode
    if MODE == 'collect':
        collect_training_data()
    elif MODE == 'train':
        train_model()
    elif MODE == 'recognize':
        recognize_sign_language()
    else:
        print("ERROR: Invalid mode!")
        print(f"Current MODE: {MODE}")
        print()
        print("Valid modes:")
        print("  'collect'   - Collect training data")
        print("  'train'     - Train the model")
        print("  'recognize' - Recognize sign language")
        print()
        print("Please set MODE variable (line 58) to one of the valid modes.")
        print("=" * 70)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         PROGRAM ENTRY POINT                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    main()
