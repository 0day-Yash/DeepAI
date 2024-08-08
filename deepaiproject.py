import cv2
import numpy as np
import tensorflow as tf

# Load pre-trained models
print("Loading face cascade...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("Loading MobileNetV2 model...")
model = tf.keras.applications.MobileNetV2(weights='imagenet')
print("Models loaded successfully.")

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return frame

def detect_objects(frame):
    input_frame = cv2.resize(frame, (224, 224))
    input_frame = np.expand_dims(input_frame, axis=0)
    input_frame = tf.keras.applications.mobilenet_v2.preprocess_input(input_frame)

    predictions = model.predict(input_frame)
    top_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

    for i, (imagenet_id, label, score) in enumerate(top_predictions):
        label_text = f"{label}: {score:.2f}"
        cv2.putText(frame, label_text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def main():
    print("Starting video capture...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    print("Video capture started.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_faces(frame)
        frame = detect_objects(frame)

        cv2.imshow('Face and Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
