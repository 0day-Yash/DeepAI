import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load pre-trained models
print("Loading face cascade...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("Loading EfficientDet model...")
detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/d0/1")

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return frame

def detect_objects(frame):
    input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_frame = cv2.resize(input_frame, (512, 512))
    input_frame = np.expand_dims(input_frame, axis=0)
    
    detector_output = detector(input_frame)
    boxes = detector_output['detection_boxes'].numpy()
    class_ids = detector_output['detection_classes'].numpy().astype(np.int32)
    scores = detector_output['detection_scores'].numpy()

    h, w, _ = frame.shape
    for box, class_id, score in zip(boxes, class_ids, scores):
        if score.any() > 0.3:
            ymin, xmin, ymax, xmax = box
            xmin, xmax, ymin, ymax = int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f"{class_id}: {score:.2f}"
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
