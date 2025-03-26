from ultralytics import YOLO
import cv2
import math

# Load the YOLO model
model = YOLO('best_with_100_epochs.pt')

# Define class names
classNames = ['awake', 'calling', 'chatting', 'closed_eye', 'drink', 'drowsy', 'eating',
              'no_yawn', 'open_eye', 'smoking', 'yawn']

# Class to handle detection logic


import time

class DriverStatus:
    def __init__(self):
        """Initialize with individual thresholds for each class."""
        self.awake_thresholds = {
            "open_eye": 80,
            "no_yawn": 75,
            "awake": 85,
            "chatting": 75
        }
        self.drowsy_thresholds = {
            "closed_eye": 65,
            "drink": 72,
            "drowsy": 55,
            "smoking": 80
        }

        # Store class-wise confidence values with timestamps
        self.class_confidence_history = {}

    def update_confidence(self, class_name, confidence):
        """Store confidence values and remove older than 300ms."""
        current_time = time.time()

        if class_name not in self.class_confidence_history:
            self.class_confidence_history[class_name] = []

        # Append new confidence value with timestamp
        self.class_confidence_history[class_name].append((current_time, confidence))

        # Remove old confidence values (older than 300ms)
        self.class_confidence_history[class_name] = [
            (t, c) for t, c in self.class_confidence_history[class_name] if (current_time - t) <= 0.3
        ]

    def get_average_confidence(self, class_name):
        """Compute the average confidence over the last 300ms."""
        if class_name not in self.class_confidence_history or not self.class_confidence_history[class_name]:
            return 0  # No data available

        values = [c for _, c in self.class_confidence_history[class_name]]
        return sum(values) / len(values)

    def check_status(self, class_name, confidence=None):
        """
        Check if the driver is awake or drowsy based on 300ms average confidence.
        If confidence is provided, it will be added before checking.
        """
        if confidence is not None:
            self.update_confidence(class_name, confidence)  # Store new confidence value

        avg_confidence = self.get_average_confidence(class_name)

        # Check awake thresholds
        if class_name in self.awake_thresholds and avg_confidence >= 75 and \
                avg_confidence >= self.awake_thresholds[class_name]:
            return "Awake"

        # Check drowsy thresholds
        elif class_name in self.drowsy_thresholds and avg_confidence >= 65 \
                and avg_confidence >= self.drowsy_thresholds[class_name]:
            return "Drowsy"

        # General fallback: If confidence is 70% or higher, classify
        if avg_confidence >= 70:
            if class_name in self.awake_thresholds:
                return "Awake"
            elif class_name in self.drowsy_thresholds:
                return "Drowsy"

        return None  # Ignore other classes or low confidence


    def display_status(self, img, status):
        """Display status on the screen with different colors."""
        if status == "Awake":
            cv2.rectangle(img, (10, 10), (250, 50), (0, 255, 0), -1)  # Green background
            cv2.putText(img, "Status: Awake", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        elif status == "Drowsy":
            cv2.rectangle(img, (10, 10), (250, 50), (0, 0, 255), -1)  # Red background
            cv2.putText(img, "Status: Drowsy", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Initialize the DriverStatus class
driver_status = DriverStatus()

# Open webcam
cap = cv2.VideoCapture(0)  # Use 0 for default camera

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)  # Perform real-time detection

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 228, 181), 3)

            # Get confidence score
            confidence = round(float(box.conf[0]) * 100, 2)
            cls = int(box.cls[0])
            class_name = classNames[cls]

            # Display class label & confidence
            label = f'{class_name} {confidence}%'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = (x1 + t_size[0], y1 - t_size[1] - 3)
            cv2.rectangle(img, (x1, y1), c2, (255, 228, 181), -1)
            cv2.putText(img, label, (x1, y1-2), 0, 1, (85, 107, 47), 2)

            # Check driver status based on conditions
            status = driver_status.check_status(class_name, confidence)
            if status:
                driver_status.display_status(img, status)

    cv2.imshow('Webcam', img)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
