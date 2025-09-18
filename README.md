# HueFocus
Real-Time Colour Identifier from Web Cam

import numpy as np
import cv2

# Open webcam safely
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not webcam.isOpened():
    print("❌ Cannot open camera")
    exit()

# Make windows fullscreen
cv2.namedWindow("Live Feed", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Live Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.namedWindow("Snapshot - Detected Colors", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Snapshot - Detected Colors", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Defining HSV ranges for colors
color_ranges = {
    "Red": ([0, 120, 70], [10, 255, 255], (0, 0, 255)),
    "Red2": ([170, 120, 70], [180, 255, 255], (0, 0, 255)),  # second red range
    "Green": ([36, 50, 70], [89, 255, 255], (0, 255, 0)),
    "Blue": ([90, 50, 70], [128, 255, 255], (255, 0, 0)),
    "Cyan": ([85, 100, 100], [95, 255, 255], (255, 255, 0)),
    "Magenta": ([140, 100, 100], [170, 255, 255], (255, 0, 255)),
    "White": ([0, 0, 200], [180, 40, 255], (255, 255, 255)),
    "Black": ([0, 0, 0], [180, 255, 30], (0, 0, 0)),
}

def detect_colors(frame):
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), "uint8")
    detected_objects = []

    for color_name, (lower, upper, bgr) in color_ranges.items():
        lower = np.array(lower, np.uint8)
        upper = np.array(upper, np.uint8)

        mask = cv2.inRange(hsvFrame, lower, upper)
        mask = cv2.dilate(mask, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 800:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)
                cv2.putText(frame, f"{color_name}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)
                detected_objects.append(color_name)

    return frame, detected_objects

# State variables
frozen_frame = None

while True:
    if frozen_frame is None:
        ret, imageFrame = webcam.read()
        if not ret:
            print("❌ Failed to grab frame")
            break
        display_frame, _ = detect_colors(imageFrame.copy())
        cv2.imshow("Live Feed", display_frame)
    else:
        display_frame, detected = detect_colors(frozen_frame.copy())
        if detected:
            text = "Detected: " + ", ".join(set(detected))
            cv2.putText(display_frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 255), 3)
        cv2.imshow("Snapshot - Detected Colors", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        frozen_frame = imageFrame.copy()
        print("📸 Snapshot taken!")
    elif key == ord('r'):
        frozen_frame = None
        cv2.destroyWindow("Snapshot - Detected Colors")
        print("🔄 Back to live feed")

webcam.release()
cv2.destroyAllWindows()
