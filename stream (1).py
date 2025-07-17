# Install dependencies and fetch video
!pip install -q imutils
!wget -q -O input_video.mp4 "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4"

import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from imutils.object_detection import non_max_suppression
from google.colab import files

# Configuration
VIDEO_SOURCE = 'input_video.mp4'
OUTPUT_VIDEO = 'annotated_output.mp4'
ALERT_LOG = 'crowd_alerts.json'
FRAME_SKIP = 3
PEOPLE_THRESHOLD = 3
ALERT_SEQUENCE = 5

# Detector setup
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(VIDEO_SOURCE)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 20

writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_idx = 0
consecutive_alerts = 0
alerts = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    if frame_idx % FRAME_SKIP != 0:
        writer.write(frame)
        continue

    boxes, _ = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    boxes = non_max_suppression(boxes, probs=None, overlapThresh=0.65)
    count = len(boxes)

    if count >= PEOPLE_THRESHOLD:
        consecutive_alerts += 1
    else:
        consecutive_alerts = 0

    if consecutive_alerts >= ALERT_SEQUENCE:
        alert = {
            "timestamp": datetime.now().isoformat(),
            "frame": frame_idx,
            "count": count
        }
        alerts.append(alert)
        consecutive_alerts = 0

    for (xA, yA, xB, yB) in boxes:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    if count >= PEOPLE_THRESHOLD:
        cv2.putText(frame, "Crowd Detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    writer.write(frame)

cap.release()
writer.release()

# Save alert log
with open(ALERT_LOG, 'w') as f:
    json.dump(alerts, f, indent=2)

# Plot alert timeline
timestamps = [datetime.fromisoformat(a['timestamp']) for a in alerts]
frames = [a['frame'] for a in alerts]

plt.figure(figsize=(10, 4))
plt.scatter(timestamps, frames, color='red')
plt.xlabel('Timestamp')
plt.ylabel('Frame Number')
plt.title('Crowd Detection Timeline')
plt.grid(True)
plt.tight_layout()
plt.savefig('alert_timeline.png')

# Download outputs
files.download(ALERT_LOG)
files.download('alert_timeline.png')
files.download(OUTPUT_VIDEO)
