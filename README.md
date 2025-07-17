Real-Time Stream Simulation & Event Trigger
This project simulates a video stream using OpenCV and applies a real-time detection system to identify crowded scenes. An alert is triggered if a crowd (3 or more people) is detected in 5 consecutive frames. The system logs alert data, provides visual overlays, and generates a timeline plot of alert occurrences.

🔍 Features
Simulates a live video stream using OpenCV

Performs object/person detection on every 3rd frame

Triggers an alert if 3 or more people are detected in 5 consecutive frames

Logs alerts in .json or .txt format

Plots a timeline chart of alert events

(Optional) Overlays alerts on frames and exports the output video

🗂 Project Structure
real-time-event-trigger/
├── stream_simulation.py        # Main simulation and detection script
├── alert_utils.py              # Functions for alert logging and timeline plotting
├── outputs/                    # Output videos, logs, and plots
├── video_input/                # Source videos
├── requirements.txt            # Required Python libraries
└── README.md

⚙️ Installation
Install required dependencies:
pip install -r requirements.txt
🚀 Usage
Run the simulation and detection pipeline:
python stream_simulation.py --video ./video_input/sample.mp4
Optional Arguments:
--save-video: Save output video with overlayed alerts

--log-format: Choose between json or txt for alert logging (default: json)

--frame-skip: Process every nth frame (default: 3)

🧠 Alert Logic
Detection is performed on every 3rd frame

If ≥ 3 people are detected in 5 consecutive processed frames, an alert is triggered

Alerts are:

Logged to a file

Visualized on a timeline plot

Optionally overlaid on video frames
🛠 Requirements
opencv-python

numpy

matplotlib

pandas

tqdm

You may also need:

torch, torchvision (if using a pretrained detection model like YOLO or Faster R-CNN)

