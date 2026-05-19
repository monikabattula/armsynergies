Put your input videos here (.mp4, .mov, .avi, .mkv, .webm).

Then from the project folder run:

  cd /path/to/armtracking
  source .venv/bin/activate   # if you use a venv
  pip install -r requirements.txt

All videos in ./videos -> CSVs in ./landmark_output/:
  python landmark_export/batch_holistic_videos.py

One specific file:
  python landmark_export/batch_holistic_videos.py --video /full/path/to/clip.mp4

Custom output path (single file):
  python landmark_export/batch_holistic_videos.py --video clip.mp4 --out /path/out.csv

Another folder of videos:
  python landmark_export/batch_holistic_videos.py --input-dir /path/to/vids --output-dir /path/to/csvs

Better tracking (slower): use a stronger pose model and/or lower confidence thresholds
  python landmark_export/batch_holistic_videos.py --video clip.mp4 --pose-quality full
  python landmark_export/batch_holistic_videos.py --video clip.mp4 --min-detection 0.4 --min-tracking 0.4
  python landmark_export/batch_holistic_videos.py --video clip.mp4 --min-hand-presence 0.4

See: python landmark_export/batch_holistic_videos.py -h

CSV files are written to ../landmark_output/ by default.

If model download fails (SSL), either:
  pip install certifi
or point to local .task files (pose + hand, not Holistic):
  export MEDIAPIPE_POSE_MODEL_PATH=/path/to/pose_landmarker_lite.task
  export MEDIAPIPE_HAND_MODEL_PATH=/path/to/hand_landmarker.task
