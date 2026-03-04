#src/evaluation/event_logger.py

import csv
from pathlib import Path


class EventLogger:
    """
    Logs vehicle crossing events to CSV.
    """

    def __init__(self, output_path="outputs/crossing_events.csv"):
        self.output_path = Path(output_path)
        self.events = []

    def log_event(self, frame_idx, fps, class_id, class_name, track_id=None):
        time_sec = frame_idx / fps

        self.events.append({
            "frame": frame_idx,
            "time_sec": round(time_sec, 3),
            "class_id": class_id,
            "class_name": class_name,
            "track_id": track_id
        })

    def save(self):
        if not self.events:
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["frame", "time_sec", "class_id", "class_name", "track_id"]
            )

            writer.writeheader()

            for event in self.events:
                writer.writerow(event)

        print("Event log saved:", self.output_path)
