import cv2
import easyocr
from ultralytics import YOLO
import numpy as np
from collections import Counter

class RIPASDetector:
    def __init__(self):
        # Load standard YOLOv8 model (nano version for speed)
        self.model = YOLO('yolov8n.pt')
        # Initialize EasyOCR
        self.reader = easyocr.Reader(['en'])
        # Class IDs for vehicles in COCO dataset: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
        self.vehicle_classes = [2, 3, 5, 7]

    def _preprocess_crop(self, crop):
        """Standard high-quality pre-processing for OCR."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Resize to improve small plate reading
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        # Simple contrast enhancement
        gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=0)
        return gray

    def process_image(self, image):
        """
        Process a single image to find a vehicle and its license plate.
        Returns: plate_text, annotated_image, confidence
        """
        results = self.model(image, verbose=False)
        plate_text = "UNKNOWN"
        annotated_image = image.copy()
        best_conf = 0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                if int(box.cls) in self.vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    vehicle_crop = image[y1:y2, x1:x2]
                    if vehicle_crop.size == 0: continue
                    
                    processed = self._preprocess_crop(vehicle_crop)
                    ocr_results = self.reader.readtext(processed)
                    
                    for (bbox, text, prob) in ocr_results:
                        clean_text = "".join(e for e in text if e.isalnum()).upper()
                        if len(clean_text) >= 3 and prob > 0.35:
                            if prob > best_conf:
                                plate_text = clean_text
                                best_conf = prob
                                cv2.putText(annotated_image, f"{plate_text} ({int(prob*100)}%)", (x1, y1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    
                    if plate_text != "UNKNOWN":
                        return plate_text, annotated_image, best_conf
                        
        return plate_text, annotated_image, 0.0

    def process_video(self, video_path):
        """
        Balanced tracking-based video processor.
        Uses a 2-vote consensus to ensure accuracy while maintaining speed.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        # Track storage: id -> list of detected plates
        track_data = {} 
        finalized_tracks = set()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            # Balanced Skip: Process every 3rd frame (10 FPS at 30 FPS source)
            if frame_count % 3 != 0: continue
            
            progress = frame_count / total_frames
            
            # Use standard resolution for better small plate detection
            results = self.model.track(frame, persist=True, verbose=False)
            
            annotated_frame = frame.copy()
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)

                for box, track_id, cls in zip(boxes, ids, classes):
                    if cls not in self.vehicle_classes: continue
                    
                    x1, y1, x2, y2 = box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    # Only scan if not already finalized
                    if track_id not in finalized_tracks:
                        area = (x2 - x1) * (y2 - y1)
                        if area > 10000: # "Prime Zone" size
                            vehicle_crop = frame[y1:y2, x1:x2]
                            # Use high-quality pre-processing for the candidate scan
                            processed = self._preprocess_crop(vehicle_crop)
                            ocr_results = self.reader.readtext(processed)
                            
                            for (_, text, prob) in ocr_results:
                                clean_text = "".join(e for e in text if e.isalnum()).upper()
                                # Require 3+ chars and decent confidence
                                if len(clean_text) >= 3 and prob > 0.4:
                                    if track_id not in track_data: track_data[track_id] = []
                                    track_data[track_id].append(clean_text)
                                    
                                    # Voting logic: Finalize if we see the SAME plate 2 times
                                    counts = Counter(track_data[track_id])
                                    most_common, frequency = counts.most_common(1)[0]
                                    
                                    if frequency >= 2:
                                        finalized_tracks.add(track_id)
                                        yield {
                                            "plate": most_common,
                                            "frame": annotated_frame,
                                            "progress": progress
                                        }
                                        break
            
            # UI update frequency (Balanced)
            if frame_count % 10 == 0:
                yield {
                    "plate": "UNKNOWN",
                    "frame": annotated_frame,
                    "progress": progress
                }

        cap.release()

# Singleton instance
_detector = None
def get_detector():
    global _detector
    if _detector is None:
        _detector = RIPASDetector()
    return _detector
