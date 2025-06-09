import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import logging
from deep_sort_realtime.deepsort_tracker import DeepSort

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PhoneDetectorTracker:
    def __init__(self, video_path, confidence_threshold=0.2, display_output=True):
        """
        Initialize the phone detector and tracker
        Args:
            video_path (str): Path to the video file
            confidence_threshold (float): Confidence threshold for detections
            display_output (bool): Whether to display the video output
        """
        self.video_path = video_path
        self.confidence_threshold = confidence_threshold
        self.display_output = display_output
        
        # Initialize YOLOv8 model
        logger.info("Loading YOLOv8 model...")
        self.model = YOLO('yolov8l.pt')
        
        # Initialize DeepSORT tracker
        logger.info("Initializing DeepSORT tracker...")
        self.tracker = DeepSort(
            max_age=30,            # Keep track of disappeared objects for 30 frames
            n_init=3,              # Require 3 detections to confirm track
            nms_max_overlap=0.7,   # Non-maxima suppression threshold
            max_cosine_distance=0.3,  # Threshold for feature similarity
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True
        )
        
        self.frame_count = 0

    def process_frame(self, frame):
        """
        Process a single frame and detect/track phones
        Args:
            frame: Video frame to process
        Returns:
            Processed frame and number of active tracks
        """
        try:
            # Run YOLOv8 inference
            results = self.model(frame)
            
            # Format detections for DeepSORT
            deepsort_detections = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Check if detection is a phone (class 67) and meets confidence threshold
                    if cls == 67 and conf >= self.confidence_threshold:
                        xyxy = box.xyxy[0].cpu().numpy()  # get box coordinates
                        x1, y1, x2, y2 = map(int, xyxy)
                        w = x2 - x1
                        h = y2 - y1
                        
                        # Add detection for DeepSORT
                        deepsort_detections.append(([x1, y1, w, h], conf, "phone"))

            # Update tracks
            tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)
            active_tracks = 0
            
            # Process and draw tracks
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                # Get track box
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                
                # Draw bounding box and track ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame,
                          f'Phone #{track.track_id}',
                          (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5,
                          (0, 255, 0),
                          2)
                
                active_tracks += 1
            
            return frame, active_tracks

        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame, 0

    def run(self):
        """Main method to run the detection and tracking pipeline"""
        try:
            logger.info("Starting video processing...")
            cap = cv2.VideoCapture(self.video_path)
            
            if not cap.isOpened():
                logger.error(f"Error: Could not open video source: {self.video_path}")
                return

            # Get video properties for output
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Initialize video writer
            output_path = self.video_path.rsplit('.', 1)[0] + '_tracked.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Error: Could not read frame")
                    break

                self.frame_count += 1
                
                # Process the frame
                processed_frame, active_tracks = self.process_frame(frame)
                
                # Add frame counter and total phones in frame
                cv2.putText(
                    processed_frame,
                    f"Phones in frame: {active_tracks}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # Write frame to output video
                out.write(processed_frame)

                # Display the output if enabled
                if self.display_output:
                    cv2.imshow('Phone Detection and Tracking', processed_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User requested stop")
                        break

        except KeyboardInterrupt:
            logger.info("Stopping detection...")
        except Exception as e:
            logger.error(f"Error in video processing: {str(e)}")
        finally:
            cap.release()
            out.release()
            if self.display_output:
                cv2.destroyAllWindows()
            logger.info("Video processing ended")
            logger.info(f"Total frames processed: {self.frame_count}")
            logger.info(f"Output saved to: {output_path}")

def main():
    # Configuration
    CONFIG = {
        'VIDEO_PATH': '/Users/amansubash/Downloads/phone.mp4',
        'CONFIDENCE_THRESHOLD': 0.5,
        'DISPLAY_OUTPUT': True
    }
    
    try:
        # Initialize phone detector and tracker
        detector = PhoneDetectorTracker(
            CONFIG['VIDEO_PATH'],
            CONFIG['CONFIDENCE_THRESHOLD'],
            CONFIG['DISPLAY_OUTPUT']
        )
        
        # Start detection and tracking
        detector.run()
        
    except Exception as e:
        logger.error(f"An error occurred in main: {str(e)}")

if __name__ == "__main__":
    main()
