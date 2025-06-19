import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PhoneDetectorImage:
    def __init__(self, confidence_threshold=0.7, display_output=True):
        """
        Initialize the phone detector for images
        Args:
            confidence_threshold (float): Confidence threshold for detections
            display_output (bool): Whether to display the image output
        """
        self.confidence_threshold = confidence_threshold
        self.display_output = display_output
        
        # Initialize YOLOv8 model
        logger.info("Loading YOLOv8 model...")
        self.model = YOLO('/Users/amansubash/Downloads/best2_specialcase_phone.pt')
        
        self.detection_count = 0

    def process_image(self, image_path):
        """
        Process a single image and detect phones
        Args:
            image_path (str): Path to the image file
        Returns:
            Processed image and number of detections
        """
        try:
            # Read the image
            frame = cv2.imread(image_path)
            if frame is None:
                logger.error(f"Could not read image: {image_path}")
                return None, 0
            
            logger.info(f"Processing image: {image_path}")
            logger.info(f"Image dimensions: {frame.shape}")
            
            # Run YOLOv8 inference
            results = self.model(frame)
            
            # Process detections
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Check if detection meets confidence threshold
                        # Note: Assuming class 0 is phone in your custom model
                        if conf >= self.confidence_threshold:
                            xyxy = box.xyxy[0].cpu().numpy()  # get box coordinates
                            x1, y1, x2, y2 = map(int, xyxy)
                            
                            # Store detection info
                            detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': conf,
                                'class': cls
                            })
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Add label with confidence
                            label = f'Phone: {conf:.2f}'
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            
                            # Draw label background
                            cv2.rectangle(frame, 
                                        (x1, y1 - label_size[1] - 10), 
                                        (x1 + label_size[0], y1), 
                                        (0, 255, 0), -1)
                            
                            # Draw label text
                            cv2.putText(frame,
                                      label,
                                      (x1, y1 - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.5,
                                      (0, 0, 0),
                                      2)
            
            # Add detection count to image
            count_text = f"Phones detected: {len(detections)}"
            cv2.putText(frame,
                       count_text,
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1,
                       (0, 255, 0),
                       2)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame,
                       f"Processed: {timestamp}",
                       (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       (255, 255, 255),
                       1)
            
            logger.info(f"Found {len(detections)} phone(s) in the image")
            
            # Print detection details
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det['bbox']
                logger.info(f"Phone {i+1}: Confidence={det['confidence']:.3f}, "
                           f"BBox=({x1},{y1},{x2},{y2})")
            
            return frame, len(detections)

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return None, 0

    def run(self, image_path):
        """Main method to run the detection pipeline on an image"""
        try:
            # Check if image file exists
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return
            
            logger.info("Starting image processing...")
            
            # Process the image
            processed_image, detection_count = self.process_image(image_path)
            
            if processed_image is None:
                logger.error("Failed to process image")
                return
            
            # Save the output image
            output_path = self.get_output_path(image_path)
            cv2.imwrite(output_path, processed_image)
            logger.info(f"Output saved to: {output_path}")
            
            # Display the output if enabled
            if self.display_output:
                # Resize image for display if it's too large
                display_image = self.resize_for_display(processed_image)
                
                cv2.imshow('Phone Detection Results', display_image)
                logger.info("Press any key to close the image window...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            # Print summary
            logger.info(f"Detection Summary:")
            logger.info(f"- Total phones detected: {detection_count}")
            logger.info(f"- Confidence threshold: {self.confidence_threshold}")
            logger.info(f"- Output image saved: {output_path}")
                
        except KeyboardInterrupt:
            logger.info("Detection interrupted by user")
        except Exception as e:
            logger.error(f"Error in image processing: {str(e)}")
        finally:
            cv2.destroyAllWindows()

    def resize_for_display(self, image, max_width=1200, max_height=800):
        """Resize image for display if it's too large"""
        h, w = image.shape[:2]
        
        if w <= max_width and h <= max_height:
            return image
        
        # Calculate scaling factor
        scale_w = max_width / w
        scale_h = max_height / h
        scale = min(scale_w, scale_h)
        
        # Resize image
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        return cv2.resize(image, (new_w, new_h))

    def get_output_path(self, input_path):
        """Generate output path for the processed image"""
        base_name = os.path.splitext(input_path)[0]
        extension = os.path.splitext(input_path)[1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_phone_detection_{timestamp}{extension}"

    def process_multiple_images(self, image_paths):
        """Process multiple images"""
        total_detections = 0
        processed_count = 0
        
        for image_path in image_paths:
            logger.info(f"\n--- Processing image {processed_count + 1}/{len(image_paths)} ---")
            processed_image, detection_count = self.process_image(image_path)
            
            if processed_image is not None:
                output_path = self.get_output_path(image_path)
                cv2.imwrite(output_path, processed_image)
                total_detections += detection_count
                processed_count += 1
                logger.info(f"Saved: {output_path}")
        
        logger.info(f"\n=== SUMMARY ===")
        logger.info(f"Processed {processed_count}/{len(image_paths)} images")
        logger.info(f"Total phones detected: {total_detections}")

def main():
    # Configuration
    CONFIG = {
        'IMAGE_PATH': '/Users/amansubash/Downloads/1.jpeg',
        'CONFIDENCE_THRESHOLD': 0.7,
        'DISPLAY_OUTPUT': True
    }
    
    try:
        # Initialize phone detector
        detector = PhoneDetectorImage(
            CONFIG['CONFIDENCE_THRESHOLD'],
            CONFIG['DISPLAY_OUTPUT']
        )
        
        # For single image
        detector.run(CONFIG['IMAGE_PATH'])
        
        # Uncomment below for multiple images
        # image_list = [
        #     '/Users/amansubash/Downloads/1.jpeg',
        #     '/Users/amansubash/Downloads/2.jpeg',
        #     '/Users/amansubash/Downloads/3.jpeg'
        # ]
        # detector.process_multiple_images(image_list)
        
    except Exception as e:
        logger.error(f"An error occurred in main: {str(e)}")

if __name__ == "__main__":
    main()
