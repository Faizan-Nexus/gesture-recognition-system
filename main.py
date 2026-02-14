"""
Simplified Gesture Recognition System - Testing Mode

Clean interface showing only predicted gesture.
Palm facing camera requirement enforced.
"""

import cv2
import numpy as np
import sys
from mediapipe_module import HandDetector
from normalization import normalize_landmarks
from calibration import Calibrator
from matcher import GestureMatcher
from actions import ActionMapper
from ui_helpers import HandAlignmentBox, draw_minimal_prediction, draw_instruction_text, draw_status_indicator
import config


class GestureRecognitionSystem:
    """Clean gesture recognition system for testing."""
    
    def __init__(self):
        """Initialize the system components."""
        self.detector = HandDetector()
        self.calibrator = Calibrator()
        self.matcher = GestureMatcher()
        self.action_mapper = ActionMapper()
        self.alignment_box = HandAlignmentBox(box_size=250)
        
        self.camera = None
        self.is_running = False
        self.calibration_complete = True  # Skip calibration - start in recognition mode
        self.last_gesture = "none"
        self.last_action = ""
        self.mode = "recognition"  # Start directly in recognition mode
    
    def initialize_camera(self) -> bool:
        """
        Initialize the camera.
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.camera = cv2.VideoCapture(0)
        
        if not self.camera.isOpened():
            print("Error: Could not open camera.")
            return False
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        
        return True
    
    def run_calibration(self):
        """Run the calibration phase."""
        print("\n" + "="*60)
        print("CALIBRATION MODE")
        print("="*60)
        print(f"Show your calibration gesture (palm facing camera).")
        print(f"Hold steady for {config.CALIBRATION_FRAMES} frames...")
        print("Press 'q' to skip calibration\n")
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            
            # Detect hand landmarks
            landmarks = self.detector.process_frame(frame)
            
            if landmarks is not None:
                # Draw landmarks
                self.detector.draw_landmarks(frame, landmarks)
                
                # Collect calibration sample
                is_complete = self.calibrator.collect_sample(landmarks)
                
                if is_complete:
                    print("\n✓ Calibration COMPLETE!")
                    self.calibration_complete = True
                    self.mode = "recognition"
                    cv2.waitKey(1000)  # Brief pause
                    break
            
            # Display progress
            current, required, percentage = self.calibrator.get_progress()
            progress_text = f"Progress: {current}/{required} ({percentage:.0f}%)"
            
            # Add status text to frame
            cv2.putText(frame, "CALIBRATION MODE", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, "Show palm facing camera", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, progress_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(frame, "Press 'q' to skip", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            cv2.imshow("Gesture Recognition", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nSkipping calibration...")
                self.calibration_complete = True
                self.mode = "recognition"
                break
    
    def run_recognition(self):
        """Main recognition loop with alignment box."""
        print("\n" + "="*60)
        print("GESTURE RECOGNITION MODE")
        print("="*60)
        print("System ready!")
        print("\nINSTRUCTIONS:")
        print("  1. Place your RIGHT HAND in the box")
        print("  2. Keep hand STABLE (don't move)")
        print("  3. Prediction appears when hand is stable")
        print("\nPress 'q' to quit\n")
        
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            # Flip frame for mirror view
            frame = cv2.flip(frame, 1)
            
            # Detect hand
            landmarks = self.detector.process_frame(frame)
            
            # Initialize status_text for all paths
            status_text = ""
            
            # Check hand in box and stability
            hand_in_box = self.alignment_box.check_hand_in_box(landmarks, frame.shape)
            hand_stable = self.alignment_box.is_hand_stable(landmarks, frame.shape)
            
            # Determine box color and status based on gesture recognition
            if hand_in_box and hand_stable:
                box_color = (0, 255, 0)  # Green - ready
                # We'll set status_text after prediction
                status_type = "success"
            elif hand_in_box:
                box_color = (0, 255, 255)  # Yellow - in box but moving
                status_text = "Keep hand STABLE (don't move)"
                status_type = "warning"
            else:
                box_color = (0, 0, 255)  # Red - out of box
                status_text = "Place hand inside the box"
                status_type = "error"
            
            # Draw alignment box
            self.alignment_box.draw_alignment_box(frame, color=box_color)
            
            # Draw instruction
            draw_instruction_text(frame, "Place hand in box & keep stable")
            
            # Draw status
            draw_status_indicator(frame, status_text, status=status_type)
            
            
            gesture_name = "none"
            confidence = 0.0
            
            # Only predict if hand is in box AND stable
            if landmarks is not None and hand_in_box and hand_stable:
                # Draw landmarks
                self.detector.draw_landmarks(frame, landmarks)
                
                # Normalize and apply calibration
                normalized = normalize_landmarks(landmarks)
                if self.calibrator.is_calibrated:
                    normalized = self.calibrator.apply_offset(normalized)
                
                # Match gesture
                gesture_name, confidence = self.matcher.match(normalized)
                
                # Map to action
                if gesture_name != "none":
                    self.last_gesture = gesture_name
                    action = self.action_mapper.get_action(gesture_name)
                    if action:
                        self.last_action = action
                    status_text = f"GESTURE: {gesture_name.upper()} ({int(confidence*100)}%)"
                else:
                    status_text = "No gesture recognized"
                
                # Draw LARGE gesture name at top-right
                h, w = frame.shape[:2]
                if gesture_name != "none":
                    gesture_text = gesture_name.upper()
                    
                    # Calculate text size and position
                    text_size = cv2.getTextSize(gesture_text, cv2.FONT_HERSHEY_DUPLEX, 2.0, 5)[0]
                    
                    text_x = w - text_size[0] - 20
                    text_y = 70
                    
                    # Black background
                    cv2.rectangle(frame, (text_x - 10, text_y - 55), (w - 10, text_y + 10), (0, 0, 0), -1)
                    
                    # Gesture name in bright cyan - LARGE
                    cv2.putText(frame, gesture_text, (text_x, text_y),
                               cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 255, 255), 5)
                
            elif landmarks is not None:
                # Just draw landmarks if hand visible but not stable/in box
                self.detector.draw_landmarks(frame)
            
            cv2.imshow("Gesture Recognition", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nExiting...")
                self.is_running = False
                break
    
    def _draw_clean_results(self, frame, gesture_name: str, confidence: float):
        """
        Draw clean, clear prediction on screen.
        
        Args:
            frame: Video frame
            gesture_name: Predicted gesture
            confidence: Confidence score
        """
        h, w = frame.shape[:2]
        
        # Black background for text
        cv2.rectangle(frame, (0, 0), (w, 200), (0, 0, 0), -1)
        
        # Gesture name (LARGE and CLEAR)
        if gesture_name != "none":
            # Show recognized gesture
            text = gesture_name.upper()
            color = (0, 255, 0)  # Green
            font_scale = 2.5
        else:
            # No gesture detected
            text = "NO GESTURE"
            color = (100, 100, 100)  # Gray
            font_scale = 1.5
        
        # Center text
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = 100
        
        # Draw gesture name
        cv2.putText(frame, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 3)
        
        # Confidence bar (if gesture detected)
        if gesture_name != "none":
            conf_text = f"Confidence: {confidence*100:.0f}%"
            cv2.putText(frame, conf_text, (text_x, text_y + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Guidelines at bottom
        cv2.rectangle(frame, (0, h - 80), (w, h), (0, 0, 0), -1)
        
        guidelines = "PALM FACING CAMERA | GOOD LIGHTING | HAND CENTERED | FULLY VISIBLE"
        guide_size = cv2.getTextSize(guidelines, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        guide_x = (w - guide_size[0]) // 2
        
        cv2.putText(frame, guidelines, (guide_x, h - 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.putText(frame, "Press 'Q' to quit", (guide_x, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    def run(self):
        """Main application loop."""
        print("\n" + "="*60)
        print("GESTURE RECOGNITION SYSTEM")
        print("="*60)
        print("Simplified System - Mean-Based Matching")
        print("="*60)
        
        if not self.initialize_camera():
            return
        
        self.is_running = True
        
        # Calibration phase
        if self.calibrator.reference_template is not None:
            self.run_calibration()
        else:
            print("\n[WARNING] No gestures registered yet.")
            print("[WARNING] Please register gestures first:")
            print("          python gesture_registration.py\n")
            self.mode = "recognition"
            self.calibration_complete = True
        
        # Recognition phase
        if self.is_running:
            self.run_recognition()
        
        self.cleanup()
    
    def cleanup(self):
        """Release resources."""
        if self.camera is not None:
            self.camera.release()
        cv2.destroyAllWindows()
        self.detector.close()
        
        print("\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)
        print(f"Last gesture: {self.last_gesture}")
        if self.last_action:
            print(f"Last action: {self.last_action}")
        print("="*60 + "\n")


def main():
    """Main entry point."""
    try:
        system = GestureRecognitionSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
