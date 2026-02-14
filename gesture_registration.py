"""
Simplified Gesture Registration - 60 Samples Per Gesture

Collects 60 samples, calculates mean template, saves to templates.py
Simple and fast registration process.
"""

import cv2
import numpy as np
import sys
import os
import cv2
import numpy as np

from mediapipe_module import HandDetector
from normalization import normalize_landmarks
from matcher import GestureMatcher
from ui_helpers import HandAlignmentBox, draw_instruction_text, draw_status_indicator
import config


class SimplifiedGestureRegistration:
    """Simple 60-sample gesture registration."""
    
    def __init__(self):
        """Initialize registration system."""
        self.detector = HandDetector()
        self.alignment_box = HandAlignmentBox(box_size=450)  # Larger box
        self.matcher = GestureMatcher()  # For showing predictions
        self.camera = None
        self.registered_gestures = []
        
    def initialize_camera(self) -> bool:
        """Initialize camera."""
        self.camera = cv2.VideoCapture(0)
        
        if not self.camera.isOpened():
            print("Error: Could not open camera.")
            return False
        
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        
        return True
    
    def check_hand_quality(self, frame, landmarks) -> tuple:
        """
        Check hand visibility and quality.
        Returns: (is_good, messages)
        """
        messages = []
        is_good = True
        
        # Check if hand detected
        if landmarks is None:
            return False, ["❌ NO HAND - Show your RIGHT hand (palm facing camera)"]
        
        # Check hand position (should be centered)
        h, w = frame.shape[:2]
        wrist = landmarks[0]
        
        if wrist.x < 0.2 or wrist.x > 0.8:
            messages.append("⚠️ Center hand in frame")
            is_good = False
        
        if wrist.y < 0.1 or wrist.y > 0.9:
            messages.append("⚠️ Move hand to center height")
            is_good = False
        
        # Check lighting
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hand_brightness = np.mean(gray[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)])
        
        if hand_brightness < 80:
            messages.append("⚠️ Improve lighting")
            is_good = False
        
        if is_good:
            messages.append("✓ GOOD - Press SPACE to capture")
        
        return is_good, messages
    
    def collect_samples(self, gesture_name: str, num_samples: int = 60) -> list:
        """Collect samples for one gesture."""
        print(f"\n{'='*60}")
        print(f"COLLECTING: {gesture_name.upper()}")
        print(f"{'='*60}")
        print(f"Target: {num_samples} samples")
        print(f"\n📋 GUIDELINES:")
        print(f"  ✓ Use RIGHT HAND only")
        print(f"  ✓ PALM FACING CAMERA (front of hand)")
        print(f"  ✓ Keep hand fully visible")
        print(f"  ✓ Good lighting required")
        print(f"  ✓ Center hand in frame")
        print(f"\nPress SPACE to capture each sample")
        print(f"Press ESC to cancel")
        print(f"{'='*60}\n")
        
        samples = []
        
        while len(samples) < num_samples:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            landmarks = self.detector.process_frame(frame)
            
            
            # Check if hand is in alignment box
            hand_in_box = self.alignment_box.check_hand_in_box(landmarks, frame.shape)
            
            # Determine box color and readiness
            if hand_in_box:
                box_color = (0, 255, 0)  # Green
                can_capture = True
            else:
                box_color = (0, 0, 255)  # Red
                can_capture = False
            
            # Draw alignment box
            self.alignment_box.draw_alignment_box(frame, color=box_color)
            
            if landmarks is not None:
                self.detector.draw_landmarks(frame, landmarks)
            
            
            # Show gesture name and prediction at top right
            h, w = frame.shape[:2]
            gesture_text = f"{gesture_name.upper()}"
            text_size = cv2.getTextSize(gesture_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = w - text_size[0] - 20
            text_y = 50
            
            # Get current prediction if hand detected
            prediction_text = ""
            if landmarks is not None and hand_in_box:
                normalized = normalize_landmarks(landmarks)
                pred_name, confidence = self.matcher.match(normalized)
                if pred_name != "none":
                    prediction_text = f"Predicts: {pred_name} ({int(confidence*100)}%)"
            
            # Black background for text
            bg_height = 90 if prediction_text else 45
            cv2.rectangle(frame, (text_x - 10, text_y - 35), (w - 10, text_y + bg_height), (0, 0, 0), -1)
            
            # Gesture name
            cv2.putText(frame, gesture_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
            # Prediction
            if prediction_text:
                cv2.putText(frame, prediction_text, (text_x, text_y + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Gesture Registration", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                if landmarks is not None and hand_in_box:
                    sample = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
                    samples.append(sample)
                    print(f"  [{len(samples)}/{num_samples}] Captured!")
                elif landmarks is None:
                    print("  No hand detected!")
                else:
                    print("  Hand must be inside the box!")
            elif key == 27:  # ESC
                print(f"\nCancelled")
                return []
        
        print(f"\n✓ Collection complete: {len(samples)} samples")
        return samples
    
    def calculate_mean_template(self, samples: list, gesture_name: str) -> dict:
        """Calculate mean template from samples."""
        print(f"\nCalculating mean template...")
        
        # Normalize all samples
        normalized_samples = []
        for sample in samples:
            normalized = normalize_landmarks(sample)
            normalized_samples.append(normalized)
        
        normalized_samples = np.array(normalized_samples)
        
        # Calculate mean
        mean_template = np.mean(normalized_samples, axis=0)
        
        print(f"  ✓ Mean template calculated")
        
        template = {
            "gesture": gesture_name,
            "samples_count": len(samples),
            "mean": mean_template
        }
        
        return template
    
    def save_template(self, template: dict, gesture_name: str):
        """Update templates.py with new gesture."""
        print(f"\nUpdating templates.py...")
        
        templates_file = os.path.join(os.path.dirname(__file__), 'templates.py')
        
        # Generate template entry
        template_code = f'''    "{gesture_name}": {{
        "mean": np.array({self._array_to_code(template['mean'])}),
        "samples_count": {template['samples_count']}
    }}'''
        
        # Read existing file
        with open(templates_file, 'r') as f:
            lines = f.readlines()
        
        # Find GESTURE_TEMPLATES = { line
        dict_start = -1
        for i, line in enumerate(lines):
            if 'GESTURE_TEMPLATES = {' in line:
                dict_start = i
                break
        
        if dict_start == -1:
            print("Error: Could not find GESTURE_TEMPLATES in templates.py")
            return
        
        # Insert new template
        if '}' in lines[dict_start]:
            # Empty dict, replace line
            lines[dict_start] = f"GESTURE_TEMPLATES = {{\n{template_code}\n}}\n"
        else:
            # Has entries, add comma and new entry
            dict_end = -1
            for i in range(dict_start + 1, len(lines)):
                if lines[i].strip() == '}':
                    dict_end = i
                    break
            
            if dict_end > dict_start + 1:
                # Add comma to previous entry
                lines[dict_end - 1] = lines[dict_end - 1].rstrip() + ',\n'
            
            # Insert new template before closing brace
            lines.insert(dict_end, template_code + '\n')
        
        # Write back
        with open(templates_file, 'w') as f:
            f.writelines(lines)
        
        print(f"  ✓ Added {gesture_name} to templates.py")
    
    def _array_to_code(self, arr: np.ndarray) -> str:
        """Convert numpy array to readable Python code."""
        rows = []
        for row in arr:
            values = ', '.join([f'{v:.6f}' for v in row])
            rows.append(f'[{values}]')
        # Use actual newlines, not literal \n
        return '[\n            ' + ',\n            '.join(rows) + '\n        ]'
    
    def register_gesture(self, gesture_name: str) -> bool:
        """Complete registration workflow for one gesture."""
        print(f"\n{'='*60}")
        print(f"REGISTERING: {gesture_name.upper()}")
        print(f"{'='*60}\n")
        
        # Collect 60 samples
        samples = self.collect_samples(gesture_name, num_samples=60)
        if not samples or len(samples) < 30:
            print(f"❌ Not enough samples collected ({len(samples)}). Minimum 30 required.")
            return False
        
        # Calculate mean template
        template = self.calculate_mean_template(samples, gesture_name)
        
        # Save template
        self.save_template(template, gesture_name)
        
        self.registered_gestures.append(gesture_name)
        
        print(f"\n✅ {gesture_name} registered successfully!")
        print(f"{'='*60}\n")
        
        return True
    
    def run(self):
        """Main interactive loop."""
        print("\n" + "="*60)
        print("SIMPLIFIED GESTURE REGISTRATION SYSTEM")
        print("="*60)
        print("Features:")
        print("  - 60 samples per gesture")
        print("  - Mean-based templates")
        print("  - Palm facing camera only")
        print("  - Quality guidelines enforced")
        print("="*60 + "\n")
        
        if not self.initialize_camera():
            return
        
        while True:
            gesture_name = input("\nEnter gesture name (or 'q' to quit): ").strip().lower()
            
            if gesture_name == 'q':
                break
            
            if not gesture_name:
                print("Invalid name! Try again.")
                continue
            
            success = self.register_gesture(gesture_name)
            
            if success:
                print(f"\nRegistered gestures: {', '.join(self.registered_gestures)}")
        
        self.cleanup()
        
        print(f"\n{'='*60}")
        print("REGISTRATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total gestures registered: {len(self.registered_gestures)}")
        for gesture in self.registered_gestures:
            print(f"  ✓ {gesture}")
        print(f"\nTemplates saved to: templates.py")
        print(f"{'='*60}\n")
    
    def cleanup(self):
        """Release resources."""
        if self.camera is not None:
            self.camera.release()
        cv2.destroyAllWindows()
        self.detector.close()


def main():
    """Main entry point."""
    try:
        system = SimplifiedGestureRegistration()
        system.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
