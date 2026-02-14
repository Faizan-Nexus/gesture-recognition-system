"""
User offset calibration for personalized gesture recognition.

Collects samples of a reference gesture (open_palm) and computes
a user-specific offset to adapt the system to individual hand
characteristics.
"""

import numpy as np
from typing import List, Optional
from normalization import normalize_landmarks
from templates import get_template
import config


class Calibrator:
    """Handles user-specific calibration for gesture recognition."""
    
    def __init__(self, reference_gesture: str = "stop"):
        """
        Initialize the calibrator.
        
        Args:
            reference_gesture: Name of gesture to use for calibration (default: "stop")
        """
        self.reference_gesture = reference_gesture
        
        # Try to load reference template, but don't crash if it doesn't exist
        try:
            template_dict = get_template(reference_gesture)
            if template_dict:
                self.reference_template = template_dict['mean']
            else:
                self.reference_template = None
        except (KeyError, Exception) as e:
            print(f"[WARNING] Calibration template '{reference_gesture}' not found.")
            print(f"[WARNING] Please register gestures first using gesture_registration.py")
            self.reference_template = None
        self.samples: List[np.ndarray] = []
        self.user_offset: Optional[np.ndarray] = None
        self.is_calibrated = False
        self.required_frames = config.CALIBRATION_FRAMES
    
    def collect_sample(self, landmarks) -> bool:
        """
        Collect a calibration sample.
        
        Args:
            landmarks: Raw hand landmarks
            
        Returns:
            bool: True if calibration is complete, False otherwise
        """
        # Skip if no reference template
        if self.reference_template is None:
            return True  # Skip calibration
        
        # Normalize landmarks
        normalized = normalize_landmarks(landmarks)
        self.samples.append(normalized)
        
        # Check if we have enough samples
        if len(self.samples) >= self.required_frames:
            self.compute_offset()
            return True
        
        return False
    
    def compute_offset(self) -> np.ndarray:
        """
        Compute the user offset from collected samples.
        
        Returns:
            np.ndarray: The computed user offset (21, 3)
        """
        # Compute mean of all user samples
        user_mean = np.mean(self.samples, axis=0)
        
        # Compute offset as difference from template
        self.user_offset = user_mean - self.reference_template
        self.is_calibrated = True
        
        return self.user_offset
    
    def apply_correction(self, normalized_landmarks: np.ndarray) -> np.ndarray:
        """
        Apply user offset correction to normalized landmarks.
        
        Args:
            normalized_landmarks: Normalized landmarks (21, 3)
            
        Returns:
            np.ndarray: Corrected landmarks (21, 3)
        """
        if not self.is_calibrated:
            # If not calibrated, return unchanged
            return normalized_landmarks
        
        return normalized_landmarks - self.user_offset
    
    def get_progress(self) -> tuple:
        """
        Get calibration progress.
        
        Returns:
            tuple: (current_samples, required_samples, percentage)
        """
        current = len(self.samples)
        required = self.required_frames
        percentage = (current / required) * 100
        return (current, required, percentage)
    
    def reset(self):
        """Reset calibration data."""
        self.samples = []
        self.user_offset = None
        self.is_calibrated = False
