"""
Simplified gesture matcher using Euclidean distance.

Compares input gesture against mean templates.
"""

import numpy as np
from typing import Tuple, Optional
from collections import deque

import config
from templates import GESTURE_TEMPLATES


class GestureMatcher:
    """Simple distance-based gesture matching."""
    
    def __init__(self, distance_threshold: float = 0.60):  # High threshold for dataset gestures
        """
        Initialize the matcher.
        
        Args:
            distance_threshold: Maximum distance for valid match
        """
        self.distance_threshold = distance_threshold
        self.smoothing_buffer = deque(maxlen=config.SMOOTHING_BUFFER_SIZE)
        self.debug_mode = False  # Disable debug output
    
    def match(self, normalized_landmarks: np.ndarray) -> Tuple[str, float]:
        """
        Match input against all templates.
        
        Args:
            normalized_landmarks: Normalized landmark array (21, 3)
            
        Returns:
            tuple: (gesture_name, confidence) or ("none", 0.0)
        """
        if not GESTURE_TEMPLATES:
            return ("none", 0.0)
        
        flattened = normalized_landmarks.flatten()
        
        best_gesture = "none"
        best_distance = float('inf')
        
        # Compare against all templates
        for gesture_name, template_data in GESTURE_TEMPLATES.items():
            mean_template = template_data['mean'].flatten()
            
            # Euclidean distance
            distance = np.linalg.norm(flattened - mean_template)
            
            if self.debug_mode and distance < self.distance_threshold * 1.5:  # Show promising matches
                print(f"  {gesture_name}: distance={distance:.4f}")
            
            if distance < best_distance:
                best_distance = distance
                best_gesture = gesture_name
        
        # Check threshold
        if best_distance > self.distance_threshold:
            if self.debug_mode:
                print(f"  REJECTED: {best_gesture} (distance {best_distance:.4f} > threshold {self.distance_threshold:.4f})")
            best_gesture = "none"
            confidence = 0.0
        else:
            # Convert distance to confidence (0-1)
            confidence = max(0.0, 1.0 - (best_distance / self.distance_threshold))
            if self.debug_mode:
                print(f"  MATCH: {best_gesture} (distance={best_distance:.4f}, confidence={confidence:.2f})")
        
        # Smoothing
        self.smoothing_buffer.append(best_gesture)
        
        if len(self.smoothing_buffer) >= config.SMOOTHING_BUFFER_SIZE:
            # Majority voting
            gesture_counts = {}
            for g in self.smoothing_buffer:
                gesture_counts[g] = gesture_counts.get(g, 0) + 1
            
            final_gesture = max(gesture_counts, key=gesture_counts.get)
            return (final_gesture, confidence)
        
        return (best_gesture, confidence)
    
    def reset(self):
        """Reset the smoothing buffer."""
        self.smoothing_buffer.clear()
