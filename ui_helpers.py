"""
UI Helper Functions for Gesture Recognition System

Provides hand alignment box, stability detection, and minimal UI displays.
"""

import cv2
import numpy as np
from collections import deque


class HandAlignmentBox:
    """Manages hand alignment box and stability tracking."""
    
    def __init__(self, box_size: int = 250):
        """
        Initialize alignment box system.
        
        Args:
            box_size: Size of square alignment box in pixels
        """
        self.box_size = box_size
        self.position_history = deque(maxlen=3)  # Track 3 frames
        
    def get_box_coords(self, frame_shape) -> tuple:
        """
        Get alignment box coordinates (centered).
        
        Args:
            frame_shape: Shape of the frame (h, w, c)
            
        Returns:
            tuple: (x1, y1, x2, y2) box coordinates
        """
        h, w = frame_shape[:2]
        
        # Center the box
        x1 = (w - self.box_size) // 2
        y1 = (h - self.box_size) // 2
        x2 = x1 + self.box_size
        y2 = y1 + self.box_size
        
        return (x1, y1, x2, y2)
    
    def draw_alignment_box(self, frame, color=(0, 255, 0), thickness=3):
        """
        Draw alignment box on frame.
        
        Args:
            frame: Video frame
            color: Box color (BGR)
            thickness: Line thickness
        """
        x1, y1, x2, y2 = self.get_box_coords(frame.shape)
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw corner markers for better visibility
        corner_len = 20
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness + 1)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness + 1)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness + 1)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness + 1)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness + 1)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness + 1)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness + 1)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness + 1)
    
    def check_hand_in_box(self, landmarks, frame_shape) -> bool:
        """
        Check if hand center (wrist) is inside alignment box.
        
        Args:
            landmarks: Hand landmarks from MediaPipe
            frame_shape: Shape of the frame
            
        Returns:
            bool: True if hand is inside box
        """
        if landmarks is None:
            return False
        
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = self.get_box_coords(frame_shape)
        
        # Get wrist position (landmark 0)
        wrist = landmarks[0]
        wrist_x = int(wrist.x * w)
        wrist_y = int(wrist.y * h)
        
        # Check if wrist is inside box
        return x1 <= wrist_x <= x2 and y1 <= wrist_y <= y2
    
    def get_hand_center(self, landmarks, frame_shape) -> tuple:
        """
        Get hand center position (wrist).
        
        Args:
            landmarks: Hand landmarks
            frame_shape: Frame shape
            
        Returns:
            tuple: (x, y) position or None
        """
        if landmarks is None:
            return None
        
        h, w = frame_shape[:2]
        wrist = landmarks[0]
        return (int(wrist.x * w), int(wrist.y * h))
    
    def is_hand_stable(self, landmarks, frame_shape, threshold: int = 10) -> bool:
        """
        Check if hand has been stable for last 3 frames.
        
        Args:
            landmarks: Current hand landmarks
            frame_shape: Frame shape
            threshold: Maximum pixel movement to be considered stable
            
        Returns:
            bool: True if hand is stable
        """
        current_pos = self.get_hand_center(landmarks, frame_shape)
        
        if current_pos is None:
            self.position_history.clear()
            return False
        
        self.position_history.append(current_pos)
        
        # Need 3 frames of history
        if len(self.position_history) < 3:
            return False
        
        # Check if all positions are within threshold
        positions = list(self.position_history)
        for i in range(len(positions) - 1):
            dx = abs(positions[i][0] - positions[i+1][0])
            dy = abs(positions[i][1] - positions[i+1][1])
            
            if dx > threshold or dy > threshold:
                return False
        
        return True
    
    def reset_stability(self):
        """Reset stability tracking."""
        self.position_history.clear()


def draw_minimal_prediction(frame, gesture_name: str, confidence: float, 
                           position: str = "top-right"):
    """
    Draw minimal prediction text in corner.
    
    Args:
        frame: Video frame
        gesture_name: Predicted gesture name
        confidence: Confidence score (0-1)
        position: Position ("top-right", "top-left", etc.)
    """
    h, w = frame.shape[:2]
    
    if gesture_name == "none":
        return
    
    # Format text
    text = f"GESTURE: {gesture_name.upper()} ({confidence*100:.0f}%)"
    
    # Text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (0, 255, 0)  # Green
    
    # Get text size
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Calculate position
    if position == "top-right":
        x = w - text_size[0] - 10
        y = 30
    elif position == "top-left":
        x = 10
        y = 30
    elif position == "bottom-right":
        x = w - text_size[0] - 10
        y = h - 10
    else:  # bottom-left
        x = 10
        y = h - 10
    
    # Draw background
    padding = 5
    cv2.rectangle(frame, 
                 (x - padding, y - text_size[1] - padding),
                 (x + text_size[0] + padding, y + padding),
                 (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


def draw_instruction_text(frame, text: str, position: str = "top-center"):
    """
    Draw instruction text on frame.
    
    Args:
        frame: Video frame
        text: Instruction text
        position: Position on screen
    """
    h, w = frame.shape[:2]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (0, 255, 255)  # Yellow
    
    # Get text size
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Calculate position
    if position == "top-center":
        x = (w - text_size[0]) // 2
        y = 40
    elif position == "bottom-center":
        x = (w - text_size[0]) // 2
        y = h - 20
    else:  # center
        x = (w - text_size[0]) // 2
        y = h // 2
    
    # Draw background
    padding = 8
    cv2.rectangle(frame,
                 (x - padding, y - text_size[1] - padding),
                 (x + text_size[0] + padding, y + padding),
                 (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


def draw_status_indicator(frame, text: str, status: str = "info"):
    """
    Draw status indicator at bottom of frame.
    
    Args:
        frame: Video frame
        text: Status text
        status: Status type ("info", "success", "warning", "error")
    """
    h, w = frame.shape[:2]
    
    # Color based on status
    colors = {
        "info": (255, 255, 255),     # White
        "success": (0, 255, 0),      # Green
        "warning": (0, 255, 255),    # Yellow
        "error": (0, 0, 255)         # Red
    }
    color = colors.get(status, (255, 255, 255))
    
    # Draw text at bottom
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x = (w - text_size[0]) // 2
    y = h - 15
    
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
