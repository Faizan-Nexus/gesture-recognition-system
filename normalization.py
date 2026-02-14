"""
Landmark normalization utilities for gesture recognition.

Normalizes MediaPipe hand landmarks to a canonical space by:
1. Translation: centering at wrist (landmark 0)
2. Scaling: normalizing by max pairwise distance
"""

import numpy as np
from typing import List, Any


def normalize_landmarks(landmarks: Any) -> np.ndarray:
    """
    Normalize MediaPipe hand landmarks to canonical space.
    
    Args:
        landmarks: MediaPipe hand landmarks (list, object with .landmark, or numpy array)
        
    Returns:
        np.ndarray: Normalized landmarks array of shape (21, 3)
    """
    # Handle different input formats
    if isinstance(landmarks, np.ndarray):
        # Already a numpy array (from data_collector.py)
        points = landmarks
    elif hasattr(landmarks, 'landmark'):
        # Old mediapipe.solutions API
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    else:
        # New mediapipe.tasks API - landmarks is already a list
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # Step 1: Translation - subtract wrist position (landmark 0)
    wrist = points[0].copy()
    translated = points - wrist
    
    # Step 2: Scaling - compute max pairwise distance
    max_distance = compute_max_distance(translated)
    
    # Avoid division by zero
    if max_distance < 1e-6:
        return translated
    
    # Scale by max distance
    normalized = translated / max_distance
    
    return normalized


def compute_max_distance(points: np.ndarray) -> float:
    """
    Compute the maximum Euclidean distance between any two points.
    
    Args:
        points: Array of shape (N, 3)
        
    Returns:
        float: Maximum pairwise distance
    """
    max_dist = 0.0
    n = len(points)
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(points[i] - points[j])
            if dist > max_dist:
                max_dist = dist
    
    return max_dist


def landmarks_to_array(landmarks: Any) -> np.ndarray:
    """
    Convert MediaPipe landmarks to numpy array.
    
    Args:
        landmarks: MediaPipe hand landmarks (list, object with .landmark, or numpy array)
        
    Returns:
        np.ndarray: Array of shape (21, 3)
    """
    # If already a numpy array, return as-is
    if isinstance(landmarks, np.ndarray):
        return landmarks
    # Handle both old API (landmarks.landmark) and new API (direct list)
    elif hasattr(landmarks, 'landmark'):
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    else:
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
