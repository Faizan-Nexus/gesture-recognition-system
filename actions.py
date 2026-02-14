"""
Gesture Actions Module

This module maps recognized gestures to specific actions that can be executed.
"""

import time

# Map gestures to action functions
GESTURE_ACTIONS = {
    "Go": "navigation_forward",
    "Come": "navigation_back",
    "Nice": "approval_positive",
    "Ok": "confirmation_ok",
}

class ActionMapper:
    """Executes actions based on recognized gestures."""
    
    def __init__(self, cooldown_seconds=1.5):
        """
        Initialize the action executor.
        
        Args:
            cooldown_seconds: Minimum time between action executions
        """
        self.cooldown = cooldown_seconds
        self.last_action_time = 0
        self.last_action = None
        
    def get_action(self, gesture_name):
        """
        Get the action string for a gesture.
        
        Args:
            gesture_name: Name of recognized gesture
            
        Returns:
            Action string or None if no action mapped
        """
        return GESTURE_ACTIONS.get(gesture_name)
    
    def execute(self, gesture_name, confidence):
        """
        Execute action for a recognized gesture.
        
        Args:
            gesture_name: Name of recognized gesture
            confidence: Confidence score (0-1)
            
        Returns:
            True if action was executed, False otherwise
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_action_time < self.cooldown:
            return False
        
        # Get action for gesture
        action = GESTURE_ACTIONS.get(gesture_name)
        
        if action:
            self._perform_action(action, gesture_name)
            self.last_action_time = current_time
            self.last_action = gesture_name
            return True
            
        return False
    
    def _perform_action(self, action, gesture_name):
        """
        Perform the specified action.
        
        Args:
            action: Action identifier
            gesture_name: Name of the gesture that triggered this action
        """
        print(f"\nACTION: {action} (triggered by '{gesture_name}')")
        
        # Navigation actions
        if action == "navigation_forward":
            print("  -> Moving forward / Next page")
            # Add your actual navigation logic here
            
        elif action == "navigation_back":
            print("  -> Moving backward / Previous page")
            # Add your actual navigation logic here
            
        elif action == "approval_positive":
            print("  -> Approval / Like / Thumbs up")
            # Add your actual logic here
            
        elif action == "confirmation_ok":
            print("  -> Confirmation / OK / Accept")
            # Add your actual logic here
            
        else:
            print(f"  -> Unknown action: {action}")
    
    def reset_cooldown(self):
        """Reset the action cooldown timer."""
        self.last_action_time = 0
