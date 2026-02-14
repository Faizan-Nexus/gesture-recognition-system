# Gesture Recognition System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange.svg)](https://google.github.io/mediapipe/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-red.svg)](https://opencv.org/)

**Real-time hand gesture recognition system using MediaPipe and statistical pattern matching**

---

## Table of Contents

- [About](#about)
- [Features](#features)
- [Demo](#demo)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## About

The **Gesture Recognition System** is a real-time hand gesture recognition application that uses **MediaPipe** for hand landmark detection and a **statistical Euclidean distance-based matching algorithm** for gesture classification. The system is designed to be robust, accurate, and easy to use, with support for custom gesture registration and user-specific calibration.

### Key Highlights

-  **Real-time Performance** - 30 FPS recognition speed
-  **High Accuracy** - Statistical matching with >90% accuracy
-  **Easy Customization** - Register your own gestures in minutes
-  **Robust Detection** - Works with various hand angles and positions
-  **Clean UI** - Minimal, intuitive interface

---

##  Features

-  **Real-time Hand Detection** - MediaPipe-powered 21-point landmark detection
-  **Statistical Matching** - Euclidean distance-based template matching
-  **Custom Gesture Registration** - Easy 60-sample collection process
-  **User Calibration** - Personalized adaptation for improved accuracy
-  **Quality Assurance** - Real-time hand visibility and lighting checks
-  **Action Mapping** - Configurable gesture-to-action bindings
-  **Smoothing Algorithm** - Majority voting to reduce jitter
-  **Dataset Training** - Train from image datasets

---

##  Demo

> **Note**: Add screenshots or GIF demonstrations of your gesture recognition system in action here.

### Supported Gestures

| Gesture | Description | Use Case |
|---------|-------------|----------|
| **Go** | Hand pointing forward | Navigation forward |
| **Come** | Beckoning gesture | Navigation back |
| **Nice** | Thumbs up | Approval/Like |
| **Ok** | Thumb and index circle | Confirmation |

---

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam
- Windows, macOS, or Linux

### Installation

1. **Clone the repository**
   ```bash
   git clone <https://github.com/Faizan-Nexus/gesture-recognition-system/>
   cd gesture_system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Register gestures** (First time only)
   ```bash
   python gesture_registration.py
   ```
   - Enter gesture name
   - Press SPACE 60 times to capture samples
   - Repeat for each gesture

4. **Run the system**
   ```bash
   python main.py
   ```
   - Complete calibration (or press 'q' to skip)
   - Perform gestures to test recognition
   - Press 'q' to quit

---

## Installation

### Using pip

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
mediapipe>=0.10.0    # Hand landmark detection
opencv-python>=4.8.0  # Video processing
numpy>=1.24.0         # Numerical operations
scipy>=1.11.0         # Statistical functions
```

---

## Usage

### Registering New Gestures

```bash
python gesture_registration.py
```


## Project Structure

```
gesture_system/
├── main.py                      # Main recognition application
├── gesture_registration.py      # Gesture registration tool
├── train_from_images.py         # Train from image dataset
├── matcher.py                   # Gesture matching algorithm
├── mediapipe_module.py          # Hand detection wrapper
├── normalization.py             # Landmark normalization
├── calibration.py               # User calibration
├── actions.py                   # Action mapping
├── ui_helpers.py                # UI components
├── templates.py                 # Gesture templates (auto-generated)
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── hand_landmarker.task         # MediaPipe model file (7.8 MB)
├── .gitignore                   # Git ignore rules
├── README.md                    # This file
├── QUICK_START.md               # Quick start guide
├── SIGN_LANGUAGE_GESTURES.md    # Gesture reference
├── PROJECT_DOCUMENTATION.md     # Technical documentation
├── CONTRIBUTING.md              # Contribution guidelines
├── LICENSE                      # MIT License
└── Data/                        # Gesture image dataset
    ├── Go/
    ├── Come/
    ├── Nice/
    └── Ok/
```

---

##  How It Works

### System Architecture

```
Camera Input
    ↓
MediaPipe Hand Detection (21 landmarks)
    ↓
Normalization (wrist-centered, scaled)
    ↓
Calibration (user-specific offset)
    ↓
Gesture Matching (Euclidean distance)
    ↓
Smoothing (majority voting)
    ↓
Action Mapping
    ↓
Output Display / Action Execution
```

### Matching Algorithm

1. **Normalize** input landmarks (wrist-centered, scaled)
2. **Calculate** Euclidean distance to each template
3. **Select** closest match below threshold
4. **Convert** distance to confidence score
5. **Apply** majority voting smoothing
6. **Execute** mapped action

---

## Configuration

Edit `config.py` to customize:

```python
# Camera Settings
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Recognition Settings
CALIBRATION_FRAMES = 30
SMOOTHING_BUFFER_SIZE = 5
MIN_CONFIDENCE = 0.3
```

Edit `matcher.py` for matching parameters:

```python
distance_threshold = 0.60  # Maximum distance for valid match
```

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **[MediaPipe](https://google.github.io/mediapipe/)** - Google's hand landmark detection framework
- **[OpenCV](https://opencv.org/)** - Computer vision library
- **[NumPy](https://numpy.org/)** & **[SciPy](https://scipy.org/)** - Scientific computing libraries

---


**Made with ❤️ by the Muhammad Faizan Anjum Shah**

**Last Updated**: February 14, 2026 | **Version**: 1.0
