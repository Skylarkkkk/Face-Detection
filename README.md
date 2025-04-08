# Face Detection Project

A modular face detection system implemented in Python, featuring extensible components for recognition, training, and data processing.

## Features
- 📷 Image data input handling
- 🤖 Face recognition
- 🛠️ Utility functions for image processing
- 🖥️ GUI by PyQt5

## Installation
```bash
git clone https://github.com/Skylarkkkk/Face-Detection.git
cd Face-Detection
# Install dependencies
pip install opencv-python
pip install opencv-contrib-python
pip install PyQt5
```

## Usage
```bash
python main.py
```

## Project Structure
```
├── images/  # Images to detect
├── models/  # Saved models
├── Data/  # Input data
├── utils/
│   ├── data_input.py    # Data loading/processing
│   ├── recognizer.py    # Face detection
│   ├── tools.py         # Image processing utilities
│   └── trainer.py       # Model training module
└── main.py              # Entry point
```

## License
Distributed under the MIT License. See LICENSE[https://github.com/Skylarkkkk/Face-Detection/blob/main/LICENSE] for more information.
