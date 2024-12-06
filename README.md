# SEDAR CAPTCHA Solver Project

## Project Overview

This project demonstrates a machine learning approach to solving CAPTCHA challenges specifically designed for the SEDAR (System for Electronic Document Analysis and Retrieval) website. The solution utilizes a Convolutional Neural Network (CNN) implemented in PyTorch to automatically recognize and decode CAPTCHA images with high accuracy.

**Important Note**: This project is now shared for educational and research purposes only. The code was developed when SEDAR's website had a specific CAPTCHA implementation. The current website has since been updated, rendering this specific solution non-functional for unauthorized scraping.

## Project Purpose

The primary objective of this project was to develop an automated solution for bypassing CAPTCHA challenges on the SEDAR website. The solution involves:
1. Taking screenshots of web pages
2. Cropping the CAPTCHA section
3. Using a trained CNN model to decode the CAPTCHA text
4. Preparing the decoded text for potential automated form filling via Selenium

## Technical Highlights

### Model Performance
- **Accuracy**: 98.87% CAPTCHA recognition
- **Architecture**: Custom CNN with multiple convolutional layers and softmax outputs
- **Framework**: PyTorch

### Model Architecture Details
The neural network (`CAptcha_CNN`) features:
- 3 Convolutional layers with ReLU activations
- MaxPooling for feature reduction
- Dropout layers to prevent overfitting
- Multi-output softmax classification for character recognition

## Repository Structure
```
project-root/
│
├── CAPTCHA.py           # Main inference script
├── captcha_cnn.ipynb    # Data preparation, training, and modeling notebook
├── requirements.txt     # Project dependencies
├── model/               # Saved model weights
│   └── model_smooth.pth
├── ss2/                 # Screenshot storage
├── ss_folders/          # Additional screenshot folders
├── Data_for_captcha.txt     # Training data
└── Data_for_captcha_val.txt # Validation data
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- Selenium
- Other dependencies listed in `requirements.txt`

### Setup
1. Clone the repository
```bash
git clone https://your-repository-url.git
cd sedar-captcha-solver
```

2. Install required dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Inference
```python
from CAPTCHA import Captcha_inf

# Initialize the CAPTCHA inference
INF = Captcha_inf()

# Solve a CAPTCHA image
result = INF.Captcha_sol("path/to/captcha/image.png")
print(result)
```

## Ethical Considerations
This project is shared transparently with the understanding that:
- The code is no longer functional for the current SEDAR website
- It was developed for research and educational purposes
- Unauthorized scraping or automated access to websites is unethical and potentially illegal

## Contributing
Contributions, discussions, and constructive feedback are welcome. Please open an issue or submit a pull request.

## License
Distributed under the MIT License. See LICENSE for more information.

## Disclaimer
This project is provided for educational purposes only. Always respect website terms of service and legal guidelines.

## Example Screenshot
![SEDAR CAPTCHA Screenshot](/DevTry.png)
