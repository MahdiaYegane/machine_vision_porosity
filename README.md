# machine_vision_porosity
# ResNet50 U-Net Inference Script

This script performs semantic segmentation inference on rock thin-section images to calculate porosity. It uses an ensemble of 7-fold cross-validation trained ResNet50-U-Net models.

## Features

- **Ensemble Prediction**: Averages predictions from 7 trained models for robust results
- **CLAHE Preprocessing**: Applies Contrast Limited Adaptive Histogram Equalization
- **Patch-based Inference**: Processes large images using sliding windows (128×128 patches)
- **Porosity Calculation**: Automatically computes porosity percentage from predicted masks
- **Visualization**: Generates comparison plots showing:
  - Input image
  - Predicted binary mask
  - Difference visualization (overlap, only GT, only prediction)

## Requirements

```bash
pip install tensorflow scikit-image opencv-python matplotlib numpy
