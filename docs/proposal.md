## AgroVision Crop Mapper proposal: Interactive Crop Mapping Using Satellite Imagery and Deep Learning

## Project Description

Accurate and timely information about crop distribution is essential for agricultural monitoring, planning, and decision-making. However, most existing satellite-based crop mapping systems require advanced knowledge of remote sensing, GIS tools, and multi-band satellite data processing, which limits their accessibility to non-technical users.

This project proposes **AgroVision Crop Mapper**, an interactive, map-based application that performs **crop mapping using semantic segmentation with custom deep learning models**. Satellite imagery is classified at the pixel level, and the resulting crop maps and area statistics are presented through a user-friendly graphical interface. The system enables users to explore, analyze, and export crop mapping results without requiring specialized technical expertise.

The expected outcome is a fully functional GUI application that demonstrates an end-to-end deep learning workflow, from data preparation and model training to inference and deployment.

## Workflow

### 1. Data Collection and Preparation

- Use the **AgriFieldNet** dataset consisting of Sentinel-2 multi-band satellite image tiles with pixel-level crop labels.
- Organize and analyze the dataset, including crop classes and label distributions.
- Preprocess satellite imagery (normalization, resizing, band selection).
- Prepare training and validation splits.
- Document all data preparation steps to ensure reproducibility.

### 2. Model Design and Training

- Design and implement a **custom CNN-based semantic segmentation model** using PyTorch (e.g., U-Net or DeepLabV3+).
- Train the model on the prepared dataset and tune basic hyperparameters.
- Experiment with appropriate loss functions for multi-class segmentation.
- Save trained model weights and training logs.

### 3. Feature Engineering and Evaluation

- Evaluate model performance using suitable segmentation metrics such as mIoU and F1-score.
- Analyze per-class performance and identify common error cases.
- Validate model robustness using a held-out validation set.
- Document evaluation results and observations.

### 4. Visualization and Presentation

- Visualize training progress (loss curves and evaluation metrics).
- Generate example prediction visualizations (RGB image with crop mask overlay).
- Create tables summarizing crop-wise area statistics.
- Prepare figures and explanations for the final project presentation.

### 5. Deployment and Web Application

- Develop an interactive **map-based GUI** using a Python framework (e.g., Streamlit or Gradio).
- Allow users to pan and zoom to select regions of interest.
- Apply automatic constraints on region size to ensure efficient inference.
- Display prediction overlays, legends, transparency controls, and statistics.
- Enable exporting results as images and CSV reports.
- Ensure the application is runnable locally and well-documented.

### 6. Documentation and Reporting

- Write a comprehensive README explaining:
    - project structure
    - setup and execution steps
    - usage of the GUI
- Document the full pipeline from data handling to deployment.
- Clearly describe challenges faced and solutions applied.
- Ensure all code is clean, commented, and reproducible.

## Team Structure and Task Distribution (5 Students)

> Tasks are divided to ensure equal contribution across all stages of the project, as required by the course.

