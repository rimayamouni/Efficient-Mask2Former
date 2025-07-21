# Efficient-Mask2Former
This repository presents our work on enhancing the Mask2Former semantic segmentation model by integrating advanced model compression techniques and environmental impact tracking. We combine global pruning, partial pruning, and baseline Mask2Former to analyze performance, efficiency, and sustainability.

##  Project Overview

- **Base Model**: [Mask2Former](https://github.com/facebookresearch/Mask2Former)
- **Techniques Applied**:
  - **Baseline**: Original Mask2Former without pruning.
  - **Global Pruning**: Global unstructured pruning applied across convolutional layers.
  - **Partial Pruning**: Layer-specific selective pruning for targeted compression.
- **Tracking Framework**:
  - [eco2AI](https://github.com/eco2ai/eco2ai) for real-time tracking of:
    - Energy consumption (kWh)
    - CO₂ emissions (kg)
    - Training duration

- **Dataset Used**:
  - [Cityscapes Dataset](https://www.cityscapes-dataset.com/)

---

##  Pretrained Checkpoints

| Model Variant        | Download Link |
|----------------------|---------------|
| **Baseline Mask2Former** | [Google Drive](https://drive.goog) |
| **Global Pruning**       | [Google Drive](https://drive.goog) |
| **Partial Pruning**      | [Google Drive](https://drive.g) |



##  Installation & Setup

###  Install Dependencies

```bash
# Clone Mask2Former official repo
git clone https://github.com/facebookresearch/Mask2Former.git
cd Mask2Former

# Install requirements
pip install -r requirements.txt

# Install Detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install eco2AI
pip install eco2ai
