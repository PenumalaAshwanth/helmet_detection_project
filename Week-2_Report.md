# 🪖 Bike Helmet Detection using YOLOv8

### 📅 Week 2 Report — Model Training, Evaluation & Fine-Tuning

---

## 🧠 1. Week 2 Overview
In Week 2, the focus shifted from dataset preparation to **model training, fine-tuning, and evaluation** using YOLOv8 on Google Colab.  
The goal was to train a robust object detection model capable of identifying **helmet** and **no-helmet** riders accurately from custom datasets.

---

## ⚙️ 2. Files & Folder Structure
At the end of Week 2, the project directory contained the following structure:

```
helmet-detection-project/
├── helmet_detection_project.ipynb       ← Google Colab notebook used for training & testing
├── helmet_data.yaml                     ← Dataset configuration file
├── yolov8n.pt                           ← Pretrained base model used for initial training
├── yolov11n.pt                          ← Newer YOLOv11 lightweight model for future experiments
├── Datasets/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── runs/
│   ├── helmet_from_scratch/             ← Folder from initial 180-epoch training
│   │   ├── weights/
│   │   │   ├── best.pt   ← ✅ Trained model with best validation performance
│   │   │   └── last.pt   ← Final model after all epochs
│   │   ├── results.csv
│   │   └── confusion_matrix.png
│   ├── helmet_ft_1024/                  ← Fine-tuned run (higher image size 1024)
│   ├── detect/                          ← Output folder for predictions
│   │   └── predict/
│   │       ├── result_images...
│   └── logs/
│       └── TensorBoard logs (for visualization)
└── utils_matching.py
```

**Key Model File:**
- `runs/helmet_from_scratch/weights/best.pt` → The final **trained model weights** used for all detections.
- `last.pt` → Saved automatically after the final epoch; contains final state, not necessarily best accuracy.

---

## 🚀 3. Training Process
Training was performed on **Google Colab GPU (Tesla T4)** using the YOLOv8 framework.

### Training Configuration
| Parameter | Value |
|------------|--------|
| Model | `yolov8n.pt` |
| Epochs | 180 |
| Image size | 896 × 896 |
| Batch size | 8 |
| Device | CUDA (GPU) |
| Optimizer | AdamW |
| Patience (Early Stop) | 80 epochs |
| Dataset | Custom Helmet / No-Helmet (2 classes) |

Training was monitored via **TensorBoard** and YOLO’s built-in visualizations under `/content/runs/`.

---

## 📈 4. Training Output Summary
### Final Model Metrics (Validation Results)
| Class | Precision (P) | Recall (R) | mAP@50 | mAP@50–95 |
|--------|----------------|-------------|----------|-------------|
| **helmet** | 0.76 | 0.88 | **0.86** | 0.53 |
| **no-helmet** | 0.58 | 0.60 | **0.56** | 0.31 |
| **Overall** | 0.67 | 0.74 | **0.71** | 0.42 |

**Speed per image:**  
`~0.83 s (CPU)` or `~0.03 s (GPU)` for inference.  

Training ran for **167 epochs** before early stopping (best performance at epoch 87).  
All logs and results were saved inside `runs/helmet_from_scratch/`.

---

## 🧪 5. Fine-Tuning (Second Experiment)
After baseline training, fine-tuning was performed:
- Used the previous `best.pt` as the starting model.  
- Increased input resolution to **1024×1024** for better small-object detection.  
- Ran for **50 epochs**.  
- Results: overall performance similar, confirming the first model was optimal.

---

## 🧠 6. Model Evaluation & Testing
The final trained model (`best.pt`) was tested on multiple unseen test images and videos.

### Example Detection Results
- **Image:** `Datasets/test/images/BikesHelmets14_png.rf.696772056ca1da3c52a31c0acf6c0140.jpg`
- **Prediction Output:**  
  - 2 helmets detected  
  - Saved annotated image in:  
    `runs/detect/predict/BikesHelmets14_png.rf.6967....jpg`

### Testing Script Example
```python
from ultralytics import YOLO
model = YOLO("runs/helmet_from_scratch/weights/best.pt")
results = model.predict(source="Datasets/test/images", conf=0.25, save=True)
```

Results are automatically saved to:
```
runs/detect/predict/
```

---

## 🧩 7. Model Insights
- **YOLOv8n** performed best considering dataset size and GPU limits.  
- Training beyond 180 epochs didn’t improve accuracy due to data saturation.  
- Increasing image resolution improved helmet detection but not much for no-helmet cases.  
- Further improvements require **data augmentation** or **synthetic data** for no-helmet riders.

---

## ⚒️ 8. Tools Used
| Tool | Purpose |
|------|----------|
| **Google Colab** | GPU training environment |
| **Ultralytics YOLOv8** | Model training and evaluation |
| **PyTorch** | Deep learning backend |
| **OpenCV** | Image pre-/post-processing |
| **Matplotlib** | Result visualization |
| **TensorBoard** | Metric tracking |
| **GitHub** | Code and version control |

---

## 📦 9. Model Export & Local Inference
The final model (`best.pt`) was downloaded from Colab and integrated into the local project for inference.

**Local setup example:**
```python
from ultralytics import YOLO
model = YOLO("best.pt")  # or "runs/helmet_from_scratch/weights/best.pt"
results = model.predict(source="data/test/images", conf=0.25, save=True)
```

Output images are saved automatically in:
```
helmet-detection-project/runs/detect/predict/
```

---

## 🧭 10. Next Steps
- Perform **targeted data augmentation** (rotation, brightness, motion blur) for no-helmet samples.  
- Experiment with **YOLOv11n** for potentially better generalization.  
- Integrate a **Streamlit app** for real-time webcam detection.  
- Document training pipeline and prepare deployment notebook.

---

## 🏁 11. Summary
| Aspect | Week 1 | Week 2 |
|---------|--------|--------|
| Focus | Dataset & Setup | Training & Evaluation |
| Model | — | YOLOv8n |
| Epochs | — | 180 |
| Best Weights | — | ✅ `runs/helmet_from_scratch/weights/best.pt` |
| Accuracy (mAP50) | — | 0.71 overall |
| Output Folder | — | `runs/detect/predict/` |

---

**📌 Author:** Ashwanth  
**📅 Duration:** Week 2 (Model Training Phase)  
**📍 Environment:** Google Colab + Local Testing  
**🧠 Framework:** Ultralytics YOLOv8  
**💾 Trained Model:** `runs/helmet_from_scratch/weights/best.pt`
