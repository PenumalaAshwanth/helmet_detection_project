# 🪖 Bike Helmet Detection using YOLOv8

## **Week 1 Report – Dataset Preparation & Project Setup**

### **1. Project Overview**
This project aims to develop an AI-based system that detects whether bike riders are wearing helmets using deep learning object detection.  
The primary objective is to enhance **road safety** and assist in **automated traffic law enforcement** by identifying *helmet* and *no-helmet* cases from static images and live video feeds.  
The project is part of the **Automotive Safety and AI-based Object Detection** domain.

---

### **2. Problem Statement**
Every year, thousands of road fatalities occur due to riders not wearing helmets.  
Monitoring helmet compliance manually is time-consuming and inefficient.  
Hence, an automated computer vision system that can **detect helmet violations in real-time** can greatly assist authorities and promote safer driving practices.

---

### **3. Objectives**
- To build a YOLOv8-based deep learning model capable of detecting helmets and non-helmets on riders.  
- To collect, annotate, and structure a high-quality dataset for training.  
- To train the model in **Google Colab** using GPU acceleration.  
- To evaluate model performance based on accuracy, precision, recall, and mAP metrics.  
- To deploy the trained model locally for inference on images and video streams.

---

### **4. Tools and Technologies**
| Category | Tools/Frameworks |
|-----------|------------------|
| Programming Language | Python |
| Object Detection Framework | Ultralytics YOLOv8 |
| Deep Learning Backend | PyTorch |
| Supporting Libraries | OpenCV, NumPy, Matplotlib, Pandas |
| Annotation Tool | LabelImg / Roboflow |
| Environment | Google Colab |
| Version Control | Git & GitHub |

---

### **5. Dataset Preparation**
- **Data Collection:**  
  - Gathered motorbike rider images showing both *helmet* and *no-helmet* cases from Roboflow and other open sources.  
  - Ensured dataset balance and diversity (different angles, lighting, environments).  
- **Data Structure:**  
  The dataset follows the YOLO directory format:
  ```
  data/
  ├── train/
  │   ├── images/
  │   └── labels/
  ├── valid/
  │   ├── images/
  │   └── labels/
  └── test/
      ├── images/
      └── labels/
  ```
- **Annotation:**  
  - Used **LabelImg** / **Roboflow Annotator** to draw bounding boxes around helmets and non-helmet heads.  
  - Saved annotations in YOLO format (`.txt` with class, x_center, y_center, width, height).  
- **Classes Defined:**
  - `0 → helmet`
  - `1 → no-helmet`

---

### **6. Data Configuration File**
Created `helmet_data.yaml` for YOLOv8 training:
```yaml
train: data/train/images
val: data/valid/images
test: data/test/images
names:
  0: helmet
  1: no-helmet
nc: 2
```

---

### **7. Model Selection and Rationale**
The model chosen for this task is **YOLOv8n (nano)** from Ultralytics.

**Reasons for choosing YOLOv8:**
- State-of-the-art accuracy and speed in real-time detection.  
- Simplified training process in Google Colab using GPU.  
- Built-in support for data augmentation and hyperparameter optimization.  
- Lightweight architecture suitable for local deployment.

**Planned Process:**
1. Start with `yolov8n.pt` pretrained weights for transfer learning.  
2. Train on custom dataset for 100–180 epochs.  
3. Evaluate model on validation and test sets.  
4. Save and export trained weights (`best.pt`) for inference.

---


---

### **9. Expected Workflow**
1. **Data Collection & Annotation** – Week 1  
2. **Model Training & Evaluation** – Week 2  
3. **Adding FRONT END** – Week 3  
4. **PRESENTATION** – Week 4  

---

### **10. Challenges Identified**
- Maintaining class balance between helmet and no-helmet images.  
- Ensuring detection accuracy for small or partially visible helmets.  
- Preventing overfitting during extended training epochs.  
- Managing GPU runtime limits in Google Colab.

---

### **11. Planned Deliverables for Week 2**
- Annotated dataset (≈ 1,000+ training images).  
- YOLOv8 model trained for 180 epochs on Colab.  
- Validation metrics (mAP, Precision, Recall).  
- Exported model (`best.pt`) for inference and local testing.

---

### **12. Dataset Source**
Dataset partially sourced from **Roboflow Universe – Helmet Detection Dataset**, supplemented with additional images collected manually to improve variety and robustness.

---

### **13. Conclusion**
The Week 1 setup established the foundation for the Bike Helmet Detection project by organizing the dataset, defining the class schema, and preparing the YOLOv8 environment in Google Colab.  
In **Week 2**, training and evaluation will be performed to obtain a working model capable of accurate helmet and no-helmet detection.

---
