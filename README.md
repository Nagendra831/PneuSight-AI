# 🩺 PneuSight AI: Clinical PACS Workstation & Triage System
*Submitted for KLE Tech Hackfest 2026*

## 🚀 Project Overview
**PneuSight AI** is a production-grade, automated chest radiograph triage and localization workstation designed to assist radiologists in high-level clinical and emergency environments. It ingests raw medical DICOM files directly, parses clinical metadata headers, applies automated safety guardrails, utilizes deep neural networks for 3-class pneumonia triage, runs YOLO for opacity bounding boxes, and generates Grad-CAM explainability heatmaps—all wrapped in a secure zero-disk-latency Streamlit dashboard.

---

## 👥 Team AeroSpark
* **Nagendra** (Team Lead & AI Architecture)
* **Aditya** (Computer Vision & UI Integration)
* **Vijay** (Backend Execution & Audit Logging)
* **Gurukiran** (Clinical Validation & Triage Logic)

---

## ⚙️ Technical Architecture & Stack
* **Deep Learning Backbone:** PyTorch EfficientNet-B0 (optimized for standard 224x224 tensor normalization).
* **Localization & Detection:** Ultralytics YOLO for spatial pulmonary opacity bounding boxes.
* **Explainability:** Hook-based Grad-CAM heatmaps extracted directly from EfficientNet's final convolutional layer (`model.features[-1]`).
* **Medical Data Ingestion:** Secure `pydicom` binary buffer streaming (`io.BytesIO`) for true 16-bit DICOM parsing and window leveling.
* **Frontend User Interface:** Streamlit PACS Dashboard with dynamic sensitivity thresholding and CSV audit logging.

---

## 📊 Clinical KPIs & Evaluation Metrics
* **Screening Sensitivity / Recall:** $\ge 95\%$ target triage accuracy for critical pneumonia cases to prevent false negatives.
* **Inference Latency:** Sub-second response time ($< 1.5$ seconds per radiograph).
* **Safety Guardrails:** Automated modality verification filters to instantly block non-medical image uploads (e.g., color photos or selfies).
* **Differential Sub-Classifier:** Secondary probability distribution mapping across Tuberculosis (TB), COVID-19 consolidation, and focal malignancy patterns.

---

## 💻 How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/Nagendra831/PneuSight-AI.git](https://github.com/Nagendra831/PneuSight-AI.git)
