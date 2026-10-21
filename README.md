# 🩺 PneuSight AI: Clinical PACS Workstation & Multi-Pathology Triage System
"Submitted for KLE Tech Hackfest 2026"

## 🚀 Project Overview
PneuSight AI is a production-grade, automated chest radiograph triage and localization workstation designed to assist radiologists in high-level clinical and emergency environments. It ingests raw medical DICOM files directly, parses clinical metadata headers, applies automated safety guardrails, utilizes deep neural networks for 3-class pneumonia triage, runs YOLO for opacity bounding boxes, generates Grad-CAM explainability heatmaps, and features a true data-driven multi-pathology deep sub-classifier—all wrapped in a secure zero-disk-latency Streamlit dashboard.

---

## 👥 Team AeroSpark
* **Nagendra** (Team Lead & AI Architecture)
* **Aditya** (Computer Vision & UI Integration)
* **Vijay** (Backend Execution & Audit Logging)
* **Gurukiran** (Clinical Validation & Triage Logic)

---

## ⚙️ Technical Architecture & Stack
* **Primary Backbone:** PyTorch EfficientNet-B0 (3-class triage trained on RSNA dataset).
* **Multi-Pathology Sub-Classifier:** **TorchXRayVision DenseNet-121** (`densenet121-res224-all`) pre-trained on aggregated global multi-institutional clinical cohorts (NIH, CheXpert, MIMIC) for unbiased, data-driven disease prediction (Pleural Effusions, Cardiomegaly, Nodules, Infiltrates).
* **Localization & Detection:** Ultralytics YOLO for spatial pulmonary opacity bounding boxes.
* **Explainability:** Hook-based Grad-CAM heatmaps extracted directly from EfficientNet's final convolutional layer (`model.features[-1]`).
* **Medical Data Ingestion:** Secure `pydicom` binary buffer streaming (`io.BytesIO`) for true 16-bit DICOM parsing and window leveling in memory.
* **Frontend User Interface:** Streamlit PACS Dashboard with dynamic sensitivity thresholding and automated CSV audit logging (`pacs_audit_log.csv`).

---

## 📊 Clinical KPIs & Evaluation Metrics
* **Screening Sensitivity / Recall:** $\ge 95\%$ target triage accuracy for critical pneumonia cases to prevent false negatives.
* **Inference Latency:** Sub-second response time ($< 1.5$ seconds per radiograph).
* **Safety Guardrails:** Automated modality verification filters to instantly block non-medical image uploads (e.g., color photos or selfies).
* **Uncertainty Stratification:** Built-in confidence boundary checks to flag borderline predictions for mandatory radiologist review.


1. Clone the repository:
   ```bash
   git clone [https://github.com/HackFest-2026/Team-6A.5.git](https://github.com/HackFest-2026/Team-6A.5.git)
