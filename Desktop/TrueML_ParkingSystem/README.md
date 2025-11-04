# 🅿️ Automated Parking Management System (APMS)

An end-to-end, deep learning-powered solution for modern parking management, integrating real-time computer vision, license plate recognition, and a user-friendly Streamlit dashboard. Developed for scalability and optimized for deployment in Indian urban environments.

---

## **Key Features**

* **Fully Automated Entry/Exit:** Seamless vehicle check-in and check-out using computer vision to minimize human intervention.
* **Deep Learning Core:** Utilizes **Ultralytics YOLOv8** for highly accurate license plate detection ($mAP@0.5 \approx 86.4\%$) and **EfficientNetV2B0** for robust car color classification ($89.95\%$ validation accuracy).
* **License Plate Recognition (LPR):** Employs **EasyOCR** for reliable text extraction from cropped license plates.
* **Real-Time Dashboard:** A responsive, operator-friendly interface built with **Streamlit** for live occupancy visualization, transactional transparency, and manual overrides.
* **Lightweight Persistence:** Uses a lightweight **JSON database** (`parking_db.json`) for efficient management of parked vehicle records.
* **Regional Adaptation:** Models are optimized using transfer learning on region-specific datasets (Indian Number Plate and Vehicle Color datasets) to handle local challenges like plate variety and lighting.

---

## 🛠️ **System Architecture & Workflow**

The APMS is a multi-module, interoperable platform structured into three primary components: Training, Main Application, and External Tools.

### **Core Workflow (Entry/Ingress)**

1.  **Image Acquisition:** Camera captures image and places it in `/car_images/`.
2.  **Parallel Model Inference:** `main.py` invokes the following simultaneously:
    * **Color Classification:** EfficientNetV2B0 (`my_color_model.h5`) predicts the vehicle color.
    * **Plate Detection:** YOLOv8 (`best_lpn_detector.pt`) detects and crops the License Plate Number (LPN) area.
3.  **Text Extraction (OCR):** **EasyOCR** reads the text from the cropped plate region.
4.  **Database Transaction:** LPN, color, entry time, and an assigned spot are logged to `parking_db.json`.
5.  **Feedback:** The Streamlit dashboard updates with the new vehicle entry and live occupancy.

### **Project Directory Hierarchy**

```bash
APMS/
├── color_training/                 # 1. Vehicle Color Classification Training
│   ├── dataset/
│   │   └── car color dataset/      # Raw images (VCOR Dataset)
│   ├── train_color_model.py        # Script for EfficientNetV2B0 training and export
│   └── my_color_model.h5           # Final Keras HDF5 color model weights
│
├── lpn_training/                   # 2. License Plate Detection & Training
│   ├── dataset/
│   │   └── indian_lpn_dataset.yaml # YOLOv8 config (label map, paths, splits)
│   ├── train_lpn_detector.py       # End-to-end YOLOv8 training script
│   └── best_lpn_detector.pt        # Final YOLOv8 weights for deployment
│
└── main_application/               # 3. Production Environment & Runtime Files
    ├── car_images/                 # Repository for entry/exit images (cached and runtime)
    ├── parking_db.json             # Main system database (current parked vehicles)
    ├── main.py                     # Production Streamlit app and system entrypoint
    ├── requirements.txt            # All Python dependencies, locked for reproducibility
    └── venv/                       # Dedicated virtual environment

