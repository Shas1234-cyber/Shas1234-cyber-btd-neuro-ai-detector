🧠 NeuroAI Detector - Multi-Class Brain Tumor Detection & Analysis System
AI-Powered Brain Tumor Classification using Deep Transfer Learning and Magnetic Resonance Imaging (MRI)

NeuroAI Detector is an end-to-end medical decision-support web application that classifies brain MRI scans into four distinct clinical categories with 92%+ validation accuracy. Powered by a fine-tuned VGG16 convolutional neural network (~21.14M parameters) and built with a Flask backend, this system delivers diagnostic confidence scores and class probability breakdowns in under 3 seconds.

✨ Features
🚀 Core Capabilities
4-Class Multi-Tumor Classification:

Glioma

Meningioma

Pituitary Tumor

No Tumor (Healthy Brain Tissue)

Transfer Learning Backbone: Custom top classifier integrated with fine-tuned VGG16 convolutional blocks (Blocks 4 & 5 unfrozen).

Clinical Confidence Analysis: Provides individual softmax percentage probabilities across all four potential diagnoses.

Instant Inference: Full image preprocessing, array transformations, and inference cycle execute in under 3 seconds.

Patient Session Logging: Tracks patient identifier/name alongside system timestamps and formatted predictions.

🎨 UI & UX Design
Bootstrap 5 Interface: Clean, responsive layout tailored for healthcare dashboards.

Dynamic Probability Indicators: Color-coded diagnostic cards highlighting selected-class confidence.

Scan Visualization: Side-by-side verification of uploaded scans alongside technical model diagnostics.

Cross-Device Support: Optimized for desktop, tablet, and mobile browsers.

🗂️ Dataset Details
The underlying model is trained and benchmarked on the standardized Brain Tumor Classification (MRI) dataset hosted on Kaggle, integrating verified clinical scans from the SARTAJ and Figshare medical archives.

Primary Dataset: Brain Tumor Classification (MRI) on Kaggle

Direct Kaggle CLI:

Bash
kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri
Dataset Volume: ~3,264+ MRI scans (Training: 2,870 | Testing: 394)

Extended Benchmark: Brain Tumor MRI Dataset (7,023 images)

Imaging Modalities: T1-weighted contrast-enhanced, T2-weighted, and FLAIR scans in Axial, Coronal, and Sagittal planes.

Input Spatial Resolution: Standardized to 224 x 224 pixels with 3-channel RGB depth.

Clinical Categories
Class Label	Pathological Category	Clinical Significance
glioma_tumor	Intra-axial Malignancy	Arises from glial cells; requires immediate grading and surgical assessment.
meningioma_tumor	Extra-axial Neoplasm	Develops from the meninges layers surrounding the brain and spinal cord.
pituitary_tumor	Sellar Region Mass	Endocrine adenomas impacting hormone regulation and optic chiasm.
no_tumor	Negative Control	Normal MRI brain scans with no pathological mass or lesion detected.
🔧 Technical Specifications
Component	Technical Implementation
Backbone Architecture	VGG16 (ImageNet weights, Blocks 4 & 5 fine-tuned)
Classification Head	Flatten → Dense(256, ReLU) → Dropout(0.4) → Dense(4, Softmax)
Total Parameters	21,140,548 (~21.14 Million parameters)
Image Preprocessing	VGG16 Zero-centering (ImageNet Mean Subtraction, BGR array order)
Backend Framework	Flask 3.x / Werkzeug WSGI
Deep Learning Stack	TensorFlow 2.10.0 / Keras 2.10.0
Data Pipelines	NumPy, Pillow (PIL), H5py
Frontend Stack	HTML5, CSS3, JavaScript, Bootstrap 5
📁 Repository Structure
Plaintext
btd-neuro-ai-detector/
├── static/
│   ├── css/
│   │   └── style.css
│   ├── models/
│   │   └── brain_tumor_vgg16_90acc.h5    # Trained model weights
│   └── user_images/                      # Processed scan storage
├── templates/
│   ├── index.html                        # Portal & upload view
│   └── prediction1.html                  # Diagnostic report view
├── .gitignore
├── app.py                                # Main Flask server & inference engine
├── requirements.txt                      # Project dependencies
└── README.md
💻 Installation & Local Setup
1. Clone the Repository
Bash
git clone https://github.com/shashank-gupta/btd-neuro-ai-detector.git
cd btd-neuro-ai-detector
2. Configure Virtual Environment
Bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
4. Verify Model Location
Ensure your trained weights file brain_tumor_vgg16_90acc.h5 is placed inside the static/models/ directory:

Plaintext
static/models/brain_tumor_vgg16_90acc.h5
🚀 Running the Application
Start the Flask development server:

Bash
python app.py
Open your browser and navigate to:

Plaintext
http://127.0.0.1:5000
(To access from another device on the same local network, use http://<YOUR_LOCAL_IP>:5000)

🖱️ Step-by-Step Usage
Access Portal: Open the landing page and navigate to the Analysis section.

Enter Patient Information: Provide the patient's full name or hospital case ID.

Select Scan: Upload a supported brain MRI image (.jpg, .jpeg, .png).

Initiate Scan Analysis: Click Analyze Image.

Review Diagnostic Dashboard:

Overall diagnostic status: TUMOR DETECTED or NO TUMOR DETECTED.

Predicted pathological subtype (Glioma, Meningioma, Pituitary, or None).

Exact confidence percentage for the selected class.

Class distribution breakdown across all 4 categories.

⚕️ Medical Disclaimer
IMPORTANT CLINICAL NOTICE:

This application is built as an academic research project and a technical proof-of-concept. It is not a certified software medical device (SaMD) and must not be used as a definitive diagnostic instrument. Final diagnostic determinations must always be confirmed by licensed radiologists, oncologists, and neurosurgeons using standard histological and clinical protocols.

👨‍💻 Author
Shashank Gupta

Department of Artificial Intelligence & Machine Learning

Noida Institute of Engineering and Technology (NIET)

GitHub: github.com/shashank-gupta

LinkedIn: linkedin.com/in/shashank-gupta

📄 License
This project is licensed under the MIT License — see the LICENSE file for full terms.