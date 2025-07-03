MetaCogAI: A Confidence-Aware Multimodal AI System

MetaCogAI is a powerful AI system that classifies images and text, estimates the confidence behind its predictions, and intelligently fuses results into a meaningful decision. Designed as a research-driven prototype, it bridges computer vision, natural language processing, and human-centric AI introspection.

---

## 📌 Project Highlights

- ✅ **Multimodal Inference**: Supports both **image** (CIFAR-10) and **text** (IMDb reviews) inputs
- ✅ **Confidence Estimation**: Every output includes a confidence score
- ✅ **Fusion Logic**: A logic-based combination of image + text predictions for higher-level decisions
- ✅ **Logging System**: Tracks predictions and metadata for each session
- ✅ **Interactive UI**: A deep vintage-styled HTML interface with background animation
- ✅ **Self-contained**: Fully local deployment, no external cloud dependencies

---

## 🗂 Project Structure
MetaCogAI/
├── app/
│   ├── main.py               # FastAPI app entry
│   ├── api.py                # API endpoints for image/text/fusion
│   ├── models/
│   │   ├── multimodal_model.py  # Logic for prediction and fusion
│   │   └── fusion.py             # Fusion rules
│   └── utils/
│       └── logger.py         # Prediction logger
├── models/
│   ├── image_model.pth       # Trained ResNet-18 model
│   └── text_model/           # Fine-tuned DistilBERT
├── data/                     # Auto-downloaded CIFAR-10, IMDb datasets
├── index.html                # Frontend user interface
├── train_image_model.py      # Script to retrain image model
├── train_text_model.py       # Script to retrain text classifier
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
└── README.md                 # You're here.

 
 Getting Started

 1. Clone the Repository
clone the repository to your local device.

 2. Set Up a Virtual Environment
python -m venv venv
source venv/bin/activate        # On macOS/Linux
venv\Scripts\activate           # On Windows

3. Install Required Libraries
pip install -r requirements.txt

 4. Run the FastAPI Backend
uvicorn app.main:app --reload
Visit the FastAPI interactive docs at http://127.0.0.1:8000

 5. Launch the Frontend UI
Open index.html in your browser directly. No server needed — it's static.

 Testing the System
The UI supports uploading a CIFAR-10-like image and a movie review. You’ll get:

Image prediction from CIFAR-10 (cat, dog, airplane, etc.)

Text sentiment from IMDb (positive, negative)

Final Fusion Decision based on both

🖼 CIFAR-10 Image Classes Used
Label Index	Class Name
0	airplane
1	automobile
2	bird
3	cat
4	deer
5	dog
6	frog
7	horse
8	ship
9	truck

 Tools & Libraries Used
Python 3.10+

PyTorch (image training & inference)

Transformers (HuggingFace) (DistilBERT for text)

FastAPI (backend API)

HTML + JS + CSS (vintage-themed frontend)

Uvicorn (ASGI server)

Datasets from:

CIFAR-10 (image classification)

IMDb (sentiment analysis)

 UI Preview
A modern but vintage-styled interface with interactive background and elegant typefaces.

Upload an image + review → Hit "Predict" → Get predictions and confidence → Fusion Result

 Logging
All predictions (image, text, fusion) are automatically logged via utils/logger.py and can be used for:

Auditing

Meta-analysis

Evaluation of model bias or overconfidence

 Retraining the Models
If needed:
python train_image_model.py   # For CIFAR-10 ResNet
python train_text_model.py    # For IMDb DistilBERT
Outputs are saved to models/.

 License
This project is released under the MIT License.

 Author
Sai Niketh Kosaraju
Computer Science Engineering
Shiv Nadar University
Email: sainiketh.kosaraju@gmail.com

Acknowledgements
CIFAR-10 Dataset by Alex Krizhevsky

IMDb Dataset via HuggingFace Datasets

FastAPI for backend simplicity

HuggingFace Transformers for NLP power

PyTorch for image modeling ease

 Ideal For
Personal AI portfolio

Teaching ML model fusion

Research foundations for confidence-aware systems
