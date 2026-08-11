# Federated Learning-Based Privacy-Preserving Medical Image Diagnosis using Explainable AI

Simulates multiple hospitals ("clients") collaboratively training a shared
CNN diagnostic model **without ever sharing raw patient images** (Federated
Learning), while making every prediction interpretable via **Grad-CAM**
heatmaps (Explainable AI).

## Project Structure

```
fl-medical-xai/
├── src/
│   ├── dataset.py            # data loading + non-IID client split
│   ├── model.py               # CNN architecture (ResNet18-based)
│   ├── client.py               # Flower FL client (one simulated hospital)
│   ├── server.py               # Flower FL server (FedAvg aggregation)
│   ├── gradcam.py              # Grad-CAM explainability
│   ├── utils.py                 # training/eval loops, metrics
│   └── train_centralized.py    # non-FL baseline, for comparison
├── backend/
│   └── app.py                   # FastAPI server (serves model to frontend)
├── frontend/
│   └── src/App.jsx              # React diagnostic console UI
├── notebooks/
│   └── colab_training.ipynb     # run this on Google Colab (free GPU)
├── data/                         # dataset goes here (see below)
├── saved_models/                 # trained weights land here
├── results/                      # metrics, plots, training history.json
└── requirements.txt
```

## 1. Dataset Setup

Download any binary/multi-class medical image classification dataset, e.g.:
- [Chest X-Ray Pneumonia (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

Arrange it like:
```
data/
  train/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

## 2. Training on Google Colab (recommended — free GPU)

Open `notebooks/colab_training.ipynb` in Colab, set runtime to **T4 GPU**,
and run the cells in order. It will:
1. Clone this repo
2. Mount your dataset from Google Drive
3. Train the centralized baseline (`train_centralized.py`)
4. Run the federated simulation (`server.py`) across N simulated hospitals
5. Save `saved_models/global_model_final.pth` and `results/history.json`

## 3. Local Development

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run federated simulation locally (small dataset / CPU, just for testing code):
```bash
python src/server.py --mode simulation --num_clients 5 --num_rounds 5
```

Run centralized baseline for comparison:
```bash
python src/train_centralized.py --epochs 5
```

## 4. Run the Demo App

**Backend (FastAPI):**
```bash
cd backend
pip install -r ../requirements.txt
uvicorn app:app --reload --port 8000
```

**Frontend (React):**
```bash
cd frontend
npm install
npm install recharts lucide-react
npm run dev
```
Open `http://localhost:5173`, upload a scan, and view the prediction + Grad-CAM heatmap.

## 5. Key Metrics to Report

- Federated vs. Centralized accuracy/F1/AUC (privacy-utility trade-off)
- Accuracy convergence across communication rounds
- Grad-CAM qualitative examples (correct vs. incorrect predictions)
- (Optional) Accuracy with/without Differential Privacy (Opacus) — shows
  privacy budget (epsilon) vs accuracy trade-off

## Notes

- All clients share the exact same model architecture (`model.py`) — this is
  required for FedAvg weight averaging to work.
- Non-IID data split (`dataset.py::split_non_iid`) simulates realistic
  hospitals where disease prevalence differs by site.
- `gradcam.py` hooks into the last convolutional block of the ResNet18
  backbone to generate heatmaps for any prediction.
