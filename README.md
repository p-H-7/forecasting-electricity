# Electricity Load Forecasting

Multi-horizon electricity load forecasting using deep learning models trained on the [UCI ElectricityLoadDiagrams 2011-2014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) dataset. The dataset contains 15-minute interval electricity consumption readings from 370 clients in Portugal, resampled to hourly resolution for modeling.

This project was developed as part of the course IEOR 4578: Forecasting – A Real-World Application at Columbia University under the instruction of Prof. Syed Haider during the Spring 2026 semester.

## What This Project Does

The goal is to forecast electricity consumption at 1-hour, 6-hour, and 24-hour horizons using four different deep learning architectures:

- **LSTM** — recurrent model with 2-layer stacked LSTM cells
- **TCN (Temporal Convolutional Network)** — causal dilated convolutions with residual connections
- **Transformer** — encoder-only transformer with learned positional embeddings
- **PatchTST** — patch-based transformer designed specifically for time series (patches of 16 with stride 8)

Clients are grouped into 3 clusters using KMeans on their mean/std consumption profiles. A separate model is trained per cluster per horizon, resulting in 36 trained models total (4 architectures x 3 clusters x 3 horizons). All models use a 168-hour (1 week) lookback window and 8 engineered time features (cyclical hour/day/month encodings, weekend flag, business hours flag).

The project also includes a Streamlit-based chat interface that lets a user ask questions in natural language about electricity forecasts. A user enters a client ID, the system maps it to the right cluster, loads the appropriate model, runs inference, and returns the forecast in plain English.

## Project Structure

```
├── eda_preprocessing.py              # Full EDA and preprocessing pipeline (Deliverable 1)
├── electricity_forecasting_code.py   # Initial model training code (Deliverable 1)
├── deliverable_2.py                  # Per-cluster model training with all 4 architectures
├── agentic_forecast.py               # CLI-based agentic forecasting interface
├── app.py                            # Streamlit chat app for interactive forecasting
├── d2_models/                        # Trained model weights (.pt) and metadata
│   ├── metadata.json                 # Client-to-cluster mapping and cluster info
│   ├── scalers.pkl                   # Fitted StandardScaler objects per cluster
│   ├── lstm_cluster0_h1.pt          # Model weights: {model}_{cluster}_{horizon}.pt
│   └── ...
├── d2_outputs/                       # Forecast plots and error boxplots
├── electricity_technical_report.docx
└── electricity_technical_report_deliverable2.docx
```

## Setup

### Requirements

Python 3.9+ with the following packages:

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn pyarrow streamlit openai
```

For the EDA pipeline specifically:
```bash
pip install statsmodels ucimlrepo
```

### Dataset

The dataset is not included in this repo due to its size (~678 MB). To get it:

1. Download `LD2011_2014.txt` from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
2. Place it in the project root directory

On the first run, the code will parse the CSV and cache it as `electricity_cache.parquet` for faster subsequent loads. If the dataset is unavailable, the scripts fall back to synthetic data generation so you can still run everything.

## How to Use

### Running the EDA Pipeline

This produces a full exploratory analysis — temporal patterns, autocorrelation, stationarity tests, clustering, and data splits:

```bash
python eda_preprocessing.py
```

Outputs are saved to `eda_outputs/`.

### Training Models

To train all models from scratch (LSTM, TCN, Transformer, PatchTST across 3 clusters and 3 horizons):

```bash
PYTHONUNBUFFERED=1 python deliverable_2.py
```

This will take a while depending on your hardware. Trained weights are saved to `d2_models/` and evaluation plots to `d2_outputs/`. The repo already includes pre-trained weights so you can skip this step if you just want to run inference.

### Running the Streamlit App

The interactive chat interface requires an OpenAI API key (used for the natural language layer):

```bash
export OPENAI_API_KEY="sk-..."
python3 -m streamlit run app.py
```

Open the URL that Streamlit prints (usually `http://localhost:8501`). You can type things like:
- "What's the forecast for client MT_044?"
- "Show me the 24-hour ahead prediction for client 100"
- "Which cluster does MT_015 belong to?"

### Running the CLI Forecast Agent

If you prefer a terminal-based interface:

```bash
export OPENAI_API_KEY="sk-..."
python agentic_forecast.py
```

## Model Details

All models take an input tensor of shape `(batch, 168, 9)` — 168 hours of lookback with 1 load value + 8 time features per timestep. The output is a vector of length equal to the forecast horizon (1, 6, or 24).

Training uses MSE loss with Adam optimizer (lr=1e-3), early stopping with patience of 3 epochs, and a batch size of 256. Each client's data is independently scaled using StandardScaler before being pooled within its cluster for training.

Evaluation is done using MAPE (Mean Absolute Percentage Error) computed on the held-out test set (last portion of the time series after an 80/10/10 train/val/test split).

## Results

Forecast comparison plots and per-model MAPE boxplots for each cluster and horizon are saved in `d2_outputs/`. The `mape_comparison.png` file shows a side-by-side comparison across all architectures.
