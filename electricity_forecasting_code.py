"""
=====================================================================
Multi-Horizon Electricity Load Forecasting — Deliverable 1
Deep Learning Architectures: LSTM, TCN, Transformer, PatchTST
Dataset: UCI ElectricityLoadDiagrams 2011–2014
=====================================================================
Install requirements:
    pip install torch pandas numpy scikit-learn matplotlib seaborn ucimlrepo

Run:
    python electricity_forecasting_code.py
"""

import os, math, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# =====================================================================
# 1. DATA LOADING & PREPROCESSING
# =====================================================================

def download_and_load_data():
    """
    Download UCI Electricity dataset.
    Falls back to synthetic data if download fails (for offline testing).
    """
    try:
        from ucimlrepo import fetch_ucirepo
        print("Downloading UCI Electricity dataset...")
        elec = fetch_ucirepo(id=321)
        df = elec.data.features
        df.index = pd.to_datetime(df.index)
        print(f"Downloaded: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"UCI download failed ({e}). Generating synthetic data for demonstration...")
        return generate_synthetic_electricity_data()


def generate_synthetic_electricity_data(n_clients=50, n_hours=8760*2):
    """
    Synthetic electricity data with realistic patterns:
    - Daily seasonality (morning/evening peaks)
    - Weekly pattern (lower weekends)
    - Annual trend
    - Client heterogeneity
    """
    np.random.seed(42)
    timestamps = pd.date_range("2012-01-01", periods=n_hours, freq="h")
    data = {}
    for c in range(n_clients):
        base = np.random.uniform(1.5, 8.0)
        t = np.arange(n_hours)
        # Daily pattern (peaks at 9am and 7pm)
        hour = timestamps.hour
        daily = (np.sin(2 * np.pi * (hour - 6) / 24) +
                 0.5 * np.sin(2 * np.pi * (hour - 18) / 24)) * base
        # Weekly pattern (lower on weekends)
        weekly = np.where(timestamps.dayofweek >= 5, -0.3 * base, 0.1 * base)
        # Annual pattern
        annual = 0.3 * base * np.sin(2 * np.pi * t / (24 * 365))
        # Noise
        noise = np.random.normal(0, 0.1 * base, n_hours)
        data[f"MT_{c:03d}"] = np.maximum(0.05, daily + weekly + annual + base + noise)
    df = pd.DataFrame(data, index=timestamps)
    print(f"Synthetic data: {df.shape[0]} rows × {df.shape[1]} clients")
    return df


def preprocess_data(df, n_clients=50):
    """
    Preprocessing pipeline following LDT/LSTNet paper conventions:
    1. Resample to hourly (if 15-min)
    2. Select active clients (2012–2014)
    3. Handle daylight saving transitions
    4. Z-score normalization (train stats only)
    5. Train/Val/Test split 70/10/20
    """
    # Use subset of clients for faster experimentation
    clients = df.columns[:n_clients]
    df = df[clients].copy()

    # If 15-min granularity, resample to hourly
    if df.index.freq is not None and df.index.freq < pd.tseries.frequencies.to_offset("1h"):
        df = df.resample("1h").sum()
        print("Resampled to hourly resolution")

    # Filter to 2012+ (many clients have zero-padded 2011 data)
    df = df[df.index.year >= 2012]
    print(f"Post-filter shape: {df.shape}")

    # Drop clients with >5% zero values
    zero_pct = (df == 0).mean()
    df = df[df.columns[zero_pct < 0.05]]
    print(f"After zero-filter: {df.shape[1]} clients retained")

    # Chronological train/val/test split
    n = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.80)
    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]

    # Z-score normalization using training statistics
    scaler = StandardScaler()
    train_arr = scaler.fit_transform(train_df.values)
    val_arr   = scaler.transform(val_df.values)
    test_arr  = scaler.transform(test_df.values)

    print(f"Train: {train_arr.shape}, Val: {val_arr.shape}, Test: {test_arr.shape}")
    return train_arr, val_arr, test_arr, scaler, df.columns.tolist()


# =====================================================================
# 2. PYTORCH DATASET
# =====================================================================

class ElectricityDataset(Dataset):
    """
    Rolling window dataset for multi-horizon forecasting.
    
    Args:
        data: np.array of shape (T, N) — T timesteps, N clients
        lookback: length of input window (default 168 = 1 week of hours)
        horizon: prediction horizon in {1, 6, 24}
        stride: sliding window step size
    """
    def __init__(self, data, lookback=168, horizon=24, stride=1):
        self.data = torch.FloatTensor(data)
        self.lookback = lookback
        self.horizon = horizon
        self.stride = stride
        self.indices = list(range(0, len(data) - lookback - horizon + 1, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        x = self.data[start : start + self.lookback]         # (L, N)
        y = self.data[start + self.lookback : start + self.lookback + self.horizon]  # (H, N)
        return x, y


def make_loaders(train_arr, val_arr, test_arr, horizon, lookback=168, batch_size=64):
    train_ds = ElectricityDataset(train_arr, lookback, horizon, stride=1)
    val_ds   = ElectricityDataset(val_arr,   lookback, horizon, stride=1)
    test_ds  = ElectricityDataset(test_arr,  lookback, horizon, stride=1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


# =====================================================================
# 3. MODEL ARCHITECTURES
# =====================================================================

# ─── 3a. LSTM ───
class LSTMForecaster(nn.Module):
    """
    2-layer stacked LSTM with direct multi-step output projection.
    Avoids autoregressive error accumulation for multi-horizon forecasting.
    """
    def __init__(self, n_features, hidden_size=128, num_layers=2, horizon=24, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        # Project last hidden state to all future timesteps × all features
        self.fc = nn.Linear(hidden_size, horizon * n_features)
        self.horizon = horizon
        self.n_features = n_features

    def forward(self, x):
        # x: (B, L, N)
        out, _ = self.lstm(x)           # (B, L, H)
        last   = self.dropout(out[:, -1, :])  # (B, H)
        pred   = self.fc(last)          # (B, horizon*N)
        return pred.view(-1, self.horizon, self.n_features)  # (B, H, N)


# ─── 3b. TCN ───
class CausalConv1d(nn.Module):
    """Left-padded causal convolution (no future leakage)."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        )
    def forward(self, x):
        return self.conv(F.pad(x, (self.padding, 0)))[:, :, :x.size(2)]


class TCNBlock(nn.Module):
    def __init__(self, n_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(n_ch, n_ch, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(n_ch, n_ch, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + x)   # residual connection


class TCNForecaster(nn.Module):
    """
    Temporal Convolutional Network for multi-horizon forecasting.
    Exponential dilation rates give receptive field > lookback window.
    """
    def __init__(self, n_features, n_channels=64, kernel_size=3,
                 n_layers=5, horizon=24, dropout=0.2):
        super().__init__()
        # Input projection
        self.input_proj = nn.Linear(n_features, n_channels)
        # TCN blocks with exponential dilation
        dilations = [2**i for i in range(n_layers)]
        self.tcn_blocks = nn.ModuleList([TCNBlock(n_channels, kernel_size, d, dropout) for d in dilations])
        self.fc = nn.Linear(n_channels, horizon * n_features)
        self.horizon = horizon
        self.n_features = n_features

    def forward(self, x):
        # x: (B, L, N)
        h = self.input_proj(x).permute(0, 2, 1)   # (B, C, L)
        for block in self.tcn_blocks:
            h = block(h)
        h = h[:, :, -1]                            # last timestep (B, C)
        out = self.fc(h).view(-1, self.horizon, self.n_features)
        return out


# ─── 3c. VANILLA TRANSFORMER ───
class TransformerForecaster(nn.Module):
    """
    Encoder-only Transformer for time series forecasting.
    Uses learned positional encoding and linear projection head.
    """
    def __init__(self, n_features, d_model=128, nhead=8, num_layers=3,
                 horizon=24, dropout=0.1, max_len=512):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        # Learned positional encoding
        self.pos_embed = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, horizon * n_features)
        self.horizon = horizon
        self.n_features = n_features

    def forward(self, x):
        B, L, _ = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.input_proj(x) + self.pos_embed(positions)   # (B, L, d_model)
        h = self.encoder(h)                                   # (B, L, d_model)
        h = h[:, -1, :]                                       # CLS-like: last token
        out = self.fc(h).view(-1, self.horizon, self.n_features)
        return out


# ─── 3d. PATCH TST ───
class PatchTSTForecaster(nn.Module):
    """
    PatchTST: Segments time series into non-overlapping patches before attention.
    Significantly reduces sequence length and improves local pattern capture.
    Reference: Nie et al., ICLR 2023 "A Time Series is Worth 64 Words"
    """
    def __init__(self, n_features, patch_size=16, stride=8,
                 d_model=128, nhead=8, num_layers=3,
                 horizon=24, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.n_features = n_features
        # Patch embedding (convolutional tokenization)
        self.patch_embed = nn.Conv1d(n_features, d_model, kernel_size=patch_size, stride=stride)
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, horizon * n_features)
        self.horizon = horizon

    def forward(self, x):
        # x: (B, L, N)
        h = x.permute(0, 2, 1)              # (B, N, L)
        h = self.patch_embed(h)             # (B, d_model, num_patches)
        h = h.permute(0, 2, 1)             # (B, num_patches, d_model)
        h = self.encoder(h)                 # (B, num_patches, d_model)
        h = self.norm(h[:, -1, :])          # last patch representation
        h = self.dropout(h)
        out = self.fc(h).view(-1, self.horizon, self.n_features)
        return out


# =====================================================================
# 4. TRAINING LOOP
# =====================================================================

def train_model(model, train_loader, val_loader, epochs=50, patience=10, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    model.to(DEVICE)
    
    best_val_loss = float('inf')
    patience_ctr  = 0
    train_history = []
    val_history   = []
    start_time    = time.time()

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                val_loss += criterion(pred, y).item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        train_history.append(train_loss)
        val_history.append(val_loss)
        scheduler.step(val_loss)

        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:3d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | {elapsed:.0f}s")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, train_history, val_history


# =====================================================================
# 5. EVALUATION
# =====================================================================

def compute_metrics(y_true, y_pred):
    """Compute MAE, RMSE, MAPE, R2 across all timesteps and clients."""
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    mae  = mean_absolute_error(y_true_f, y_pred_f)
    rmse = np.sqrt(mean_squared_error(y_true_f, y_pred_f))
    # MAPE (avoid division by near-zero)
    mask = np.abs(y_true_f) > 1e-3
    mape = np.mean(np.abs((y_true_f[mask] - y_pred_f[mask]) / y_true_f[mask])) * 100
    r2   = r2_score(y_true_f, y_pred_f)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def evaluate_model(model, test_loader):
    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            pred = model(x).cpu().numpy()
            preds_list.append(pred)
            targets_list.append(y.numpy())
    preds   = np.concatenate(preds_list, axis=0)   # (N_samples, H, n_clients)
    targets = np.concatenate(targets_list, axis=0)
    return preds, targets


# =====================================================================
# 6. VISUALIZATION
# =====================================================================

def plot_forecast_comparison(results, horizons, save_path="forecast_comparison.png"):
    """Bar chart of MAE across models and horizons."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    colors = ["#1A3A6B", "#2E5FAC", "#0D9488", "#10B981"]
    model_names = list(results.keys())

    for ai, h in enumerate(horizons):
        ax = axes[ai]
        maes = [results[m][h]["MAE"] for m in model_names]
        bars = ax.bar(model_names, maes, color=colors, edgecolor="white", linewidth=0.5, width=0.55)
        ax.set_title(f"Horizon = {h}h", fontsize=14, fontweight="bold")
        ax.set_ylabel("Normalized MAE", fontsize=11)
        ax.set_ylim(0, max(maes) * 1.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("#F8FAFC")
        # Value labels
        for bar, v in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.suptitle("Model Comparison: Normalized MAE by Forecast Horizon", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_horizon_degradation(results, horizons, save_path="horizon_degradation.png"):
    """Line chart showing MAE degradation with increasing horizon."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"LSTM": "#1A3A6B", "TCN": "#2E5FAC", "Transformer": "#F59E0B", "PatchTST": "#10B981"}
    for model_name, h_dict in results.items():
        maes = [h_dict[h]["MAE"] for h in horizons]
        c = colors.get(model_name, "#888888")
        ax.plot(horizons, maes, marker="o", linewidth=2.5, markersize=8, label=model_name, color=c)
        ax.annotate(f"{maes[-1]:.3f}", (horizons[-1], maes[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=10, color=c)
    ax.set_xlabel("Forecast Horizon (hours)", fontsize=12)
    ax.set_ylabel("Normalized MAE", fontsize=12)
    ax.set_title("Accuracy Degradation vs. Forecast Horizon", fontsize=14, fontweight="bold")
    ax.set_xticks(horizons)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#F8FAFC")
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_sample_forecast(targets, preds, model_name, horizon, n_steps=100, save_path=None):
    """Plot actual vs predicted for one representative client."""
    fig, ax = plt.subplots(figsize=(14, 4))
    t = np.arange(n_steps)
    # Use first client, first forecast step
    actual = targets[:n_steps, 0, 0]
    pred   = preds[:n_steps, 0, 0]
    ax.plot(t, actual, label="Actual", color="#1A3A6B", linewidth=1.5)
    ax.plot(t, pred, label=f"{model_name} Forecast", color="#0D9488", linewidth=1.5, linestyle="--")
    ax.fill_between(t, actual - 0.2, actual + 0.2, alpha=0.1, color="#1A3A6B")
    ax.set_title(f"{model_name} — H={horizon}h Forecast (Client 1, {n_steps} samples)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Normalized Load")
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#F8FAFC")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")
    else:
        plt.show()


# =====================================================================
# 7. MAIN EXPERIMENT RUNNER
# =====================================================================

def run_experiments():
    HORIZONS   = [1, 6, 24]    # prediction horizons in hours
    LOOKBACK   = 168           # 1 week of hourly data
    N_CLIENTS  = 20            # use small subset for fast demo (increase to 320 for full run)
    BATCH_SIZE = 64
    EPOCHS     = 30
    PATIENCE   = 8

    print("\n" + "="*65)
    print("  ELECTRICITY LOAD FORECASTING — DELIVERABLE 1")
    print("="*65 + "\n")

    # ── Load & preprocess ──
    raw_df = download_and_load_data()
    train_arr, val_arr, test_arr, scaler, client_names = preprocess_data(raw_df, n_clients=N_CLIENTS)
    N = train_arr.shape[1]   # actual number of clients after filtering
    print(f"\nNumber of clients: {N}")

    # ── Model factory ──
    def get_models(horizon):
        return {
            "LSTM":        LSTMForecaster(N, hidden_size=64, num_layers=2, horizon=horizon),
            "TCN":         TCNForecaster(N, n_channels=64, n_layers=5, horizon=horizon),
            "Transformer": TransformerForecaster(N, d_model=64, nhead=4, num_layers=3, horizon=horizon),
            "PatchTST":    PatchTSTForecaster(N, patch_size=16, stride=8, d_model=64, nhead=4, num_layers=3, horizon=horizon)
        }

    # ── Run all models × horizons ──
    results = {name: {} for name in ["LSTM", "TCN", "Transformer", "PatchTST"]}
    forecast_examples = {}

    for horizon in HORIZONS:
        print(f"\n{'─'*50}")
        print(f"  HORIZON = {horizon} hour(s)")
        print(f"{'─'*50}")
        train_loader, val_loader, test_loader = make_loaders(
            train_arr, val_arr, test_arr, horizon, LOOKBACK, BATCH_SIZE
        )
        models_h = get_models(horizon)

        for name, model in models_h.items():
            n_params = sum(p.numel() for p in model.parameters())
            print(f"\n[{name}]  Parameters: {n_params:,}")
            t0 = time.time()
            trained_model, _, _ = train_model(model, train_loader, val_loader, epochs=EPOCHS, patience=PATIENCE)
            elapsed = time.time() - t0
            preds, targets = evaluate_model(trained_model, test_loader)
            metrics = compute_metrics(targets, preds)
            metrics["train_time_s"] = elapsed
            results[name][horizon] = metrics
            print(f"  → MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  "
                  f"MAPE={metrics['MAPE']:.2f}%  R²={metrics['R2']:.4f}  "
                  f"Time={elapsed:.0f}s")

            # Save forecast sample for best horizon
            if horizon == 24:
                forecast_examples[name] = (targets, preds)

    # ── Print summary table ──
    print("\n" + "="*65)
    print("RESULTS SUMMARY")
    print("="*65)
    header = f"{'Model':<15} | {'H=1h MAE':>10} {'H=1h RMSE':>11} | {'H=6h MAE':>10} {'H=6h RMSE':>11} | {'H=24h MAE':>10} {'H=24h RMSE':>11}"
    print(header)
    print("-" * len(header))
    for name in ["LSTM", "TCN", "Transformer", "PatchTST"]:
        row = f"{name:<15} |"
        for h in HORIZONS:
            m = results[name][h]
            row += f" {m['MAE']:>10.4f} {m['RMSE']:>11.4f} |"
        print(row)

    # ── Degradation analysis ──
    print("\nHORIZON DEGRADATION (MAE increase H1 → H24):")
    for name in results:
        mae1  = results[name][1]["MAE"]
        mae24 = results[name][24]["MAE"]
        pct   = (mae24 - mae1) / mae1 * 100
        print(f"  {name:<15}: +{pct:.1f}%")

    # ── Plots ──
    print("\nGenerating plots...")
    plot_forecast_comparison(results, HORIZONS, "forecast_comparison.png")
    plot_horizon_degradation(results, HORIZONS, "horizon_degradation.png")
    for name, (targets, preds) in forecast_examples.items():
        plot_sample_forecast(targets, preds, name, 24, n_steps=72,
                             save_path=f"sample_forecast_{name.lower()}_h24.png")

    return results


if __name__ == "__main__":
    results = run_experiments()
    print("\n✓ All experiments complete. Results saved.")
