"""
=====================================================================
Deliverable 2 — Multi-Horizon Electricity Load Forecasting
Per-Cluster Models (LSTM, TCN, Transformer, PatchTST) with MAPE
Dataset: UCI ElectricityLoadDiagrams 2011–2014
=====================================================================
Install:
    pip install torch pandas numpy scikit-learn matplotlib seaborn pyarrow

Run:
    PYTHONUNBUFFERED=1 python deliverable_2.py
=====================================================================
"""

import os, time, pickle, json, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# ── Config ───────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
HORIZONS    = [1, 6, 24]
LOOKBACK    = 168          # 1 week
N_CLUSTERS  = 3
BATCH_SIZE  = 256
EPOCHS      = 10
PATIENCE    = 3
LR          = 1e-3
OUT_DIR     = "d2_outputs"
MODEL_DIR   = "d2_models"
N_TIME_FEAT = 8            # number of time features
MODEL_NAMES = ["lstm", "tcn", "transformer", "patchtst"]
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Horizons: {HORIZONS}, Lookback: {LOOKBACK}h, Clusters: {N_CLUSTERS}")


# =====================================================================
# 1. DATA LOADING
# =====================================================================

def load_data():
    """Load UCI Electricity dataset. Caches as parquet for fast reloads."""
    cache_path = "electricity_cache.parquet"
    if os.path.exists(cache_path):
        print(f"Loading cached parquet: {cache_path}")
        df = pd.read_parquet(cache_path)
        print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} clients")
        return df

    local_paths = ["LD2011_2014.txt", "data/LD2011_2014.txt"]
    for path in local_paths:
        if os.path.exists(path):
            print(f"Loading CSV: {path} (first time, ~2 min)...")
            df = pd.read_csv(path, sep=";", decimal=",", index_col=0, parse_dates=True)
            df = df.apply(pd.to_numeric, errors="coerce")
            df.to_parquet(cache_path)
            print(f"  Loaded & cached: {df.shape}")
            return df

    # Synthetic fallback
    print("No dataset found — generating synthetic data...")
    np.random.seed(42)
    idx = pd.date_range("2012-01-01", periods=26280, freq="h")
    data = {}
    for c in range(80):
        base = np.random.uniform(1.0, 40.0)
        hour, dow, doy = idx.hour, idx.dayofweek, idx.dayofyear
        daily = (np.sin(2*np.pi*(hour-8)/24)*0.5 + 0.3*np.sin(2*np.pi*(hour-18)/12))*base
        weekly = np.where(dow >= 5, -0.2*base, 0.05*base)
        annual = 0.15*base*np.sin(2*np.pi*doy/365 - np.pi/2)
        noise = np.random.normal(0, 0.08*base, len(idx))
        data[f"MT_{c+1:03d}"] = np.maximum(0.01, base + daily + weekly + annual + noise)
    df = pd.DataFrame(data, index=idx)
    print(f"  Synthetic: {df.shape}")
    return df


# =====================================================================
# 2. PREPROCESSING — CHANNEL-INDEPENDENT APPROACH
# =====================================================================

def build_time_features(index):
    """Cyclical + binary time features."""
    hour, dow, month = index.hour, index.dayofweek, index.month
    return np.column_stack([
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7),
        np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
        (dow >= 5).astype(float),
        ((hour >= 8) & (hour <= 18) & (dow < 5)).astype(float),
    ]).astype(np.float32)


def preprocess(df):
    """
    Channel-independent preprocessing:
    - Each client is treated as a separate univariate time series
    - One model per cluster, but each client is a separate sample in the batch
    - This makes training feasible on CPU with 300+ clients
    """
    # Resample to hourly if 15-min
    freq = pd.infer_freq(df.index[:100])
    if freq and ("15" in str(freq) or "T" in str(freq)):
        print("Resampling 15-min → hourly (sum)...")
        df = df.resample("1h").sum()

    # Filter to 2012+
    df = df[df.index.year >= 2012].copy()
    print(f"After 2012 filter: {df.shape[0]:,} rows × {df.shape[1]} clients")

    # Drop clients with >5% zeros
    zero_pct = (df == 0).mean()
    df = df[df.columns[zero_pct < 0.05]].copy()
    print(f"Active clients (<=5% zeros): {df.shape[1]}")

    # Build time features
    time_feats = build_time_features(df.index)
    print(f"Time features: {time_feats.shape[1]} features")

    # Cluster clients by daily load profile (K-Means on normalized hourly means)
    print(f"\nClustering {df.shape[1]} clients into {N_CLUSTERS} groups...")
    daily_profiles = df.groupby(df.index.hour).mean().T  # (n_clients, 24)
    daily_norm = (daily_profiles - daily_profiles.mean(axis=1).values[:, None]) / \
                 (daily_profiles.std(axis=1).values[:, None] + 1e-8)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(daily_norm)

    client_cluster_map = {}
    cluster_clients = {}
    for i, col in enumerate(df.columns):
        cl = int(cluster_labels[i])
        client_cluster_map[col] = cl
        cluster_clients.setdefault(cl, []).append(col)

    for cl in sorted(cluster_clients):
        members = cluster_clients[cl]
        avg = df[members].mean().mean()
        print(f"  Cluster {cl}: {len(members)} clients, avg load = {avg:.2f} kWh/hr")

    # Train/Val/Test split 70/10/20 (chronological)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.80)
    print(f"\n  Train: 0–{train_end} ({df.index[0].date()} → {df.index[train_end-1].date()})")
    print(f"  Val:   {train_end}–{val_end} ({df.index[train_end].date()} → {df.index[val_end-1].date()})")
    print(f"  Test:  {val_end}–{n} ({df.index[val_end].date()} → {df.index[-1].date()})")

    # Per-client normalization (Z-score, fit on train only)
    # Store as dict: {client_id: {"train": arr, "val": arr, "test": arr, "scaler": scaler}}
    client_data = {}
    for col in df.columns:
        vals = df[col].values.reshape(-1, 1).astype(np.float32)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(vals[:train_end])
        val_scaled = scaler.transform(vals[train_end:val_end])
        test_scaled = scaler.transform(vals[val_end:])
        client_data[col] = {
            "train": train_scaled.flatten(),
            "val": val_scaled.flatten(),
            "test": test_scaled.flatten(),
            "scaler": scaler,
        }

    metadata = {
        "client_cluster_map": client_cluster_map,
        "cluster_clients": {str(k): v for k, v in cluster_clients.items()},
        "n_clusters": N_CLUSTERS,
        "test_index": df.index[val_end:].tolist(),
        "all_clients": df.columns.tolist(),
        "train_end": train_end,
        "val_end": val_end,
    }

    return client_data, time_feats, metadata


# =====================================================================
# 3. DATASET — CHANNEL-INDEPENDENT
# =====================================================================

class ChannelIndependentDataset(Dataset):
    """
    Each sample = one (lookback window, horizon target) from ONE client.
    All clients in a cluster are stacked into the same dataset.
    Input: (L, 1 + N_time_feat)  — univariate load + time features
    Target: (H,)  — future load values
    """
    def __init__(self, client_arrays, time_feats, lookback=168, horizon=24, stride=4):
        self.L = lookback
        self.H = horizon
        # Build index: (client_idx, time_start)
        self.samples = []
        self.load_data = []
        self.time_data = torch.FloatTensor(time_feats)

        for ci, arr in enumerate(client_arrays):
            self.load_data.append(torch.FloatTensor(arr))
            n_windows = (len(arr) - lookback - horizon) // stride + 1
            for w in range(n_windows):
                self.samples.append((ci, w * stride))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ci, start = self.samples[idx]
        load = self.load_data[ci]
        # Input: lookback window of load + time features
        x_load = load[start:start+self.L].unsqueeze(1)     # (L, 1)
        x_time = self.time_data[start:start+self.L]        # (L, N_time)
        x = torch.cat([x_load, x_time], dim=1)             # (L, 1+N_time)
        # Target: next H steps
        y = load[start+self.L:start+self.L+self.H]         # (H,)
        return x, y, ci


def make_loaders(client_ids, client_data, time_feats_train, time_feats_val,
                 time_feats_test, horizon, lookback=168, batch_size=128):
    train_arrays = [client_data[c]["train"] for c in client_ids]
    val_arrays = [client_data[c]["val"] for c in client_ids]
    test_arrays = [client_data[c]["test"] for c in client_ids]

    train_ds = ChannelIndependentDataset(train_arrays, time_feats_train, lookback, horizon, stride=96)
    val_ds = ChannelIndependentDataset(val_arrays, time_feats_val, lookback, horizon, stride=48)
    test_ds = ChannelIndependentDataset(test_arrays, time_feats_test, lookback, horizon, stride=24)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


# =====================================================================
# 4. MODELS (Channel-Independent)
# =====================================================================

# ─── 4a. LSTM ───
class LSTMForecaster(nn.Module):
    """
    2-layer stacked LSTM — channel-independent.
    Input: (B, L, n_input)  Output: (B, H)
    """
    def __init__(self, n_input=9, hidden_size=64, num_layers=2,
                 horizon=24, dropout=0.2):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(
            input_size=n_input, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x):
        out, _ = self.lstm(x)                    # (B, L, H_dim)
        last = self.dropout(out[:, -1, :])       # (B, H_dim)
        return self.head(last)                   # (B, H)


# ─── 4b. TCN ───
class CausalConv1d(nn.Module):
    """Left-padded causal convolution (no future leakage)."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        )
    def forward(self, x):
        return self.conv(torch.nn.functional.pad(x, (self.padding, 0)))[:, :, :x.size(2)]


class TCNBlock(nn.Module):
    def __init__(self, n_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(n_ch, n_ch, kernel_size, dilation),
            nn.ReLU(), nn.Dropout(dropout),
            CausalConv1d(n_ch, n_ch, kernel_size, dilation),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.net(x) + x)


class TCNForecaster(nn.Module):
    """
    Temporal Convolutional Network — channel-independent.
    Input: (B, L, n_input)  Output: (B, H)
    """
    def __init__(self, n_input=9, n_channels=48, kernel_size=3,
                 n_layers=5, horizon=24, dropout=0.2):
        super().__init__()
        self.horizon = horizon
        self.input_proj = nn.Linear(n_input, n_channels)
        dilations = [2**i for i in range(n_layers)]
        self.tcn_blocks = nn.ModuleList(
            [TCNBlock(n_channels, kernel_size, d, dropout) for d in dilations]
        )
        self.head = nn.Sequential(
            nn.Linear(n_channels, n_channels),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(n_channels, horizon),
        )

    def forward(self, x):
        h = self.input_proj(x).permute(0, 2, 1)   # (B, C, L)
        for block in self.tcn_blocks:
            h = block(h)
        h = h[:, :, -1]                            # (B, C)
        return self.head(h)                        # (B, H)


# ─── 4c. Vanilla Transformer ───
class TransformerForecaster(nn.Module):
    """
    Encoder-only Transformer — channel-independent.
    Input: (B, L, n_input)  Output: (B, H)
    """
    def __init__(self, n_input=9, d_model=48, nhead=4, num_layers=2,
                 horizon=24, dropout=0.1, max_len=512):
        super().__init__()
        self.horizon = horizon
        self.input_proj = nn.Linear(n_input, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x):
        B, L, _ = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.input_proj(x) + self.pos_embed(positions)
        h = self.encoder(h)
        h = h[:, -1, :]                           # last token
        return self.head(h)                       # (B, H)


# ─── 4d. PatchTST ───

class PatchTSTForecaster(nn.Module):
    """
    Channel-independent PatchTST.
    Input:  (B, L, 1 + N_time_features)
    Output: (B, H)
    """
    def __init__(self, n_input=9, d_model=64, nhead=4, num_layers=3,
                 patch_size=16, stride=8, horizon=24, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.patch_size = patch_size
        self.stride = stride
        # Patch embedding via Conv1d
        self.patch_embed = nn.Conv1d(n_input, d_model, kernel_size=patch_size, stride=stride)
        n_patches = (LOOKBACK - patch_size) // stride + 1
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x):
        # x: (B, L, n_input)
        h = x.permute(0, 2, 1)              # (B, n_input, L)
        h = self.patch_embed(h)              # (B, d_model, n_patches)
        h = h.permute(0, 2, 1)              # (B, n_patches, d_model)
        h = h + self.pos_embed
        h = self.encoder(h)
        h = self.norm(h[:, -1, :])           # last patch
        return self.head(h)                  # (B, H)


def build_model(name, n_input, horizon):
    """Factory to create a model by name."""
    if name == "lstm":
        return LSTMForecaster(n_input=n_input, hidden_size=64, num_layers=2,
                              horizon=horizon, dropout=0.2)
    elif name == "tcn":
        return TCNForecaster(n_input=n_input, n_channels=48, kernel_size=3,
                             n_layers=4, horizon=horizon, dropout=0.2)
    elif name == "transformer":
        return TransformerForecaster(n_input=n_input, d_model=48, nhead=4,
                                     num_layers=2, horizon=horizon, dropout=0.1)
    elif name == "patchtst":
        return PatchTSTForecaster(n_input=n_input, d_model=48, nhead=4,
                                   num_layers=2, patch_size=16, stride=8,
                                   horizon=horizon, dropout=0.1)
    else:
        raise ValueError(f"Unknown model: {name}")


# =====================================================================
# 5. TRAINING
# =====================================================================

def train_model(model, train_loader, val_loader, epochs=EPOCHS, patience=PATIENCE, lr=LR):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    model.to(DEVICE)

    best_val_loss = float("inf")
    patience_ctr = 0
    best_state = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            n_train += x.size(0)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
                val_loss += criterion(model(x), y).item() * x.size(0)
                n_val += x.size(0)
        val_loss /= n_val
        scheduler.step()

        if epoch % 3 == 0 or epoch == 1:
            print(f"    Ep {epoch:3d} | Train: {train_loss:.5f} | Val: {val_loss:.5f} | {time.time()-t0:.0f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"    Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()
    elapsed = time.time() - t0
    print(f"    Best val: {best_val_loss:.5f} | Time: {elapsed:.0f}s")
    return model, elapsed


# =====================================================================
# 6. EVALUATION — MAPE FOCUSED
# =====================================================================

def evaluate_on_test(model, test_loader, client_scalers, client_ids):
    """
    Run inference, inverse-transform per-client to original scale, compute MAPE.
    Returns per-sample APEs and overall MAPE.
    """
    model.eval()
    preds_list, targets_list, cidx_list = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x, y, ci = batch[0].to(DEVICE), batch[1], batch[2]
            pred = model(x).cpu().numpy()   # (B, H)
            preds_list.append(pred)
            targets_list.append(y.numpy())
            cidx_list.append(ci.numpy())

    preds = np.concatenate(preds_list, axis=0)      # (N, H)
    targets = np.concatenate(targets_list, axis=0)   # (N, H)
    cidx = np.concatenate(cidx_list, axis=0)         # (N,)

    # Per-client inverse transform
    preds_orig = np.zeros_like(preds)
    targets_orig = np.zeros_like(targets)
    for i in range(len(preds)):
        ci = int(cidx[i])
        scaler = client_scalers[client_ids[ci]]
        mean, std = scaler.mean_[0], scaler.scale_[0]
        preds_orig[i] = preds[i] * std + mean
        targets_orig[i] = targets[i] * std + mean

    # MAPE (0-100 scale)
    mask = np.abs(targets_orig) > 0.5
    if mask.sum() == 0:
        mape = 0.0
    else:
        mape = np.mean(np.abs((targets_orig[mask] - preds_orig[mask]) / targets_orig[mask])) * 100

    # Per-sample APE (averaged over horizon steps)
    n_samples = len(preds)
    per_sample_ape = np.zeros(n_samples)
    for i in range(n_samples):
        t = targets_orig[i]
        p = preds_orig[i]
        m = np.abs(t) > 0.5
        if m.sum() > 0:
            per_sample_ape[i] = np.mean(np.abs((t[m] - p[m]) / t[m])) * 100

    return mape, per_sample_ape, preds_orig, targets_orig


# =====================================================================
# 7. VISUALIZATION
# =====================================================================

def plot_mape_boxplots(period_results, horizon, cluster_id, save_path):
    """Box plot of APE distribution across test periods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    data = list(period_results.values())
    labels = list(period_results.keys())
    mapes = [np.mean(v) for v in data]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=3, alpha=0.3),
                    whis=1.5)
    colors = ["#1A3A6B", "#2E5FAC", "#0D9488", "#10B981"]
    for patch, color in zip(bp["boxes"], colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, mape_val in enumerate(mapes):
        ax.text(i + 1, ax.get_ylim()[1] * 0.92, f"MAPE={mape_val:.1f}%",
                ha="center", fontsize=10, fontweight="bold", color=colors[i % len(colors)])

    ax.set_title(f"Error Distribution by Test Period — Cluster {cluster_id}, H={horizon}h",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Absolute Percentage Error (%)")
    ax.set_xlabel("Test Period")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


def plot_overall_comparison(all_results, save_path):
    """Grouped bar chart: MAPE per model per cluster per horizon."""
    model_colors = {
        "lstm": "#E74C3C", "tcn": "#F39C12",
        "transformer": "#2E5FAC", "patchtst": "#0D9488",
    }
    n_horizons = len(HORIZONS)
    fig, axes = plt.subplots(1, n_horizons, figsize=(6 * n_horizons, 6), sharey=False)

    for ai, h in enumerate(HORIZONS):
        ax = axes[ai]
        clusters = sorted(set(cl for m in all_results for cl in all_results[m]))
        models_present = [m for m in MODEL_NAMES if m in all_results]
        n_models = len(models_present)
        bar_w = 0.8 / n_models
        x = np.arange(len(clusters))

        for mi, mname in enumerate(models_present):
            mapes = []
            for cl in clusters:
                if cl in all_results[mname] and h in all_results[mname][cl]:
                    mapes.append(all_results[mname][cl][h]["mape"])
                else:
                    mapes.append(0)
            offset = (mi - n_models / 2 + 0.5) * bar_w
            bars = ax.bar(x + offset, mapes, bar_w, label=mname.upper(),
                          color=model_colors.get(mname, "#999"), edgecolor="white")
            for bar, v in zip(bars, mapes):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                            f"{v:.1f}", ha="center", fontsize=7, fontweight="bold")

        cl_labels = [f"Cl {cl}" for cl in clusters]
        ax.set_xticks(x)
        ax.set_xticklabels(cl_labels)
        ax.set_title(f"H = {h}h", fontsize=14, fontweight="bold")
        ax.set_ylabel("MAPE (%)" if ai == 0 else "")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        if ai == n_horizons - 1:
            ax.legend(fontsize=9)

    plt.suptitle("MAPE by Model, Cluster & Horizon", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_sample_forecast(targets, preds, cluster_id, horizon, model_name, save_path):
    """Plot actual vs predicted for a sample segment."""
    fig, ax = plt.subplots(figsize=(14, 5))
    n = min(200, len(targets))
    actual = targets[:n, 0]
    predicted = preds[:n, 0]
    ax.plot(actual, label="Actual", color="#1A3A6B", lw=1.5)
    ax.plot(predicted, label=f"{model_name.upper()} Forecast", color="#0D9488", lw=1.5, ls="--")
    ax.set_title(f"Cluster {cluster_id} — H={horizon}h Sample ({model_name.upper()})",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Load (kWh/hr)")
    ax.set_xlabel("Sample Index")
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


def analyze_test_periods(apes, n_periods=4):
    """Split APEs into equal periods, return dict."""
    chunk = len(apes) // n_periods
    periods = {}
    for p in range(n_periods):
        start = p * chunk
        end = (p+1) * chunk if p < n_periods - 1 else len(apes)
        periods[f"Period {p+1}"] = apes[start:end]
    return periods


# =====================================================================
# 8. MAIN EXPERIMENT
# =====================================================================

def run_deliverable_2():
    print("\n" + "=" * 65)
    print("  DELIVERABLE 2 — 4-MODEL PER-CLUSTER FORECASTING")
    print("  LSTM · TCN · Transformer · PatchTST")
    print("  Channel-Independent Architecture")
    print("=" * 65 + "\n")

    # Load & preprocess
    raw_df = load_data()
    client_data, time_feats, metadata = preprocess(raw_df)

    train_end = metadata["train_end"]
    val_end = metadata["val_end"]
    tf_train = time_feats[:train_end]
    tf_val = time_feats[train_end:val_end]
    tf_test = time_feats[val_end:]

    cluster_clients = metadata["cluster_clients"]
    n_input = 1 + N_TIME_FEAT  # 1 load channel + 8 time features

    # all_results[model_name][cluster][horizon] = {...}
    all_results = {m: {} for m in MODEL_NAMES}
    n_test_periods = 4
    best_models = {}  # (cluster, horizon) → model_name with lowest MAPE

    # Client scalers dict for inverse transform
    client_scalers = {cid: client_data[cid]["scaler"] for cid in metadata["all_clients"]}

    total_models = len(MODEL_NAMES) * len(cluster_clients) * len(HORIZONS)
    model_counter = 0

    for cl_key in sorted(cluster_clients.keys()):
        cl = int(cl_key)
        clients = cluster_clients[cl_key]

        print(f"\n{'═' * 60}")
        print(f"  CLUSTER {cl}  ({len(clients)} clients)")
        print(f"{'═' * 60}")

        for horizon in HORIZONS:
            print(f"\n  ── Horizon = {horizon}h ──")

            # Build loaders (shared across all models for this cluster+horizon)
            train_loader, val_loader, test_loader = make_loaders(
                clients, client_data, tf_train, tf_val, tf_test,
                horizon, LOOKBACK, BATCH_SIZE
            )
            print(f"    Samples: train={len(train_loader.dataset)}, "
                  f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")

            best_mape_here = float("inf")

            for model_name in MODEL_NAMES:
                model_counter += 1
                print(f"\n    [{model_counter}/{total_models}] {model_name.upper()}")

                model = build_model(model_name, n_input, horizon)
                n_params = sum(p.numel() for p in model.parameters())
                print(f"      Params: {n_params:,}")

                # Train
                trained_model, train_time = train_model(model, train_loader, val_loader)

                # Evaluate
                mape, per_sample_ape, preds_orig, targets_orig = evaluate_on_test(
                    trained_model, test_loader, client_scalers, clients
                )
                print(f"      Test MAPE: {mape:.2f}%  (median APE: {np.median(per_sample_ape):.2f}%)")

                # Period analysis
                periods = analyze_test_periods(per_sample_ape, n_test_periods)
                print(f"      Periods: {', '.join(f'{k}: {np.mean(v):.1f}%' for k, v in periods.items())}")

                # Box plot
                plot_mape_boxplots(periods, horizon, cl,
                                   f"{OUT_DIR}/boxplot_{model_name}_cluster{cl}_h{horizon}.png")

                # Sample forecast (H=24 only)
                if horizon == 24:
                    plot_sample_forecast(targets_orig, preds_orig, cl, horizon, model_name,
                                         f"{OUT_DIR}/forecast_{model_name}_cluster{cl}_h{horizon}.png")

                # Store results
                if cl not in all_results[model_name]:
                    all_results[model_name][cl] = {}
                all_results[model_name][cl][horizon] = {
                    "mape": round(mape, 2),
                    "median_ape": round(float(np.median(per_sample_ape)), 2),
                    "std_ape": round(float(np.std(per_sample_ape)), 2),
                    "n_clients": len(clients),
                    "n_params": n_params,
                    "train_time_s": round(train_time, 1),
                    "period_mapes": {k: round(float(np.mean(v)), 2) for k, v in periods.items()},
                }

                # Track best model
                if mape < best_mape_here:
                    best_mape_here = mape
                    best_models[(cl, horizon)] = model_name

                # Save model checkpoint
                torch.save({
                    "model_state": trained_model.state_dict(),
                    "model_name": model_name,
                    "n_input": n_input,
                    "horizon": horizon,
                    "cluster_id": cl,
                }, f"{MODEL_DIR}/{model_name}_cluster{cl}_h{horizon}.pt")

            print(f"\n    >> Best for Cluster {cl} H={horizon}h: "
                  f"{best_models[(cl, horizon)].upper()} ({best_mape_here:.2f}%)")

    # Overall comparison plot
    plot_overall_comparison(all_results, f"{OUT_DIR}/mape_comparison.png")

    # ── Print final summary ──
    print("\n" + "=" * 70)
    print("  FINAL RESULTS — MAPE (%) on Test Set — ALL MODELS")
    print("=" * 70)

    for model_name in MODEL_NAMES:
        print(f"\n  {model_name.upper()}")
        header = f"  {'Cluster':<12} | {'Clients':>8} | {'H=1h':>8} | {'H=6h':>8} | {'H=24h':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for cl in sorted(all_results[model_name].keys()):
            n = all_results[model_name][cl][HORIZONS[0]]["n_clients"]
            ms = [all_results[model_name][cl][h]["mape"] for h in HORIZONS]
            print(f"  Cluster {cl:<4} | {n:>8} | {ms[0]:>7.2f}% | {ms[1]:>7.2f}% | {ms[2]:>7.2f}%")

    # Best model per cluster+horizon
    print(f"\n{'─' * 60}")
    print("  BEST MODEL PER CLUSTER + HORIZON")
    print(f"{'─' * 60}")
    for cl in sorted(set(k[0] for k in best_models)):
        for h in HORIZONS:
            bm = best_models[(cl, h)]
            bv = all_results[bm][cl][h]["mape"]
            print(f"  Cluster {cl}, H={h}h → {bm.upper():>14s}  ({bv:.2f}%)")

    # Weighted overall per model
    print(f"\n{'─' * 60}")
    print("  WEIGHTED OVERALL MAPE")
    print(f"{'─' * 60}")
    for model_name in MODEL_NAMES:
        for h in HORIZONS:
            tw = sum(all_results[model_name][cl][h]["mape"] * all_results[model_name][cl][h]["n_clients"]
                     for cl in all_results[model_name])
            tn = sum(all_results[model_name][cl][h]["n_clients"] for cl in all_results[model_name])
            print(f"  {model_name.upper():>14s} H={h:>2d}h: {tw / tn:.2f}%")

    # Save metadata for agentic AI
    save_meta = {
        "client_cluster_map": metadata["client_cluster_map"],
        "cluster_clients": metadata["cluster_clients"],
        "all_results": {m: {str(cl): v for cl, v in res.items()} for m, res in all_results.items()},
        "best_models": {f"{cl}_{h}": m for (cl, h), m in best_models.items()},
        "horizons": HORIZONS,
        "model_names": MODEL_NAMES,
        "model_dir": MODEL_DIR,
        "n_clusters": N_CLUSTERS,
    }
    with open(f"{MODEL_DIR}/metadata.json", "w") as f:
        json.dump(save_meta, f, indent=2, default=str)

    # Save per-client scalers
    scalers = {cid: client_data[cid]["scaler"] for cid in metadata["all_clients"]}
    with open(f"{MODEL_DIR}/scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)

    print(f"\nMetadata: {MODEL_DIR}/metadata.json")
    print(f"Scalers:  {MODEL_DIR}/scalers.pkl")
    print(f"Plots:    {OUT_DIR}/")
    print(f"\nDeliverable 2 complete! Trained {total_models} models.")
    return all_results


if __name__ == "__main__":
    results = run_deliverable_2()
