"""
=====================================================================
Electricity Load Forecasting — Agentic AI Interface
Streamlit chat app with 4 DL models (LSTM, TCN, Transformer, PatchTST)
=====================================================================
Run:
    export OPENAI_API_KEY="sk-..."
    python3 -m streamlit run app.py
=====================================================================
"""

import os, json, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
from openai import OpenAI

MODEL_DIR = "d2_models"
LOOKBACK = 168
DEVICE = "cpu"
N_TIME_FEAT = 8

st.set_page_config(
    page_title="Electricity Forecasting Agent",
    page_icon="⚡",
    layout="wide",
)

# =====================================================================
# 1. MODEL DEFINITIONS (same as deliverable_2.py)
# =====================================================================

class LSTMForecaster(nn.Module):
    def __init__(self, n_input=9, hidden_size=64, num_layers=2,
                 horizon=24, dropout=0.2):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(input_size=n_input, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, horizon))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :]))


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation))
    def forward(self, x):
        return self.conv(torch.nn.functional.pad(x, (self.padding, 0)))[:, :, :x.size(2)]


class TCNBlock(nn.Module):
    def __init__(self, n_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(n_ch, n_ch, kernel_size, dilation),
            nn.ReLU(), nn.Dropout(dropout),
            CausalConv1d(n_ch, n_ch, kernel_size, dilation),
            nn.ReLU(), nn.Dropout(dropout))
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.net(x) + x)


class TCNForecaster(nn.Module):
    def __init__(self, n_input=9, n_channels=48, kernel_size=3,
                 n_layers=5, horizon=24, dropout=0.2):
        super().__init__()
        self.horizon = horizon
        self.input_proj = nn.Linear(n_input, n_channels)
        dilations = [2**i for i in range(n_layers)]
        self.tcn_blocks = nn.ModuleList(
            [TCNBlock(n_channels, kernel_size, d, dropout) for d in dilations])
        self.head = nn.Sequential(
            nn.Linear(n_channels, n_channels), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(n_channels, horizon))

    def forward(self, x):
        h = self.input_proj(x).permute(0, 2, 1)
        for block in self.tcn_blocks:
            h = block(h)
        return self.head(h[:, :, -1])


class TransformerForecaster(nn.Module):
    def __init__(self, n_input=9, d_model=48, nhead=4, num_layers=2,
                 horizon=24, dropout=0.1, max_len=512):
        super().__init__()
        self.horizon = horizon
        self.input_proj = nn.Linear(n_input, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, horizon))

    def forward(self, x):
        B, L, _ = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.input_proj(x) + self.pos_embed(positions)
        h = self.encoder(h)
        return self.head(h[:, -1, :])


class PatchTSTForecaster(nn.Module):
    def __init__(self, n_input=9, d_model=48, nhead=4, num_layers=2,
                 patch_size=16, stride=8, horizon=24, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.patch_embed = nn.Conv1d(n_input, d_model, kernel_size=patch_size, stride=stride)
        n_patches = (LOOKBACK - patch_size) // stride + 1
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, horizon))

    def forward(self, x):
        h = x.permute(0, 2, 1)
        h = self.patch_embed(h)
        h = h.permute(0, 2, 1)
        h = h + self.pos_embed
        h = self.encoder(h)
        h = self.norm(h[:, -1, :])
        return self.head(h)


MODEL_CLASSES = {
    "lstm": LSTMForecaster,
    "tcn": TCNForecaster,
    "transformer": TransformerForecaster,
    "patchtst": PatchTSTForecaster,
}


# =====================================================================
# 2. FORECAST ENGINE
# =====================================================================

def build_time_features(index):
    hour, dow, month = index.hour, index.dayofweek, index.month
    return np.column_stack([
        np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
        np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7),
        np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12),
        (dow >= 5).astype(float),
        ((hour >= 8) & (hour <= 18) & (dow < 5)).astype(float),
    ]).astype(np.float32)


@st.cache_resource
def load_all_resources():
    """Load metadata, scalers, data, and all models once."""
    meta_path = os.path.join(MODEL_DIR, "metadata.json")
    if not os.path.exists(meta_path):
        return None, None, None, None

    with open(meta_path) as f:
        metadata = json.load(f)

    scaler_path = os.path.join(MODEL_DIR, "scalers.pkl")
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)

    # Load raw data
    if os.path.exists("electricity_cache.parquet"):
        df = pd.read_parquet("electricity_cache.parquet")
        freq = pd.infer_freq(df.index[:100])
        if freq and ("15" in str(freq) or "T" in str(freq)):
            df = df.resample("1h").sum()
        df = df[df.index.year >= 2012]
        zero_pct = (df == 0).mean()
        df = df[df.columns[zero_pct < 0.05]]
    else:
        df = None

    # Load ALL model checkpoints
    models = {}  # (model_name, cluster, horizon) → model
    model_names = metadata.get("model_names", ["patchtst"])
    n_clusters = metadata.get("n_clusters", 3)
    horizons = metadata.get("horizons", [1, 6, 24])

    for mname in model_names:
        for cl in range(n_clusters):
            for h in horizons:
                path = os.path.join(MODEL_DIR, f"{mname}_cluster{cl}_h{h}.pt")
                if not os.path.exists(path):
                    # Fallback: old naming (patchtst only)
                    if mname == "patchtst":
                        path = os.path.join(MODEL_DIR, f"patchtst_cluster{cl}_h{h}.pt")
                    if not os.path.exists(path):
                        continue
                ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
                n_input = ckpt.get("n_input", 9)
                horizon = ckpt.get("horizon", h)
                cls = MODEL_CLASSES.get(mname)
                if cls is None:
                    continue
                # Infer architecture params from checkpoint state dict
                state = ckpt["model_state"]
                if mname == "tcn":
                    n_layers = max(int(k.split(".")[1]) for k in state if k.startswith("tcn_blocks.")) + 1
                    model = cls(n_input=n_input, horizon=horizon, n_layers=n_layers)
                elif mname == "transformer":
                    n_enc = max(int(k.split(".")[2]) for k in state if k.startswith("encoder.layers.")) + 1
                    model = cls(n_input=n_input, horizon=horizon, num_layers=n_enc)
                elif mname == "patchtst":
                    n_enc = max(int(k.split(".")[2]) for k in state if k.startswith("encoder.layers.")) + 1
                    model = cls(n_input=n_input, horizon=horizon, num_layers=n_enc)
                else:
                    model = cls(n_input=n_input, horizon=horizon)
                model.load_state_dict(state)
                model.eval()
                models[(mname, cl, h)] = model

    return metadata, scalers, df, models


def resolve_client(client_id, cmap):
    """Resolve client ID to canonical form."""
    if client_id in cmap:
        return client_id
    if client_id.isdigit():
        padded = f"MT_{int(client_id):03d}"
        if padded in cmap:
            return padded
    if not client_id.startswith("MT_"):
        for key in cmap:
            if client_id.lower() in key.lower():
                return key
    return None


def run_forecast(metadata, scalers, df, models, client_id, horizon, model_name=None):
    """Run inference. If model_name is None, use best model for cluster+horizon."""
    cmap = metadata["client_cluster_map"]
    resolved = resolve_client(client_id, cmap)
    if resolved is None:
        return {"error": f"Client '{client_id}' not found. Try IDs like MT_042 or just 42."}

    cluster = cmap[resolved]

    # Determine which model to use
    if model_name is None:
        best_map = metadata.get("best_models", {})
        model_name = best_map.get(f"{cluster}_{horizon}", "patchtst")

    model_key = (model_name, cluster, horizon)
    if model_key not in models:
        # Fallback to any available model
        for mname in ["patchtst", "lstm", "tcn", "transformer"]:
            if (mname, cluster, horizon) in models:
                model_name = mname
                model_key = (mname, cluster, horizon)
                break
        else:
            return {"error": f"No model for cluster {cluster}, horizon {horizon}h."}

    model = models[model_key]
    scaler = scalers.get(resolved)
    if scaler is None:
        return {"error": f"No scaler found for client {resolved}."}

    if df is None or resolved not in df.columns:
        return {"error": "Raw data not available for live inference."}

    # Get latest 168h of data
    raw_series = df[resolved].values[-LOOKBACK:].reshape(-1, 1).astype(np.float32)
    latest_index = df.index[-LOOKBACK:]
    normalized = scaler.transform(raw_series).flatten()
    time_feats = build_time_features(latest_index)

    x_load = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(2)
    x_time = torch.FloatTensor(time_feats).unsqueeze(0)
    x = torch.cat([x_load, x_time], dim=2)

    with torch.no_grad():
        pred_normalized = model(x).numpy().flatten()
    pred_orig = pred_normalized * scaler.scale_[0] + scaler.mean_[0]

    last_ts = df.index[-1]
    forecast_times = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=horizon, freq="h")

    forecast_values = []
    for i, (ts, val) in enumerate(zip(forecast_times, pred_orig)):
        forecast_values.append({
            "hour": i + 1,
            "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
            "predicted_kWh": round(float(val), 2),
        })

    # Get model accuracy from metadata
    all_results = metadata.get("all_results", {})
    perf = all_results.get(model_name, {}).get(str(cluster), {}).get(str(horizon), {})

    # Also get all model MAPEs for comparison
    model_comparison = {}
    for mname in metadata.get("model_names", ["patchtst"]):
        m_perf = all_results.get(mname, {}).get(str(cluster), {}).get(str(horizon), {})
        if m_perf:
            model_comparison[mname.upper()] = m_perf.get("mape", "N/A")

    cl_clients = metadata["cluster_clients"].get(str(cluster), [])

    return {
        "client_id": resolved,
        "cluster_id": cluster,
        "cluster_size": len(cl_clients),
        "horizon_hours": horizon,
        "model_used": model_name.upper(),
        "model_mape": perf.get("mape", "N/A"),
        "all_model_mapes": model_comparison,
        "forecast": forecast_values,
        "last_known_load": round(float(raw_series[-1][0]), 2),
        "last_timestamp": df.index[-1].strftime("%Y-%m-%d %H:%M"),
    }


# =====================================================================
# 3. OPENAI TOOLS
# =====================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_client",
            "description": "Look up an electricity client by ID to find their cluster, consumption profile, and available models.",
            "parameters": {
                "type": "object",
                "properties": {
                    "client_id": {"type": "string", "description": "Client ID (e.g. 'MT_042', '42', '150')"}
                },
                "required": ["client_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_forecast",
            "description": "Run a deep learning model to generate an electricity load forecast for a specific client. Uses the best-performing model (LSTM, TCN, Transformer, or PatchTST) automatically, or you can specify a model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "client_id": {"type": "string", "description": "Client ID"},
                    "horizon": {"type": "integer", "description": "Hours ahead (1, 6, or 24)", "enum": [1, 6, 24]},
                    "model_name": {"type": "string", "description": "Optional model choice", "enum": ["lstm", "tcn", "transformer", "patchtst"]}
                },
                "required": ["client_id", "horizon"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_clients",
            "description": "List available clients grouped by cluster with model accuracy stats for all 4 architectures.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_cluster_info",
            "description": "Get performance metrics for all 4 models (LSTM, TCN, Transformer, PatchTST) on a specific cluster.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cluster_id": {"type": "integer", "description": "Cluster ID (0, 1, or 2)"}
                },
                "required": ["cluster_id"]
            }
        }
    }
]

SYSTEM_PROMPT = """You are an AI Electricity Load Forecasting Agent for a Portuguese utility company.

You have DIRECT ACCESS to 4 deep learning model architectures trained on historical electricity data for 314 clients:
- LSTM (Long Short-Term Memory)
- TCN (Temporal Convolutional Network)
- Transformer (Encoder-only)
- PatchTST (Patch Time Series Transformer)

When a user asks for a forecast, you ACTUALLY RUN the best-performing model and return real predicted values.

ARCHITECTURE:
- 314 clients clustered into 3 groups by daily load profile (K-Means on normalized daily profiles)
- Separate model trained per architecture × cluster × horizon = 36 models total
- Horizons: 1h (real-time dispatch), 6h (intra-day), 24h (day-ahead market)
- Input: 168h lookback window (1 week) + 8 cyclical time features
- Metric: MAPE (Mean Absolute Percentage Error)
- The system automatically selects the best-performing model per cluster+horizon

WORKFLOW:
1. User enters a client ID or name
2. Call lookup_client to find their cluster and see all model accuracies
3. Call run_forecast to EXECUTE the model and get real predictions
4. Present forecast with business context and model comparison

ALWAYS:
- Show the actual predicted kWh values in a table
- State which model was used and its MAPE
- Show how all 4 models compare on accuracy for that cluster
- If showing 24h forecast, highlight peak and off-peak hours
- Suggest business actions"""


def execute_tool(tool_name, args, metadata, scalers, df, models):
    if tool_name == "lookup_client":
        cid = args["client_id"]
        cmap = metadata["client_cluster_map"]
        resolved = resolve_client(cid, cmap)
        if resolved is None:
            return json.dumps({"error": f"Client '{cid}' not found.", "sample_ids": list(cmap.keys())[:5]})
        cluster = cmap[resolved]
        cl_clients = metadata["cluster_clients"].get(str(cluster), [])
        all_results = metadata.get("all_results", {})
        model_perf = {}
        for mname in metadata.get("model_names", []):
            m_res = all_results.get(mname, {}).get(str(cluster), {})
            model_perf[mname.upper()] = {
                f"H{h}h_mape": m_res.get(str(h), {}).get("mape", "N/A") for h in [1, 6, 24]
            }
        best_map = metadata.get("best_models", {})
        best_per_h = {f"H{h}h": best_map.get(f"{cluster}_{h}", "?").upper() for h in [1, 6, 24]}
        return json.dumps({
            "client_id": resolved, "cluster": cluster, "cluster_size": len(cl_clients),
            "horizons_available": [1, 6, 24], "model_performance": model_perf,
            "best_model_per_horizon": best_per_h,
        })

    elif tool_name == "run_forecast":
        model_name = args.get("model_name")
        result = run_forecast(metadata, scalers, df, models, args["client_id"], args["horizon"], model_name)
        return json.dumps(result, default=str)

    elif tool_name == "list_clients":
        all_results = metadata.get("all_results", {})
        out = {}
        for cl_id, cl_clients in metadata["cluster_clients"].items():
            cluster_models = {}
            for mname in metadata.get("model_names", []):
                m_res = all_results.get(mname, {}).get(cl_id, {})
                cluster_models[mname.upper()] = {
                    "H1_mape": m_res.get("1", {}).get("mape", "N/A"),
                    "H24_mape": m_res.get("24", {}).get("mape", "N/A"),
                }
            out[f"Cluster {cl_id}"] = {
                "count": len(cl_clients), "clients": cl_clients[:10],
                "models": cluster_models,
            }
        return json.dumps(out)

    elif tool_name == "get_cluster_info":
        cl_id = str(args["cluster_id"])
        all_results = metadata.get("all_results", {})
        perf = {}
        for mname in metadata.get("model_names", []):
            m_res = all_results.get(mname, {}).get(cl_id, {})
            perf[mname.upper()] = {f"H={h}h": m_res.get(str(h), {}) for h in [1, 6, 24]}
        cl_clients = metadata["cluster_clients"].get(cl_id, [])
        best_map = metadata.get("best_models", {})
        best_per_h = {f"H{h}h": best_map.get(f"{int(cl_id)}_{h}", "?").upper() for h in [1, 6, 24]}
        return json.dumps({
            "cluster": cl_id, "n_clients": len(cl_clients),
            "model_performance": perf, "best_model_per_horizon": best_per_h,
        })

    return json.dumps({"error": "Unknown tool"})


# =====================================================================
# 4. STREAMLIT APP
# =====================================================================

def main():
    metadata, scalers, df, models = load_all_resources()

    st.title("⚡ Electricity Load Forecasting Agent")
    st.caption("4 DL models (LSTM, TCN, Transformer, PatchTST) — enter a client ID for real-time forecasts")

    if metadata is None:
        st.error("Model metadata not found. Run `python3 deliverable_2.py` first.")
        return

    # Sidebar
    with st.sidebar:
        st.header("📊 Model Dashboard")
        all_results = metadata.get("all_results", {})
        model_names = metadata.get("model_names", ["patchtst"])
        best_map = metadata.get("best_models", {})

        # Overall MAPE per model
        st.subheader("Weighted MAPE")
        for mname in model_names:
            m_res = all_results.get(mname, {})
            for h in [1, 6, 24]:
                tw = sum(float(m_res.get(str(cl), {}).get(str(h), {}).get("mape", 0)) *
                         m_res.get(str(cl), {}).get(str(h), {}).get("n_clients", 0)
                         for cl in range(3))
                tn = sum(m_res.get(str(cl), {}).get(str(h), {}).get("n_clients", 0) for cl in range(3))
                if tn > 0:
                    st.markdown(f"**{mname.upper()}** H={h}h: `{tw/tn:.1f}%`")

        st.divider()
        st.subheader("Clusters")
        for cl_id, cl_clients in metadata["cluster_clients"].items():
            best_h24 = best_map.get(f"{cl_id}_24", "?").upper()
            st.markdown(f"**Cluster {cl_id}** — {len(cl_clients)} clients  \n"
                        f"Best H24: {best_h24}")

        st.divider()
        st.subheader("Try asking:")
        st.markdown("""
- `Forecast for client MT_042`
- `What will client 150 consume in 24h?`
- `Compare models for cluster 1`
- `Run LSTM forecast for client 200`
        """)
        st.divider()
        n_loaded = len(models)
        st.caption(f"🔧 {n_loaded} models loaded · 314 clients · 3 clusters")

    # API Key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        api_key = st.text_input("🔑 OpenAI API Key:", type="password")
        if not api_key:
            st.info("Enter your OpenAI API key to start.")
            return

    client = OpenAI(api_key=api_key)

    # Chat state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant", avatar="⚡"):
                st.markdown(msg["content"])
        elif msg["role"] == "tool_trace":
            with st.expander(f"🔧 Agent: {msg['tool_name']}", expanded=False):
                st.json(json.loads(msg["content"]))

    if prompt := st.chat_input("Enter a client ID or ask about forecasts..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in st.session_state.messages:
            if msg["role"] in ("user", "assistant"):
                api_messages.append({"role": msg["role"], "content": msg["content"]})

        with st.chat_message("assistant", avatar="⚡"):
            status = st.status("🤖 Agent working...", expanded=True)
            final_text = ""

            for iteration in range(6):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=api_messages,
                    tools=TOOLS,
                    tool_choice="auto",
                )
                msg = response.choices[0].message
                api_messages.append(msg)

                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        fn_name = tc.function.name
                        fn_args = json.loads(tc.function.arguments)
                        status.write(f"🔧 **Calling:** `{fn_name}({json.dumps(fn_args)})`")

                        result = execute_tool(fn_name, fn_args, metadata, scalers, df, models)
                        result_obj = json.loads(result)

                        if "forecast" in result_obj:
                            status.write(f"✅ {result_obj['model_used']} returned "
                                         f"{len(result_obj['forecast'])} forecast values")
                        elif "error" in result_obj:
                            status.write(f"❌ {result_obj['error']}")
                        else:
                            status.write(f"✅ Got result for `{fn_name}`")

                        api_messages.append({
                            "role": "tool", "tool_call_id": tc.id, "content": result,
                        })
                        st.session_state.messages.append({
                            "role": "tool_trace", "tool_name": fn_name, "content": result,
                        })
                else:
                    final_text = msg.content
                    status.update(label="✅ Agent complete", state="complete", expanded=False)
                    break

            st.markdown(final_text)
            st.session_state.messages.append({"role": "assistant", "content": final_text})


if __name__ == "__main__":
    main()
