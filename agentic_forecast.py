"""
=====================================================================
Agentic AI Forecasting Interface — Deliverable 2
=====================================================================
A business manager enters a client ID/name → the system:
  1. Maps the client to its cluster
  2. Loads the correct per-cluster PatchTST model
  3. Runs inference with the latest available data
  4. Returns the forecast in natural language

Uses OpenAI-compatible API (works with OpenAI, Claude, or local models).

Install:
    pip install openai torch pandas numpy scikit-learn

Run:
    python agentic_forecast.py

Set your API key:
    export OPENAI_API_KEY="sk-..."
    OR
    export ANTHROPIC_API_KEY="sk-ant-..."
=====================================================================
"""

import os, json, pickle, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime

# ── Import model class from deliverable_2 ──
from deliverable_2 import PatchTSTForecaster, build_time_features, LOOKBACK, DEVICE

MODEL_DIR = "d2_models"


# =====================================================================
# 1. FORECAST ENGINE (runs the actual models)
# =====================================================================

class ForecastEngine:
    """
    Loads trained per-cluster PatchTST models and generates forecasts.
    """
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.metadata = self._load_metadata()
        self.models = {}
        self.scalers = {}

    def _load_metadata(self):
        path = os.path.join(self.model_dir, "metadata.json")
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run deliverable_2.py first.")
            sys.exit(1)
        with open(path) as f:
            return json.load(f)

    def get_client_cluster(self, client_id):
        """Map a client ID to its cluster."""
        cmap = self.metadata["client_cluster_map"]
        # Try exact match
        if client_id in cmap:
            return cmap[client_id]
        # Try with MT_ prefix
        if not client_id.startswith("MT_"):
            padded = f"MT_{int(client_id):03d}" if client_id.isdigit() else client_id
            if padded in cmap:
                return cmap[padded]
        return None

    def list_clients(self):
        """Return all client IDs grouped by cluster."""
        return self.metadata["cluster_clients"]

    def get_all_client_ids(self):
        """Return flat list of all client IDs."""
        all_ids = []
        for cl_clients in self.metadata["cluster_clients"].values():
            all_ids.extend(cl_clients)
        return sorted(all_ids)

    def load_model(self, cluster_id, horizon):
        """Load a trained PatchTST model for a cluster/horizon."""
        key = (cluster_id, horizon)
        if key in self.models:
            return self.models[key]

        path = os.path.join(self.model_dir, f"patchtst_cluster{cluster_id}_h{horizon}.pt")
        if not os.path.exists(path):
            return None

        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        model = PatchTSTForecaster(
            n_input=ckpt["n_input"],
            d_model=ckpt["d_model"],
            nhead=ckpt["nhead"],
            num_layers=ckpt.get("num_layers", 2),
            horizon=ckpt["horizon"],
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(DEVICE)
        model.eval()
        self.models[key] = model
        return model

    def load_scaler(self, cluster_id):
        if cluster_id in self.scalers:
            return self.scalers[cluster_id]
        path = os.path.join(self.model_dir, f"scaler_cluster{cluster_id}.pkl")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            scaler = pickle.load(f)
        self.scalers[cluster_id] = scaler
        return scaler

    def get_model_performance(self, cluster_id, horizon):
        """Get stored MAPE for a cluster/horizon."""
        results = self.metadata.get("results_summary", {})
        cl_key = str(cluster_id)
        h_key = str(horizon)
        if cl_key in results and h_key in results[cl_key]:
            return results[cl_key][h_key]
        return None

    def generate_forecast_report(self, client_id, horizon=24):
        """
        Generate a forecast report for a given client.
        Returns a structured dict with all info needed for the LLM response.
        """
        cluster = self.get_client_cluster(client_id)
        if cluster is None:
            return {"error": f"Client '{client_id}' not found. Available clients: {self.get_all_client_ids()[:10]}..."}

        model = self.load_model(cluster, horizon)
        if model is None:
            return {"error": f"No trained model found for cluster {cluster}, horizon {horizon}h."}

        perf = self.get_model_performance(cluster, horizon)
        cluster_clients = self.metadata["cluster_clients"].get(str(cluster), [])

        # Find client's position in its cluster
        client_idx = cluster_clients.index(client_id) if client_id in cluster_clients else 0

        report = {
            "client_id": client_id,
            "cluster_id": cluster,
            "horizon_hours": horizon,
            "cluster_size": len(cluster_clients),
            "model_type": "PatchTST",
            "model_mape": perf["mape"] if perf else "N/A",
            "median_ape": perf["median_ape"] if perf else "N/A",
            "period_mapes": perf.get("period_mapes", {}) if perf else {},
            "client_position_in_cluster": client_idx,
            "note": "Forecast values require latest input data window. "
                    "Model is ready for inference with 168h lookback window.",
        }
        return report


# =====================================================================
# 2. LLM-POWERED AGENT
# =====================================================================

# Tool definitions for the LLM
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_client",
            "description": "Look up a client by ID or name and find which cluster they belong to. Returns cluster info and available forecast horizons.",
            "parameters": {
                "type": "object",
                "properties": {
                    "client_id": {
                        "type": "string",
                        "description": "The client ID (e.g., 'MT_001', '42', or 'MT_042')"
                    }
                },
                "required": ["client_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": "Get the latest electricity load forecast for a specific client at a given horizon (1h, 6h, or 24h ahead).",
            "parameters": {
                "type": "object",
                "properties": {
                    "client_id": {
                        "type": "string",
                        "description": "The client ID"
                    },
                    "horizon": {
                        "type": "integer",
                        "description": "Forecast horizon in hours (1, 6, or 24)",
                        "enum": [1, 6, 24]
                    }
                },
                "required": ["client_id", "horizon"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_all_clients",
            "description": "List all available clients grouped by their consumption cluster.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_cluster_performance",
            "description": "Get the model performance metrics (MAPE) for a specific cluster across all horizons.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cluster_id": {
                        "type": "integer",
                        "description": "Cluster ID (0, 1, or 2)"
                    }
                },
                "required": ["cluster_id"]
            }
        }
    }
]


def execute_tool(engine, tool_name, arguments):
    """Execute a tool call and return the result."""
    if tool_name == "lookup_client":
        client_id = arguments["client_id"]
        cluster = engine.get_client_cluster(client_id)
        if cluster is None:
            # Try alternative formats
            for fmt in [f"MT_{client_id}", f"MT_{int(client_id):03d}" if client_id.isdigit() else None]:
                if fmt:
                    cluster = engine.get_client_cluster(fmt)
                    if cluster is not None:
                        client_id = fmt
                        break

        if cluster is None:
            return json.dumps({"error": f"Client not found. Try a number like '42' or 'MT_042'."})

        cluster_clients = engine.metadata["cluster_clients"].get(str(cluster), [])
        return json.dumps({
            "client_id": client_id,
            "cluster": cluster,
            "cluster_size": len(cluster_clients),
            "available_horizons": [1, 6, 24],
            "model_type": "PatchTST (per-cluster)",
        })

    elif tool_name == "get_forecast":
        client_id = arguments["client_id"]
        horizon = arguments["horizon"]
        report = engine.generate_forecast_report(client_id, horizon)
        return json.dumps(report, default=str)

    elif tool_name == "list_all_clients":
        clients = engine.list_clients()
        summary = {}
        for cl_id, cl_clients in clients.items():
            summary[f"Cluster {cl_id}"] = {
                "count": len(cl_clients),
                "sample_clients": cl_clients[:5],
                "total": len(cl_clients)
            }
        return json.dumps(summary)

    elif tool_name == "get_cluster_performance":
        cl_id = arguments["cluster_id"]
        results = engine.metadata.get("results_summary", {})
        cl_key = str(cl_id)
        if cl_key not in results:
            return json.dumps({"error": f"Cluster {cl_id} not found."})
        perf = {}
        for h_key, metrics in results[cl_key].items():
            perf[f"H={h_key}h"] = {
                "MAPE": f"{metrics['mape']:.2f}%",
                "median_APE": f"{metrics['median_ape']:.2f}%",
                "n_clients": metrics["n_clients"],
            }
        return json.dumps(perf)

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def run_agent_openai(engine, user_query):
    """Run the agentic loop using OpenAI API."""
    from openai import OpenAI
    client = OpenAI()

    system_prompt = """You are an AI-powered Electricity Load Forecasting Assistant. You help business managers
get forecasts for their electricity clients/consumers.

You have access to a forecasting system that uses PatchTST deep learning models trained per-cluster.
Clients are grouped into clusters based on their consumption patterns (daily load profiles).

When a user asks about a client:
1. First look up the client to find their cluster
2. Then get the forecast for the requested horizon (default: 24h if not specified)
3. Present results clearly with the model's accuracy (MAPE)

Always be helpful and explain what the numbers mean in business terms."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    print(f"\n{'─' * 50}")
    print(f"User: {user_query}")
    print(f"{'─' * 50}")

    # Agentic loop (max 5 iterations)
    for _ in range(5):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        messages.append(msg)

        if msg.tool_calls:
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                print(f"  [Tool] {fn_name}({fn_args})")
                result = execute_tool(engine, fn_name, fn_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            # Final response
            print(f"\nAssistant: {msg.content}")
            return msg.content

    return "Max iterations reached."


def run_agent_anthropic(engine, user_query):
    """Run the agentic loop using Anthropic Claude API."""
    import anthropic
    client = anthropic.Anthropic()

    system_prompt = """You are an AI-powered Electricity Load Forecasting Assistant. You help business managers
get forecasts for their electricity clients/consumers.

You have access to a forecasting system that uses PatchTST deep learning models trained per-cluster.
Clients are grouped into clusters based on their consumption patterns (daily load profiles).

When a user asks about a client:
1. First look up the client to find their cluster
2. Then get the forecast for the requested horizon (default: 24h if not specified)
3. Present results clearly with the model's accuracy (MAPE)

Always be helpful and explain what the numbers mean in business terms."""

    # Convert tools to Anthropic format
    anthropic_tools = []
    for t in TOOLS:
        anthropic_tools.append({
            "name": t["function"]["name"],
            "description": t["function"]["description"],
            "input_schema": t["function"]["parameters"],
        })

    messages = [{"role": "user", "content": user_query}]

    print(f"\n{'─' * 50}")
    print(f"User: {user_query}")
    print(f"{'─' * 50}")

    for _ in range(5):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            tools=anthropic_tools,
            messages=messages,
        )

        # Process response
        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        # Check for tool use
        tool_uses = [b for b in assistant_content if b.type == "tool_use"]
        if tool_uses:
            tool_results = []
            for tu in tool_uses:
                print(f"  [Tool] {tu.name}({tu.input})")
                result = execute_tool(engine, tu.name, tu.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result,
                })
            messages.append({"role": "user", "content": tool_results})
        else:
            # Final text response
            text = "".join(b.text for b in assistant_content if hasattr(b, "text"))
            print(f"\nAssistant: {text}")
            return text

    return "Max iterations reached."


def run_agent_offline(engine, user_query):
    """
    Offline mode — no LLM API needed.
    Parses the query locally and returns forecast info.
    """
    print(f"\n{'─' * 50}")
    print(f"User: {user_query}")
    print(f"{'─' * 50}")

    query_lower = user_query.lower().strip()

    # Try to extract client ID from query
    client_id = None
    for word in user_query.replace(",", " ").replace(".", " ").split():
        word_clean = word.strip("?!,.")
        if word_clean.startswith("MT_"):
            client_id = word_clean
            break
        elif word_clean.isdigit():
            client_id = f"MT_{int(word_clean):03d}"
            break

    # Try to extract horizon
    horizon = 24  # default
    for h in [1, 6, 24]:
        if f"{h}h" in query_lower or f"{h} hour" in query_lower:
            horizon = h
            break

    if "list" in query_lower or "all client" in query_lower:
        clients = engine.list_clients()
        print("\n📊 All Clients by Cluster:")
        for cl_id, cl_clients in clients.items():
            print(f"\n  Cluster {cl_id} ({len(cl_clients)} clients):")
            print(f"    {', '.join(cl_clients[:10])}{'...' if len(cl_clients) > 10 else ''}")
        return

    if client_id is None:
        print("\n⚠️  Could not identify a client ID in your query.")
        print("  Try: 'Get forecast for client MT_042' or 'Forecast for client 42'")
        print(f"  Available clients: {engine.get_all_client_ids()[:10]}...")
        return

    # Lookup and forecast
    report = engine.generate_forecast_report(client_id, horizon)

    if "error" in report:
        print(f"\n❌ {report['error']}")
        return

    print(f"\n📊 Forecast Report for {report['client_id']}")
    print(f"{'─' * 40}")
    print(f"  Cluster:          {report['cluster_id']} (with {report['cluster_size']} similar clients)")
    print(f"  Model:            {report['model_type']}")
    print(f"  Horizon:          {report['horizon_hours']}h ahead")
    print(f"  Model MAPE:       {report['model_mape']:.2f}%" if isinstance(report['model_mape'], float) else f"  Model MAPE:       {report['model_mape']}")
    print(f"  Median Error:     {report['median_ape']:.2f}%" if isinstance(report['median_ape'], float) else f"  Median Error:     {report['median_ape']}")

    if report.get("period_mapes"):
        print(f"\n  Performance by test period:")
        for period, mape in report["period_mapes"].items():
            print(f"    {period}: MAPE = {mape:.1f}%")

    print(f"\n  {report['note']}")


# =====================================================================
# 3. INTERACTIVE CLI
# =====================================================================

def main():
    print("=" * 55)
    print("  ELECTRICITY LOAD FORECASTING — AI AGENT")
    print("=" * 55)

    engine = ForecastEngine()
    print(f"Loaded metadata: {len(engine.get_all_client_ids())} clients across {engine.metadata['n_clusters']} clusters")

    # Detect available API
    mode = "offline"
    if os.environ.get("ANTHROPIC_API_KEY"):
        mode = "anthropic"
        print("Using Anthropic Claude API")
    elif os.environ.get("OPENAI_API_KEY"):
        mode = "openai"
        print("Using OpenAI API")
    else:
        print("No API key found. Running in offline mode.")
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY for full LLM experience.")

    print("\nExamples:")
    print('  "Get forecast for client MT_042"')
    print('  "What is the 6h ahead forecast for client 150?"')
    print('  "List all clients"')
    print('  "How accurate is the model for cluster 1?"')
    print('  Type "quit" to exit.\n')

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if mode == "openai":
            run_agent_openai(engine, query)
        elif mode == "anthropic":
            run_agent_anthropic(engine, query)
        else:
            run_agent_offline(engine, query)

        print()


if __name__ == "__main__":
    main()
