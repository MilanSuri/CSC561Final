# Solana Pre-Finality Transaction Anomaly Detection

## Executive Summary:

Solana's high transaction throughput creates a need for efficient monitoring of anomalous transaction activity. This system could support several applications within the Solana ecosystem, including network monitoring, providing additional monitoring signals to validators, MEV research, and post-trade analysis.

This proposed algorithm would use a heuristic-guided Isolation Forest to identify trends that deviate from the normal baseline behavior defined by historical activity on the network. The current proof of concept uses six execution-derived features to calculate anomaly scores, while using statistical methods like modified Z-scores to calculate the weights for each heuristic. These weights influence the probability of each feature being selected during tree construction, allowing the model to incorporate feature-level statistical characteristics into its isolation process. The future expansion of this system would focus on building an inference-based evaluation framework to process transactions before block finality, meaning it has to operate within Alpenglow's approximately 150 ms finality target  (now on mainnet-beta).

## File Structure:

The `src` folder houses the code for the project
Inside `src` there are:
- `DataPipeline/`
  - `DataPipeline.py` — Gets real data
  - `SyntheticData.py` — Generates synthetic data using a CTGAN
- `MedianWeights.py` — Calculates the weights that will be applied
- `isoforest.py` — Creates the class structure for a custom isolation forest with heuristic guided splits
- `main.py` — Main pipeline for training the model + export to ONNX

To view dependencies use the `uv.lock`, `pyproject.toml`, and `python-version` files.

To view the architecture and the whitepaper of the project in more depth view: `MEV_Whitepaper(3).pdf`

## Architecture Diagram:
```mermaid
graph TD
    subgraph Key ["Legend / Key"]
        direction TB
        K1["1. Ingestion Layer"]
        K2["2. Concurrency & Buffering"]
        K3["3. Inference Pipeline"]
        K4["4. Scoring & Audit"]
    end

    subgraph Ingestion ["1. Data Ingestion Layer"]
        A[Solana Network / Raydium AMM] -->|Helius RPC WebSocket / gRPC| B[DPDK Kernel Bypass / NIC]
        B --> C["Ingestion Workers (Pinned Threads)"]
    end

    subgraph Queue ["2. Concurrency & Buffering"]
        C --> D[Pre-allocated Lock-Free Ring-Buffers]
    end

    subgraph Inference ["3. Inference Pipeline (C++ / ONNX Runtime)"]
        D --> E[Metadata Retrieval & Enrichment]
        E --> F["Feature Extraction <br/><i>(6 Execution Features)</i>"]
        F --> G["Weighted Isolation Forest <br/><i>(ONNX Runtime Execution)</i>"]
    end

    subgraph Output ["4. Scoring & Audit"]
        G --> H["Anomaly Score <br/>vs. Threshold"]
        H -->|Above Threshold| I[Classified as Anomalous]
        H -->|Below Threshold| J[Classified as Normal]
        
        %% Off-path audit
        G -.->|Off-Hot-Path| K["SHAP Value Generation & Storage <br/><i>(Audit & Interpretability)</i>"]
    end

    %% Styling
    style Key fill:#ffffff,stroke:#999,stroke-width:1px,stroke-dasharray: 3 3
    style Ingestion fill:#f9f,stroke:#333,stroke-width:2px
    style Queue fill:#bbf,stroke:#333,stroke-width:2px
    style Inference fill:#bfb,stroke:#333,stroke-width:2px
    style Output fill:#ff9,stroke:#333,stroke-width:2px
```


