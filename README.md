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
flowchart TD
    subgraph Ingestion ["DATA INGESTION"]
        A["Solana Network / Raydium AMM"] -->|Helius RPC WebSocket / gRPC| B["DPDK Kernel Bypass / NIC"]
        B --> C["Ingestion Workers<br/>(Pinned Threads)"]
    end

    subgraph Queue ["CONCURRENCY & BUFFERING"]
        C --> D["Pre-allocated Lock-Free<br/>Ring Buffers"]
    end

    subgraph Inference ["INFERENCE PIPELINE - C++ / ONNX"]
        D --> E["Metadata Retrieval<br/>& Enrichment"]
        E --> F["Feature Extraction<br/>(6 Execution Features)"]
        F --> G["Weighted Isolation Forest<br/>(ONNX Runtime)"]
    end

    subgraph Output ["SCORING & AUDIT"]
        G --> H["Anomaly Score<br/>vs. Threshold"]
        H -->|Above Threshold| I["CLASSIFIED AS ANOMALOUS"]
        H -->|Below Threshold| J["CLASSIFIED AS NORMAL"]
        G -.->|Off-Hot-Path| K["SHAP Value Generation<br/>& Storage<br/>(Audit & Interpretability)"]
    end

    style Ingestion fill:#E8F1FF,stroke:#2563EB,stroke-width:2px,color:#172554
    style Queue fill:#F0EAFE,stroke:#7C3AED,stroke-width:2px,color:#2E1065
    style Inference fill:#E8F8EF,stroke:#15803D,stroke-width:2px,color:#14532D
    style Output fill:#FFF7E6,stroke:#D97706,stroke-width:2px,color:#78350F

    style A fill:#FFFFFF,stroke:#64748B,color:#111827
    style B fill:#FFFFFF,stroke:#64748B,color:#111827
    style C fill:#FFFFFF,stroke:#64748B,color:#111827
    style D fill:#FFFFFF,stroke:#64748B,color:#111827
    style E fill:#FFFFFF,stroke:#64748B,color:#111827
    style F fill:#FFFFFF,stroke:#64748B,color:#111827
    style G fill:#FFFFFF,stroke:#64748B,color:#111827
    style H fill:#FFFFFF,stroke:#64748B,color:#111827

    style I fill:#FEE2E2,stroke:#DC2626,stroke-width:2px,color:#7F1D1D
    style J fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#14532D
    style K fill:#DBEAFE,stroke:#2563EB,stroke-width:2px,color:#1E3A8A

    linkStyle default stroke:#64748B,stroke-width:1.5px
```
