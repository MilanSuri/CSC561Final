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
    subgraph Ingestion ["Data Ingestion"]
        A[Solana Network / Raydium AMM] -->|Helius RPC WebSocket / gRPC| B[DPDK Kernel Bypass / NIC]
        B --> C["Ingestion Workers (Pinned Threads)"]
    end

    subgraph Queue ["Concurrency & Buffering"]
        C --> D[Pre-allocated Lock-Free Ring-Buffers]
    end

    subgraph Inference ["Inference Pipeline (C++ / ONNX)"]
        D --> E[Metadata Retrieval & Enrichment]
        E --> F["Feature Extraction <br/><i>(6 Execution Features)</i>"]
        F --> G["Weighted Isolation Forest <br/><i>(ONNX Runtime Execution)</i>"]
    end

    subgraph Output ["Scoring & Audit"]
        G --> H["Anomaly Score <br/>vs. Threshold"]
        H -->|Above Threshold| I[Classified as Anomalous]
        H -->|Below Threshold| J[Classified as Normal]
        
        %% Off-path audit
        G -.->|Off-Hot-Path| K["SHAP Value Generation & Storage <br/><i>(Audit & Interpretability)</i>"]
    end

    %% Section Styling
    style Ingestion fill:#f9f,stroke:#333,stroke-width:2px
    style Queue fill:#bbf,stroke:#333,stroke-width:2px
    style Inference fill:#bfb,stroke:#333,stroke-width:2px
    style Output fill:#fff3cd,stroke:#333,stroke-width:2px

    %% Node Highlighting for Scoring & Audit
    style I fill:#ff7675,stroke:#333,stroke-width:2px,color:#fff
    style J fill:#55efc4,stroke:#333,stroke-width:2px,color:#000
    style K fill:#74b9ff,stroke:#333,stroke-width:2px,color:#000
```


