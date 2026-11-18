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
    subgraph ING["DATA INGESTION"]
        A["Solana Network"]
        A --> B["Pre-Finality Transaction Feed<br/>(Processed Transactions)"]
        B --> C["Ingestion Pipeline<br/>(Helius LaserStream / gRPC)"]
    end

    subgraph FAST["STAGE 1 — PRE-FINALITY SCREENING"]
        C --> D["Transaction Sender Extraction"]
        D --> E["Address Anomaly Lookup Table"]
        E --> F{"Historical Anomaly Rate<br/>Above Threshold?"}
        F -->|Yes| G["Preliminary Risk Flag"]
        F -->|No| H["No Preliminary Flag"]
        G --> I["Preliminary Screening Results"]
        H --> I
    end

    subgraph FINAL["FINALITY & DATA ENRICHMENT"]
        A --> J["Transaction Finalization"]
        J --> K["Finalized Transaction Data"]
        K --> L["Metadata Retrieval & Enrichment"]
        L --> M["Feature Extraction<br/>(6 Execution Features)"]
    end

    subgraph INFERENCE["STAGE 2 — FULL INFERENCE"]
        M --> N["Weighted Isolation Forest<br/>(C++ / ONNX Runtime)"]
        N --> O["Anomaly Score"]
        O --> P{"Score Above<br/>Calibrated Threshold?"}
        P -->|Yes| Q["Statistically Anomalous"]
        P -->|No| R["Within Baseline"]
    end

    subgraph BACKGROUND["BACKGROUND PROCESSING — OFF HOT PATH"]
        Q --> S["Historical Transaction Statistics"]
        R --> S
        S --> T["Update Address Anomaly Rates"]
        T --> E
        S --> U["Moving MAD / Robust<br/>Statistical Baseline Updates"]
        U --> V["Update Heuristic Feature Weights"]
        V --> N
        N -.-> W["SHAP Feature Attribution"]
        W --> X["Audit Logs & Model Explanations"]
    end

    style ING fill:#E8F1FF,stroke:#2563EB,color:#172554
    style FAST fill:#FFF7E6,stroke:#D97706,color:#78350F
    style FINAL fill:#F0EAFE,stroke:#7C3AED,color:#2E1065
    style INFERENCE fill:#E8F8EF,stroke:#15803D,color:#14532D
    style BACKGROUND fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A

    style G fill:#FEE2E2,stroke:#DC2626,color:#7F1D1D
    style Q fill:#FEE2E2,stroke:#DC2626,color:#7F1D1D
    style R fill:#DCFCE7,stroke:#16A34A,color:#14532D
    style X fill:#FFFFFF,stroke:#2563EB,color:#1E3A8A
```
