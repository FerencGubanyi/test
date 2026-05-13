# BKK Passenger Flow Redistribution Predictor

**MSc Thesis — Óbuda University, Neumann János Faculty of Informatics**  
**Supervisor:** Sipos Miklós László

Predicts zone-level passenger flow redistribution (ΔOD) in Budapest's public transit network following infrastructure changes (new stops, route extensions, closures). Compares two deep learning architectures trained on BKK EFM VISUM scenario OD matrix exports.

---

## Project Structure

```
test/
├── config/
│   └── paths.py                     # Colab / local path resolution
├── models/
│   ├── gat_lstm.py                  # GAT + LSTM architecture
│   └── hypergraph_lstm.py           # Hypergraph Neural Net + LSTM
├── utils/
│   ├── data.py                      # OD matrix parsing, GTFS zone features
│   ├── loss.py                      # Combined MSE + Huber loss
│   ├── metrics.py                   # MAE, RMSE, R², Spearman ρ, Top-K, Moran's I
│   ├── synthetic_scenarios.py       # Synthetic scenario generator (6 types × 30 = 180)
│   └── metr_la_loader.py            # METR-LA benchmark data loader
├── db/
│   ├── __init__.py
│   └── init_db.py                   # SQLite inference history
├── tests/
│   ├── conftest.py
│   ├── pytest.ini
│   ├── test_models.py
│   ├── test_data.py
│   ├── test_loss.py
│   ├── test_metrics.py
│   └── test_app.py
├── checkpoints/                     # Saved model weights (see Download below)
├── streamlit.py                     # Interactive inference frontend
├── train.py                         # BKK training entry point
├── evaluate.py                      # BKK evaluation entry point
├── benchmark_metr_la.py             # METR-LA architecture validation
├── BKK_Thesis_Demo.ipynb            # Full demo notebook (no Drive needed)
└── requirements.txt
```

---

## Architectures

**GAT+LSTM** — Zone adjacency graph → 8-head graph attention (×2) → bidirectional LSTM → ΔOD  
**Hypergraph+LSTM** — BKK transit routes as hyperedges → HGNN (Feng et al. 2019) → bidirectional LSTM → ΔOD

Both models predict the per-zone ΔOD vector given a base OD state and a scenario descriptor.

---

## Quick Start (Local)

```bash
git clone https://github.com/FerencGubanyi/test.git
cd test
pip install -r requirements.txt
```

> **Note:** `torch-geometric` may require a separate install matching your CUDA version.  
> See https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

### Training

```bash
# Full training (thesis Table 3 defaults: 500 epochs, patience 50, real_weight 5.0)
python train.py --model gat
python train.py --model hypergraph

# Quick smoke test
python train.py --model gat --epochs 5 --patience 5
```

### Evaluation

```bash
python evaluate.py --model all
```

### METR-LA Architecture Benchmark

Validates the architecture on public traffic data — runs without any BKK data.

```bash
python benchmark_metr_la.py
```

### Streamlit Inference App

```bash
streamlit run streamlit.py
```

Runs in **demo mode** if no checkpoint is present — generates a synthetic OD matrix so the UI is fully usable without trained weights. Every inference run is saved to `db/inference.db` (SQLite) and browsable in the **Inference History** section.

### Tests

```bash
pytest                          # all tests (273 passed, 19 skipped)
pytest -m "not slow"            # skip slow integration tests
pytest tests/test_models.py -v  # single module
```

---

## Google Colab (GPU Training — no Drive needed)

The demo notebook [`BKK_Thesis_Demo.ipynb`](BKK_Thesis_Demo.ipynb) downloads all required
data automatically from a public Google Drive folder.

```python
# Clone and install
!git clone https://github.com/FerencGubanyi/test.git
%cd test
!pip install -r requirements.txt

# Download BKK data + pretrained checkpoints (public Drive, no login needed)
!pip install gdown -q
import gdown, os

FILES = {
    'checkpoints/gat_lstm_best.pt':  '1vVZZEsq1qfBGYgoLeVJAc_jsV_cQlvZo',
    'checkpoints/hg_lstm_best.pt':   '1rH1rlvqbRE2eMuSz5J71QBYBJH2rEKmi',
    'data/visum/base_kozossegi_kozlekedes_matrix.xlsx':              '1lC_ZJQwHiECDLLxcAu_pAvUqaPlEUF0k',
    'data/visum/m2_meghosszabitas_kozossegi_kozlekedes_matrix.xlsx': '17DCGj4BZsFcVvZ1aBSvcvi-rxp2EQl2r',
    'data/visum/m1_kozossegi_kozlekedes_matrix.xlsx':                '1F2USo0C7_9-QONv0P4J3nWyKq9TSjJDx',
    'data/visum/m1_diff_KK.xlsx':                                    '1XR-MNAHs8H8zFIzC-KifhJ5Rlfsw50x7',
    'data/visum/35_autobusz_kozossegi_kozlekedes_matrix.xlsx':       '1BVH4yYp0D8WyrUcTtrjC80crv69gzzny',
    'data/visum/35_autobusz_diff_KK.xlsx':                           '1QZBc14HS23upO47pyzGuyRa7qVDHEfKk',
    'data/budapest_gtfs.zip':                                         '1NrA-o-mGj3wSYS0jjSLXOGb_-pV1n49J',
}

for dest, fid in FILES.items():
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        gdown.download(id=fid, output=dest, quiet=False)

# Train
!python train.py --model gat
!python train.py --model hypergraph

# Evaluate
!python evaluate.py --model all
```

---

## Data

OD matrix Excel exports from BKK EFM VISUM (not publicly available — provided by BKK).  
Expected format: zone IDs in row 0 starting at column 3; flow data from row 3 onwards.

| Scenario | Role | Loss weight |
|---|---|---|
| M2 metro extension | Training | 5× |
| Bus 35 Pesterzsébet | Training | 5× |
| M1 metro extension | **Validation (held out)** | — |
| S000144 | Excluded (unknown content) | — |
| Synthetic (180 total) | Training | 1× |

**Synthetic scenario types** (30 each): `bus_new`, `tram_extension`, `stop_closure`, `metro_extension`, `bus_freq`, `parallel`

---

## Results (Thesis Table 5 — M1 Extension Validation)

| Model | MAE | RMSE | R² | Moran's I |
|---|---|---|---|---|
| GAT+LSTM | 13.06 | 70.12 | −0.0066 | 0.1566 |
| Hypergraph+LSTM | 13.26 | 70.06 | −0.0048 | 0.1546 |

Low R² is expected given only 2 real VISUM training scenarios — see thesis Section 8.1 for discussion. The METR-LA benchmark (Section 6.7.3) confirms the architecture is structurally sound.

---

## References

- Feng et al. (2019) — *Hypergraph Neural Networks*, AAAI
- Wang et al. (2021) — *Dynamic Hypergraph Convolution for metro flow prediction*, IEEE T-ITS
- Li et al. (2018) — *Diffusion Convolutional Recurrent Neural Network*, ICLR
- Zhang et al. (2019) — *Spatial-Temporal Graph Attention Networks*, IEEE Access
- Chai & Draxler (2014) — *RMSE or MAE?*, Geoscientific Model Development
- Moran (1950) — *Notes on continuous stochastic phenomena*, Biometrika