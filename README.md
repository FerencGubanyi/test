# BKK Passenger Flow Redistribution Predictor

MSc thesis
Supervisor: Sipos Miklós László

Predicts zone-level passenger flow redistribution in Budapest's public transit network following infrastructure changes (new stops, route extensions, closures) using two deep learning architectures trained on BKK EFM VISUM scenario OD matrix exports.

---

## Project Structure
---
```
test/
├── config/
│   └── paths.py                  # Colab / local path switching
├── models/
│   ├── gat_lstm.py               # GAT + LSTM architecture
│   └── hypergraph_lstm.py        # Hypergraph Neural Net + LSTM
├── utils/
│   ├── data.py                   # OD matrix parsing, GTFS features
│   ├── loss.py                   # Combined MSE + Huber loss
│   ├── metrics.py                # MAE, RMSE, R², Spearman, Top-K, Moran's I
│   └── synthetic_scenarios.py    # Synthetic scenario generator (6 types, 180 scenarios)
├── db/
│   ├── __init__.py
│   └── init_db.py                # SQLite inference history layer
├── tests/
│   ├── conftest.py               # Shared fixtures
│   ├── pytest.ini
│   ├── test_models.py
│   ├── test_data.py
│   ├── test_scenarios.py
│   └── test_app.py
├── checkpoints/                  
│   ├── gat_lstm_best.pt
│   └── hypergraph_lstm_best.pt
├── data/                         # VISUM OD exports, GTFS, shapefiles (git-ignored)
├── streamlit.py                  # Inference frontend
├── train.py                      # Training entry point
├── evaluate.py                   # Evaluation entry point
└── requirements.txt
```
## Architectures

**GAT+LSTM** — Zone adjacency graph → multi-head graph attention → LSTM → ΔOD  
**Hypergraph+LSTM** — BKK transit lines as hyperedges → HGNN (Feng et al. 2019) → LSTM → ΔOD

---

## Local Setup

```bash
git clone https://github.com/FerencGubanyi/test.git
cd test
pip install -r requirements.txt
```

> **Note:** `torch-geometric` may need a separate install matching your CUDA version.  
> See https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

### Training

```bash
python train.py --model gat
python train.py --model hypergraph
# Defaults match thesis Table 3: --epochs 500 --patience 50 --real_weight 5.0
# For a quick smoke test: --epochs 5 --patience 5
```

### Evaluation

```bash
python evaluate.py --model all
```

### Streamlit App

```bash
streamlit run streamlit.py
```

The app runs in **demo mode** if no checkpoint is found — it generates a synthetic OD matrix so the UI is fully usable without trained weights.

Every inference run is automatically saved to `db/inference.db` (SQLite).  
The **Inference History** section at the bottom of the app lets you browse, inspect, and export past runs.

### Tests

```bash
# All tests
pytest

# Skip slow tests
pytest -m "not slow"

# Single module
pytest tests/test_models.py -v
```

---

## Google Colab (GPU Training)

```python
!git clone https://github.com/FerencGubanyi/test.git
%cd test
!pip install -r requirements.txt

from google.colab import drive
drive.mount('/content/drive')

!python train.py --model gat
!python train.py --model hypergraph
!python evaluate.py --model all
```

Checkpoints mentése Drive-ra:

```python
import shutil
shutil.copy("checkpoints/gat_lstm_best.pt", "/content/drive/MyDrive/bkk_checkpoints/")
shutil.copy("checkpoints/hypergraph_lstm_best.pt", "/content/drive/MyDrive/bkk_checkpoints/")
```

---

## Data

OD matrix Excel exports from BKK EFM VISUM.  
Expected format: zone IDs in row 0 starting at column 3, flow data from row 3.

Real VISUM scenarios:
- M2 metro extension *(training set)*
- Bus 35 Pesterzsébet *(training set)*
- M1 metro extension *(validation set — held out)*

Synthetic scenarios (180 total, 30 per type): `bus_new`, `tram_extension`, `stop_closure`,
`metro_extension`, `bus_freq`, `parallel` (see `utils/synthetic_scenarios.py`)

---

## References

- Feng et al. (2019) — *Hypergraph Neural Networks*
- Wang et al. (2021) — *Dynamic Hypergraph Convolution for metro flow prediction*
