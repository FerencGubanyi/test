---

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
python train.py --model gat --epochs 300 --lr 5e-4 --patience 30
python train.py --model hypergraph --epochs 300 --lr 5e-4 --patience 30
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

!python train.py --model gat --epochs 300 --patience 30
!python train.py --model hypergraph --epochs 300 --patience 30
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
- M1 metro extension
- M2 metro extension
- Bus 35 Pesterzsébet *(validation set)*

Synthetic scenarios (90 total): `bus_new`, `tram_extension`, `stop_closure`

---

## References

- Feng et al. (2019) — *Hypergraph Neural Networks*
- Wang et al. (2021) — *Dynamic Hypergraph Convolution for metro flow prediction*
