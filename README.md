# BKK Utasforgalom Predikció


## Struktúra

```
bkk_thesis/
├── config/paths.py          # elérési utak (Colab + lokális)
├── models/
│   ├── gat_lstm.py          # GAT+LSTM architektúra
│   └── hypergraph_lstm.py   # Hypergraph+LSTM architektúra
├── utils/
│   ├── data.py              # adat betöltés, feature engineering
│   └── synthetic_scenarios.py  # szintetikus scenarió generátor
├── train.py                 # training futtatása
└── evaluate.py              # kiértékelés, összehasonlítás
```

---

## Gyors start

### Google Colab
```python
!git clone https://github.com/FELHASZNALONEV/bkk_thesis.git
%cd bkk_thesis
!pip install -r requirements.txt

from google.colab import drive
drive.mount('/content/drive')

# Training
!python train.py --model gat
!python train.py --model hypergraph

# Kiértékelés
!python evaluate.py --model all
```

### Lokális VS Code
```bash
git clone https://github.com/FELHASZNALONEV/bkk_thesis.git
cd bkk_thesis
pip install -r requirements.txt
# Módosítsd a config/paths.py BASE_DIR változóját

python train.py --model gat --epochs 100
python evaluate.py --model all
```

---

## Adatok

A nagy adatfájlok (xlsx, shp, pt) nem kerülnek a repóba — Google Drive-on tárolva.

| Scenarió | Típus | Forrás |
|----------|-------|--------|
| M2 meghosszabbítás | Metró | BKK (Gergely) |
| S000144 | Metró | VISUM export |
| M1 meghosszabbítás | Metró | BKK (Gergely) |
| 35-ös autóbusz | Busz | BKK (Gergely) |
| 9× szintetikus | Vegyes | Generált |

---

## Architektúrák

**GAT+LSTM**: zóna szomszédossági gráf → multi-head attention → LSTM → ΔOD  
**Hypergraph+LSTM**: BKK vonalak mint hyperedge-ek → HGNN → LSTM → ΔOD  

Irodalom: Feng et al. (2019) — Hypergraph Neural Networks
