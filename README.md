# BKK Thesis work


## Structure

```
bkk_thesis/
├     config/paths.py         
├     models/
│   ├     gat_lstm.py         
│   └     hypergraph_lstm.py   
├     utils/
│   ├     data.py              
│   └     synthetic_scenarios.py
├     train.py                
└     evaluate.py             
```


### Google Colab
```python
!git clone https://github.com/username/test.git
%cd test
!pip install -r requirements.txt

from google.colab import drive
drive.mount('/content/drive')

# Training
!python train.py --model gat
!python train.py --model hypergraph

# Evaluation
!python evaluate.py --model all
```

### Local VS Code
```bash
git clone https://github.com/USERNAME/bkk_thesis.git
cd bkk_thesis
pip install -r requirements.txt

python train.py --model gat --epochs 100
python evaluate.py --model all
```

---

## Datas from Google Drive 


---

## Architectures

**GAT+LSTM**: Zone neighbouring graph → multi-head attention → LSTM → ΔOD  
**Hypergraph+LSTM**: BKK lines like hyperedges → HGNN → LSTM → ΔOD  

Cource: Feng et al. (2019) — Hypergraph Neural Networks
