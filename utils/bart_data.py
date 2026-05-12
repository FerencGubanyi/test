"""
BART data adapter — reads extracted xlsx files directly from data/bart/.
"""
import os, io, zipfile, requests, warnings
import numpy as np, pandas as pd, torch

BART_GTFS_URL = "https://www.bart.gov/dev/schedules/google_transit.zip"
DATA_DIR      = "data/bart"

MONTHS_TO_FETCH = {
    "before_berryessa": "2020_02_average_weekday.xlsx",
    "after_berryessa":  "2020_09_average_weekday.xlsx",
    "before_antioch":   "2023_09_average_weekday.xlsx",
    "after_antioch":    "2024_01_average_weekday.xlsx",
    "baseline_2019_01": "2019_01_average_weekday.xlsx",
    "baseline_2019_06": "2019_06_average_weekday.xlsx",
    "baseline_2019_10": "2019_10_average_weekday.xlsx",
}

_NON_STATION_LABELS = {"exits", "entries", "total", "nan", "", "none"}

def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def _is_valid_station(s):
    sl = s.lower().strip()
    return (sl not in _NON_STATION_LABELS and 0 < len(s) <= 6
            and not s.startswith("#") and not (s.isdigit() and len(s) > 4))

def _load_od_excel(path: str) -> tuple:
    xl    = pd.ExcelFile(path)
    old_c = ["Avg Weekday OD", "Avg Weekday"]
    new_c = ["Average Weekday"]
    sheet = next((s for s in old_c + new_c if s in xl.sheet_names), xl.sheet_names[0])
    is_new = sheet in new_c and sheet not in old_c

    df = pd.read_excel(path, sheet_name=sheet, header=None)

    # Old format: station codes on row 1; New 2024 format: row 3
    for station_row in ([3, 1] if is_new else [1, 3]):
        raw = [str(s).strip() for s in df.iloc[station_row, 1:].tolist()]
        stations = [s for s in raw if _is_valid_station(s)]
        if len(stations) > 10:
            data_start = station_row + 1
            break

    N = len(stations)
    if N == 0:
        raise ValueError(f"No valid stations in {path}. Sample: {raw[:8]}")

    matrix_raw = df.iloc[data_start:data_start + N, 1:1 + N]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matrix = (matrix_raw.apply(pd.to_numeric, errors="coerce")
                             .fillna(0).values.astype(np.float32))

    if matrix.shape != (N, N):
        raise ValueError(f"Shape mismatch: {matrix.shape} != ({N},{N})")

    return stations, matrix

def _load_od_matrices(data_dir, verbose):
    od = {}
    for label, filename in MONTHS_TO_FETCH.items():
        path = next((p for p in [
            os.path.join(data_dir, filename),
            os.path.join(data_dir, "raw", filename),
        ] if os.path.exists(p)), None)
        if path is None:
            if verbose: print(f"  [skip] {label}: {filename} not found")
            continue
        try:
            stations, matrix = _load_od_excel(path)
            od[label] = (stations, matrix)
            if verbose: print(f"  [ok] {label}: {len(stations)} stations, total={matrix.sum():,.0f}")
        except Exception as e:
            if verbose: print(f"  [warning] {label}: {e}")
    return od

def _download_gtfs(data_dir, verbose):
    _ensure_dir(data_dir)
    zip_path = os.path.join(data_dir, "google_transit.zip")
    gtfs_dir = os.path.join(data_dir, "gtfs")
    if not os.path.exists(zip_path):
        if verbose: print("  [download] BART GTFS...")
        try:
            r = requests.get(BART_GTFS_URL, timeout=30)
            r.raise_for_status()
            open(zip_path, "wb").write(r.content)
        except Exception as e:
            if verbose: print(f"  [warning] GTFS failed: {e}")
            return gtfs_dir
    else:
        if verbose: print("  [cached] BART GTFS")
    if not os.path.exists(gtfs_dir) and os.path.exists(zip_path):
        if verbose: print("  [extract] BART GTFS")
        with zipfile.ZipFile(zip_path) as zf: zf.extractall(gtfs_dir)
    return gtfs_dir

def _align_matrices(b_st, b_mx, a_st, a_mx):
    all_st = sorted(set(b_st) | set(a_st))
    N = len(all_st)
    ib = {s:i for i,s in enumerate(b_st)}
    ia = {s:i for i,s in enumerate(a_st)}
    B = np.zeros((N,N), dtype=np.float32)
    A = np.zeros((N,N), dtype=np.float32)
    for i,si in enumerate(all_st):
        for j,sj in enumerate(all_st):
            if si in ib and sj in ib: B[i,j] = b_mx[ib[si],ib[sj]]
            if si in ia and sj in ia: A[i,j] = a_mx[ia[si],ia[sj]]
    return all_st, B, A

def _fuzzy_match(name, stations):
    nl = name.lower().strip()
    for s in stations:
        if s.lower().strip() == nl: return s
    for s in stations:
        if nl in s.lower() or s.lower() in nl: return s
    return name

def build_bart_graph(gtfs_dir, stations):
    def _fallback():
        N=len(stations); src,dst=[],[]
        for i in range(N):
            for j in range(i+1,N): src+=[i,j]; dst+=[j,i]
        return {"edge_index":torch.tensor([src,dst],dtype=torch.long),
                "edge_attr":torch.ones(len(src),2),"station_to_idx":{s:i for i,s in enumerate(stations)},
                "hyperedge_index":torch.zeros(2,0,dtype=torch.long),"n_hyperedges":0}
    try:
        stops=pd.read_csv(os.path.join(gtfs_dir,"stops.txt"))
        trips=pd.read_csv(os.path.join(gtfs_dir,"trips.txt"))
        st=pd.read_csv(os.path.join(gtfs_dir,"stop_times.txt"),usecols=["trip_id","stop_id","stop_sequence"])
    except Exception as e:
        print(f"  [warning] GTFS: {e}"); return _fallback()
    snm=dict(zip(stops.stop_id,stops.stop_name)); s2i={s:i for i,s in enumerate(stations)}
    t2r=dict(zip(trips.trip_id,trips.route_id)); edges={}; hes={}; ct=None; pi=None
    for _,row in st.sort_values(["trip_id","stop_sequence"]).iterrows():
        tid=row.trip_id; sn=snm.get(row.stop_id,""); m=_fuzzy_match(sn,stations)
        ni=s2i.get(m); rid=t2r.get(tid,"unknown")
        if ni is not None: hes.setdefault(rid,set()).add(ni)
        if tid!=ct: ct,pi=tid,ni; continue
        if pi is not None and ni is not None and pi!=ni:
            k=(min(pi,ni),max(pi,ni)); edges.setdefault(k,set()).add(rid)
        pi=ni
    if not edges: return _fallback()
    el=list(edges.keys())
    ei=torch.tensor([[e[0] for e in el]+[e[1] for e in el],[e[1] for e in el]+[e[0] for e in el]],dtype=torch.long)
    ea=torch.tensor([[len(edges[e]),5.0] for e in el]*2,dtype=torch.float)
    rids=sorted(hes.keys()); hs,hr=[],[]
    for ri,rid in enumerate(rids):
        for si in hes[rid]: hs.append(si); hr.append(ri)
    return {"edge_index":ei,"edge_attr":ea,"station_to_idx":s2i,
            "hyperedge_index":(torch.tensor([hs,hr],dtype=torch.long) if hs else torch.zeros(2,0,dtype=torch.long)),
            "n_hyperedges":len(rids)}

def compute_bart_node_features(od):
    N=od.shape[0]; feats=[]
    for i in range(N):
        r,c=od[i,:],od[:,i]
        feats.append([float(r.sum()),float(c.sum()),float(r.mean()),float(r.std()+1e-8),
                      float(c.mean()),float(c.std()+1e-8),float((r>0).sum()),float((c>0).sum()),
                      float(r.max()),float(c.max()),float(np.percentile(r,75)),float(np.percentile(c,75)),
                      float(np.percentile(r,25)),float(np.percentile(c,25)),
                      float(r.sum()/(c.sum()+1e-6)),float(np.log1p(max(r.sum(),0)))])
    f=np.array(feats,dtype=np.float32); m=f.mean(0); s=f.std(0)+1e-8
    return np.nan_to_num((f-m)/s,nan=0.,posinf=0.,neginf=0.)

def generate_bart_synthetic_scenarios(baseline_matrices, n_synthetic=60, rng_seed=42):
    rng=np.random.default_rng(rng_seed); scenarios=[]
    for _ in range(n_synthetic):
        base=baseline_matrices[rng.integers(len(baseline_matrices))].copy()
        N=base.shape[0]; na=rng.integers(2,max(3,N//4))
        aff=rng.choice(N,size=na,replace=False); delta=np.zeros_like(base); mag=rng.uniform(0.05,0.25)
        for s in aff:
            id_=base[:,s]*mag*rng.uniform(0.5,1.5); na_=np.setdiff1d(np.arange(N),aff)
            if len(na_)>0: delta[na_,s]-=id_.sum()*0.7/len(na_)
            delta[aff,s]+=id_[aff]
        delta-=delta.mean()
        if np.all(np.isfinite(delta)): scenarios.append(delta)
    return scenarios

def load_bart_transfer_dataset(data_dir=DATA_DIR, n_synthetic=60, verbose=True):
    if verbose:
        print("=== Loading BART transfer dataset ===")
        print(f"  Reading from: {os.path.abspath(data_dir)}")
    od = _load_od_matrices(data_dir, verbose)
    if not od:
        raise RuntimeError(
            f"No BART xlsx files found in {os.path.abspath(data_dir)}/\n"
            f"Run the extraction cell first to extract xlsx files from the zip archives."
        )
    gtfs_dir = _download_gtfs(data_dir, verbose)
    ref_key  = next((k for k in ["after_berryessa","before_berryessa"] if k in od), next(iter(od)))
    all_st   = od[ref_key][0]
    if verbose: print(f"  Reference: '{ref_key}' — {len(all_st)} stations")
    graph = build_bart_graph(gtfs_dir, all_st)
    graph["n_nodes"] = len(all_st); graph["station_names"] = all_st
    scenarios = []
    def _real(bl, al, name, split):
        if bl not in od or al not in od:
            if verbose: print(f"  [skip] {name}: missing '{bl}' or '{al}'"); return
        bs,bm=od[bl]; as_,am=od[al]
        _,ba,aa=_align_matrices(bs,bm,as_,am)
        _,bf,af=_align_matrices(all_st,ba,all_st,aa)
        d=af-bf; std=float(d.sum(axis=1).std())+1e-8   # ← changed
        scenarios.append({"node_features":compute_bart_node_features(bf),"delta_od":d,
                           "delta_od_normalized":d/std,"std":std,"is_real":True,"label":name,"split":split})
        if verbose: print(f"  Real '{name}': MAE={np.abs(d).mean():.2f}, split={split}")
    _real("before_berryessa","after_berryessa","berryessa_extension_2020","val")
    _real("before_antioch","after_antioch","antioch_extension_2023","train")
    bl_mats=[]
    for lbl in ["baseline_2019_01","baseline_2019_06","baseline_2019_10","before_berryessa"]:
        if lbl in od:
            bs,bm=od[lbl]; _,ba,_=_align_matrices(bs,bm,all_st,np.zeros((len(all_st),len(all_st))))
            bl_mats.append(ba)
    if bl_mats:
        if verbose: print(f"  Generating {n_synthetic} synthetic scenarios...")
        ref=bl_mats[0]
        for i,d in enumerate(generate_bart_synthetic_scenarios(bl_mats,n_synthetic)):
            std=float(d.sum(axis=1).std())+1e-8   # ← changed
            scenarios.append({"node_features":compute_bart_node_features(ref),"delta_od":d,
                               "delta_od_normalized":d/std,"std":std,"is_real":False,
                               "label":f"synthetic_bart_{i:03d}","split":"train"})
    tc=sum(1 for s in scenarios if s["split"]=="train"); vc=sum(1 for s in scenarios if s["split"]=="val")
    rc=sum(1 for s in scenarios if s["is_real"])
    if verbose:
        print(f"\n  Total: {len(scenarios)} ({rc} real, {len(scenarios)-rc} synthetic)")
        print(f"  Train: {tc}  |  Val: {vc}")
        print("=== BART dataset ready ===\n")
    return graph, scenarios