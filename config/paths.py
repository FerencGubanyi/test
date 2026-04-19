"""
config/paths.py
                       
Elérési utak konfigurációja — Colab és lokális VS Code egyaránt.
Csak ezt a fájlt kell módosítani környezetváltáskor.
                       
"""
import os
 
 
def _detect_env():
    try:
        import google.colab
        return 'colab'
    except ImportError:
        return 'local'
 
 
ENV = _detect_env()
 
if ENV == 'colab':
    BASE_DIR = '/content/drive/MyDrive/bkk_thesis'
else:
    BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    BASE_DIR = os.path.abspath(BASE_DIR)
 
#  Geometria 
ZONES_SHP = os.path.join(BASE_DIR, 'zones_shp', 'm2_zones_zone.shp')
GTFS_ZIP  = os.path.join(BASE_DIR, 'budapest_gtfs.zip')
 
#  M2 matrices 
M2_BASE_KK  = os.path.join(BASE_DIR, 'base_kozossegi_kozlekedes_matrix.xlsx')
M2_DEV_KK   = os.path.join(BASE_DIR, 'm2_meghosszabitas_kozossegi_kozlekedes_matrix.xlsx')
 
#  S000144 matrices
S144_BASE_KK = os.path.join(BASE_DIR, 's000144_base_KK.xlsx')
S144_DIFF_KK = os.path.join(BASE_DIR, 's000144_diff_KK.xlsx')
 
#      M1 matricesb         
M1_KK      = os.path.join(BASE_DIR, 'm1_kozossegi_kozlekedes_matrix.xlsx')
M1_DIFF_KK = os.path.join(BASE_DIR, 'm1_diff_KK.xlsx')
 
# Bus 35 Pesterzsébet 
BUS35_KK      = os.path.join(BASE_DIR, '35_autobusz_kozossegi_kozlekedes_matrix.xlsx')
BUS35_DIFF_KK = os.path.join(BASE_DIR, '35_autobusz_diff_KK.xlsx')
 
#Synthetic scenarios
SYNTHETIC_DIR = os.path.join(BASE_DIR, 'synthetic_scenarios')
 
# Model checkpoints 
GAT_CHECKPOINT = os.path.join(BASE_DIR, 'gat_lstm_best.pt')
HG_CHECKPOINT  = os.path.join(BASE_DIR, 'hg_lstm_best.pt')
 
 
if __name__ == '__main__':
    print(f'Környezet: {ENV}')
    print(f'BASE_DIR:  {BASE_DIR}')
    print(f'Létezik:   {os.path.exists(BASE_DIR)}')
    print()
    files = {
        'M2 baseline':  M2_BASE_KK,
        'M2 after':     M2_DEV_KK,
        'S000144 base': S144_BASE_KK,
        'S000144 diff': S144_DIFF_KK,
        'M1 OD':        M1_KK,
        'M1 diff':      M1_DIFF_KK,
        'Bus35 OD':     BUS35_KK,
        'Bus35 diff':   BUS35_DIFF_KK,
        'GTFS zip':     GTFS_ZIP,
        'Zone SHP':     ZONES_SHP,
    }
    for name, path in files.items():
        icon = '✅' if os.path.exists(path) else '❌'
        print(f'  {icon} {name:15s} {os.path.basename(path)}')