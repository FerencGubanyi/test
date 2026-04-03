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
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

ZONES_SHP     = os.path.join(BASE_DIR, 'zones_shp', 'm2_zones_zone.shp')
GTFS_ZIP      = os.path.join(BASE_DIR, 'budapest_gtfs.zip')
SYNTHETIC_DIR = os.path.join(BASE_DIR, 'synthetic_scenarios')

# M2
M2_BASE_KK = os.path.join(BASE_DIR, 'base_kozossegi_kozlekedes_matrix.xlsx')
M2_DEV_KK  = os.path.join(BASE_DIR, 'm2_meghosszabitas_kozossegi_kozlekedes_matrix.xlsx')

# S000144 - Data from bkks initial VISUM 
S144_BASE_KK = os.path.join(BASE_DIR, 's000144_base_KK.xlsx')
S144_DIFF_KK = os.path.join(BASE_DIR, 's000144_diff_KK.xlsx')

# M1
M1_KK      = os.path.join(BASE_DIR, 'm1_kozossegi_kozlekedes_matrix.xlsx')
M1_DIFF_KK = os.path.join(BASE_DIR, 'm1_diff_KK.xlsx')

# 35 bus
BUS35_KK       = os.path.join(BASE_DIR, '35_autobusz_kozossegi_kozlekedes_matrix.xlsx')
BUS35_DIFF_KK  = os.path.join(BASE_DIR, '35_autobusz_diff_KK.xlsx')

# Checkpoints
GAT_CHECKPOINT = os.path.join(BASE_DIR, 'gat_lstm_best.pt')
HG_CHECKPOINT  = os.path.join(BASE_DIR, 'hg_lstm_best.pt')

if __name__ == '__main__':
    print(f'Enviroment: {ENV}')
    print(f'BASE_DIR:  {BASE_DIR}')
    print(f'Exist:   {os.path.exists(BASE_DIR)}')
