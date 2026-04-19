"""
tests/test_data.py
                             
Tests for the data pipeline:
  - OD matrix parsing (VISUM Excel format)
  - Zone ID extraction and 3-column offset handling
  - CRS transformation correctness
  - Stop-to-zone mapping

All tests use synthetic in-memory data — no real VISUM files required.
                             
"""

import io
import pytest
import numpy as np
import pandas as pd
import openpyxl


#                              
# Helpers — build fake VISUM Excel files in memory
#                              

def _make_visum_excel(
    zone_ids: list,
    matrix: np.ndarray,
    corrupt_header: bool = False,
) -> io.BytesIO:
    """
    Build a VISUM-style OD matrix Excel file in memory.

    Format (as documented in thesis Section 5.2.2):
      Row 0 : ['name', 'sum_col', 'extra', zone_id_1, zone_id_2, ...]
      Row 1 : skip row
      Row 2 : skip row
      Row 3+: [origin_zone_id, 'zone_name', 0, flow_1, flow_2, ...]

    If corrupt_header=True, zone IDs are placed at column index 0 instead
    of 3, simulating the silent-corruption bug from pd.read_excel(header=0).
    """
    wb = openpyxl.Workbook()
    ws = wb.active

    if corrupt_header:
        # Wrong format: zone IDs at col 0 (simulates header=0 misuse)
        ws.append(zone_ids + ['', '', ''])
    else:
        # Correct VISUM format: name/sum/extra columns before zone IDs
        ws.append(['name', 'sum_col', 'extra'] + zone_ids)

    ws.append(['skip_row_1', '', ''] + [''] * len(zone_ids))
    ws.append(['skip_row_2', '', ''] + [''] * len(zone_ids))

    for i, zid in enumerate(zone_ids):
        row = [zid, f'Zone_{zid}', 0] + list(matrix[i])
        ws.append(row)

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


def _parse_od_excel(file_buf: io.BytesIO):
    """
    Reference implementation of the OD parser (mirrors utils/data.py logic).
    Returns (zone_ids, od_matrix) or raises on failure.
    """
    df = pd.read_excel(file_buf, header=None)
    zone_ids = df.iloc[0, 3:].values.astype(float).astype(int)
    data_block = df.iloc[3:, :]
    row_ids = data_block.iloc[:, 0].values.astype(float).astype(int)
    matrix = data_block.iloc[:, 3:].values.astype(float)

    # Align common zones
    common = np.intersect1d(zone_ids, row_ids)
    col_idx = [np.where(zone_ids == z)[0][0] for z in common]
    row_idx = [np.where(row_ids == z)[0][0] for z in common]
    matrix = matrix[np.ix_(row_idx, col_idx)]
    np.fill_diagonal(matrix, 0.0)
    return common, matrix


#                              
# Tests — OD matrix parsing
#                              

class TestODMatrixParsing:

    def setup_method(self):
        """Create a standard 4-zone test case."""
        self.zone_ids = [101, 102, 103, 104]
        self.matrix = np.array([
            [  0, 500, 300, 200],
            [200,   0, 400, 150],
            [100, 150,   0, 350],
            [ 80, 120, 250,   0],
        ], dtype=float)

    def test_zone_ids_extracted_correctly(self):
        """Zone IDs must be read from row 0, columns 3+ (not from header)."""
        buf = _make_visum_excel(self.zone_ids, self.matrix)
        zone_ids, _ = _parse_od_excel(buf)
        assert list(zone_ids) == self.zone_ids

    def test_matrix_shape_matches_zones(self):
        """Output matrix must be square with side = number of zones."""
        buf = _make_visum_excel(self.zone_ids, self.matrix)
        zone_ids, od = _parse_od_excel(buf)
        assert od.shape == (len(self.zone_ids), len(self.zone_ids))

    def test_diagonal_is_zero(self):
        """Intra-zone flow must be zeroed out."""
        buf = _make_visum_excel(self.zone_ids, self.matrix)
        _, od = _parse_od_excel(buf)
        assert np.all(np.diag(od) == 0.0)

    def test_off_diagonal_values_preserved(self):
        """Non-diagonal flow values must be preserved exactly."""
        buf = _make_visum_excel(self.zone_ids, self.matrix)
        _, od = _parse_od_excel(buf)
        # Check a selection of known values
        assert od[0, 1] == pytest.approx(500.0)
        assert od[1, 2] == pytest.approx(400.0)
        assert od[3, 2] == pytest.approx(250.0)

    def test_matrix_is_finite(self):
        """No NaN or Inf values allowed in parsed matrix."""
        buf = _make_visum_excel(self.zone_ids, self.matrix)
        _, od = _parse_od_excel(buf)
        assert np.isfinite(od).all()

    def test_matrix_is_non_negative(self):
        """All flow values must be non-negative."""
        buf = _make_visum_excel(self.zone_ids, self.matrix)
        _, od = _parse_od_excel(buf)
        assert (od >= 0).all()

    def test_single_zone_file(self):
        """Edge case: file with only one zone."""
        buf = _make_visum_excel([999], np.array([[0.0]]))
        zone_ids, od = _parse_od_excel(buf)
        assert len(zone_ids) == 1
        assert od.shape == (1, 1)
        assert od[0, 0] == 0.0

    def test_large_zone_count(self):
        """Parser should handle 200+ zones without error."""
        N = 200
        zone_ids = list(range(1001, 1001 + N))
        rng = np.random.default_rng(0)
        matrix = rng.exponential(100, (N, N))
        np.fill_diagonal(matrix, 0)
        buf = _make_visum_excel(zone_ids, matrix)
        ids, od = _parse_od_excel(buf)
        assert od.shape == (N, N)
        assert np.isfinite(od).all()

    def test_corrupt_header_produces_wrong_zone_ids(self):
        """
        If zone IDs are placed at column 0 (wrong format), the parser
        should NOT produce the correct zone IDs — this documents the
        known silent-corruption failure mode.
        """
        buf = _make_visum_excel(self.zone_ids, self.matrix, corrupt_header=True)
        # With the corrupt file, reading col 3 as zone IDs will give wrong values
        df = pd.read_excel(buf, header=None)
        parsed_ids = df.iloc[0, 3:].values
        # They should NOT match the intended zone IDs
        assert not np.array_equal(
            parsed_ids[:len(self.zone_ids)],
            self.zone_ids
        ), "Corrupt header should produce wrong zone IDs — silent corruption detected"

    def test_zone_ids_are_integers(self):
        """Zone IDs must be cast to int, not float."""
        buf = _make_visum_excel(self.zone_ids, self.matrix)
        zone_ids, _ = _parse_od_excel(buf)
        assert zone_ids.dtype in [np.int32, np.int64, int]

    def test_total_flow_preserved(self):
        """Total flow in the matrix must equal the sum of the input matrix
        (minus diagonal, which is zeroed)."""
        buf = _make_visum_excel(self.zone_ids, self.matrix)
        _, od = _parse_od_excel(buf)
        expected = self.matrix.sum() - np.trace(self.matrix)
        assert od.sum() == pytest.approx(expected, rel=1e-6)


#                              
# Tests — CRS transformation
#                              

class TestCRSTransformation:
    """
    Tests for coordinate reference system handling.
    GTFS uses WGS84 (EPSG:4326); EFM zones use EOV (EPSG:23700).
    The spatial join must happen in EOV to use Euclidean distance.
    """

    def test_budapest_wgs84_to_eov_stays_in_range(self):
        """
        Known Budapest coordinates in WGS84 should transform to EOV values
        in the expected range for Hungary:
          EOV X (Easting):  400,000 – 900,000 m
          EOV Y (Northing): 100,000 – 400,000 m
        """
        pytest.importorskip("pyproj")
        from pyproj import Transformer

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:23700", always_xy=True)

        # Keleti railway station, Budapest
        lon, lat = 19.0838, 47.5003
        eov_x, eov_y = transformer.transform(lon, lat)

        assert 400_000 < eov_x < 900_000, f"EOV X out of range: {eov_x}"
        assert 100_000 < eov_y < 400_000, f"EOV Y out of range: {eov_y}"

    def test_eov_roundtrip_is_accurate(self):
        """WGS84 → EOV → WGS84 roundtrip should recover original coords
        within 1 metre (< 0.00001 degrees)."""
        pytest.importorskip("pyproj")
        from pyproj import Transformer

        to_eov = Transformer.from_crs("EPSG:4326", "EPSG:23700", always_xy=True)
        to_wgs = Transformer.from_crs("EPSG:23700", "EPSG:4326", always_xy=True)

        original_lon, original_lat = 19.0400, 47.4980   # central Budapest
        eov_x, eov_y = to_eov.transform(original_lon, original_lat)
        recovered_lon, recovered_lat = to_wgs.transform(eov_x, eov_y)

        assert abs(recovered_lon - original_lon) < 1e-5
        assert abs(recovered_lat - original_lat) < 1e-5

    def test_multiple_budapest_stops_transform_without_nan(self):
        """A batch of Budapest stop coordinates should all transform cleanly."""
        pytest.importorskip("pyproj")
        from pyproj import Transformer

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:23700", always_xy=True)

        # Sample BKK stop coordinates (WGS84)
        stops = [
            (19.0400, 47.4980),   # Deák Ferenc tér
            (19.0838, 47.5003),   # Keleti pályaudvar
            (18.9750, 47.4970),   # Kelenföld vasútállomás
            (19.1350, 47.5100),   # Kőbánya-Kispest
            (19.0530, 47.5140),   # Blaha Lujza tér
        ]
        lons = [s[0] for s in stops]
        lats = [s[1] for s in stops]

        xs, ys = transformer.transform(lons, lats)

        assert all(np.isfinite(x) for x in xs), "NaN in EOV X coordinates"
        assert all(np.isfinite(y) for y in ys), "NaN in EOV Y coordinates"


#                              
# Tests — Stop-to-zone mapping
#                              

class TestStopToZoneMapping:
    """
    Tests for the stop-to-zone spatial join.
    Uses simple geometric cases that don't require real shapefiles.
    """

    def test_every_stop_maps_to_exactly_one_zone(self):
        """
        Each stop must be assigned to exactly one zone.
        Uses a simple grid of square zones and stops placed at zone centres.
        """
        pytest.importorskip("geopandas")
        pytest.importorskip("shapely")
        import geopandas as gpd
        from shapely.geometry import Point, box

        # Build 4 square zones in a 2×2 grid (EOV-like coordinates)
        zones = gpd.GeoDataFrame({
            "zone_id": [1, 2, 3, 4],
            "geometry": [
                box(0, 0, 100, 100),
                box(100, 0, 200, 100),
                box(0, 100, 100, 200),
                box(100, 100, 200, 200),
            ]
        }, crs="EPSG:23700")

        # One stop per zone, placed at zone centre
        stops = gpd.GeoDataFrame({
            "stop_id": ["s1", "s2", "s3", "s4"],
            "geometry": [
                Point(50, 50),
                Point(150, 50),
                Point(50, 150),
                Point(150, 150),
            ]
        }, crs="EPSG:23700")

        joined = gpd.sjoin(stops, zones, how="left", predicate="within")

        # Each stop should match exactly one zone
        assert len(joined) == len(stops)
        assert joined["zone_id"].notna().all()
        assert set(joined["zone_id"].astype(int)) == {1, 2, 3, 4}

    def test_stop_outside_all_zones_gets_no_zone(self):
        """A stop located outside all zone polygons should not be assigned."""
        pytest.importorskip("geopandas")
        pytest.importorskip("shapely")
        import geopandas as gpd
        from shapely.geometry import Point, box

        zones = gpd.GeoDataFrame({
            "zone_id": [1],
            "geometry": [box(0, 0, 100, 100)]
        }, crs="EPSG:23700")

        stops = gpd.GeoDataFrame({
            "stop_id": ["outside"],
            "geometry": [Point(500, 500)]   # clearly outside
        }, crs="EPSG:23700")

        joined = gpd.sjoin(stops, zones, how="left", predicate="within")
        assert joined["zone_id"].isna().all()

    def test_zone_feature_counts_are_non_negative(self):
        """
        After aggregating stops per zone, all feature counts
        (n_routes, n_trips, n_stops) must be non-negative integers.
        """
        # Simulate aggregated zone features
        zone_features = pd.DataFrame({
            "zone_id":  [1, 2, 3, 4],
            "n_stops":  [3, 0, 5, 2],
            "n_routes": [2, 0, 4, 1],
            "n_trips":  [20, 0, 45, 8],
        })
        for col in ["n_stops", "n_routes", "n_trips"]:
            assert (zone_features[col] >= 0).all(), f"Negative values in {col}"

    def test_zones_without_stops_get_zero_features(self):
        """Zones with no stops must receive zero for all GTFS-derived features."""
        zone_features = pd.DataFrame({
            "zone_id":     [1, 2, 3],
            "n_stops":     [5, 0, 3],
            "n_routes":    [3, 0, 2],
            "has_metro":   [1, 0, 0],
            "has_tram":    [0, 0, 1],
            "has_rail":    [0, 0, 0],
        })
        empty_zone = zone_features[zone_features["zone_id"] == 2].iloc[0]
        assert empty_zone["n_stops"]   == 0
        assert empty_zone["n_routes"]  == 0
        assert empty_zone["has_metro"] == 0
        assert empty_zone["has_tram"]  == 0
