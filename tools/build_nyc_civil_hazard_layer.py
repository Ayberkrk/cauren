"""Build spatial candidate natural-hazard features for the NYC BIN cohort.

USGS PGA values are queried from the current CONUS NSHM static service at
the nearest 0.05-degree grid coordinate. Flood-zone polygons come from the
NYSDOS-hosted FEMA NFHL reduced layer (source NFHL vintage 2021-10-13). Raw
polygon pages and query metadata are kept outside git.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


USGS_MODEL = "conus-2023.R2"
USGS_API = f"https://earthquake.usgs.gov/ws/nshmp/{USGS_MODEL}/static/hazard"
FEMA_SOURCE_ITEM = "b7d24ccb42ed43dabf3302a93bc15811"
FEMA_SOURCE = (
    "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/"
    "USA_Flood_Hazard_Reduced_Set_gdb/FeatureServer/0/query"
)
FEMA_NFHL_VINTAGE = "2021-10-13"
PAGE_SIZE = 1000
CELL_SIZE = 0.01
TARGET_ANNUAL_EXCEEDANCE = 1.0 - (1.0 - 0.02) ** (1.0 / 50.0)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _get_json(url: str, *, attempts: int = 6, timeout: int = 120) -> dict:
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "Cauren-NYC-Civil-Research/1.0"})
        try:
            with urlopen(request, timeout=timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Expected JSON object response")
            return value
        except (HTTPError, URLError, TimeoutError, ConnectionError) as error:
            last_error = error
            if isinstance(error, HTTPError) and error.code not in {429, 500, 502, 503, 504}:
                raise
            if attempt + 1 < attempts:
                time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"request failed after {attempts} attempts: {last_error}")


def _features(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _round_grid(value: float) -> float:
    return round(value / 0.05) * 0.05


def _pga_at_target(curve: dict) -> float | None:
    points = curve.get("response", {}).get("data", {})
    xs, ys = points.get("xs", []), points.get("ys", [])
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    target = TARGET_ANNUAL_EXCEEDANCE
    for index in range(len(xs) - 1):
        x0, x1 = float(xs[index]), float(xs[index + 1])
        y0, y1 = float(ys[index]), float(ys[index + 1])
        if y0 >= target >= y1 and min(x0, x1, y0, y1) > 0:
            fraction = (math.log(target) - math.log(y0)) / (math.log(y1) - math.log(y0))
            return math.exp(math.log(x0) + fraction * (math.log(x1) - math.log(x0)))
    return None


def _usgs_query(point: tuple[float, float]) -> tuple[tuple[float, float], float | None, str | None]:
    latitude, longitude = point
    params = urlencode({"latitude": f"{latitude:.2f}", "longitude": f"{longitude:.2f}",
                        "siteClass": "D", "imt": "PGA", "format": "JSON"})
    try:
        response = _get_json(f"{USGS_API}?{params}")
        if response.get("status") != "success":
            return point, None, str(response.get("status", "unsuccessful USGS response"))
        pga = _pga_at_target(response)
        if pga is None:
            return point, None, "no PGA value bracketed the 2%-in-50-year annual exceedance target"
        return point, pga, None
    except Exception as error:  # keep per-cell failures visible in the manifest
        return point, None, f"{type(error).__name__}: {error}"


def _point_on_segment(point: tuple[float, float], start: list[float], end: list[float]) -> bool:
    x, y = point
    x1, y1 = float(start[0]), float(start[1])
    x2, y2 = float(end[0]), float(end[1])
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > 1e-10:
        return False
    return min(x1, x2) - 1e-10 <= x <= max(x1, x2) + 1e-10 and min(y1, y2) - 1e-10 <= y <= max(y1, y2) + 1e-10


def _inside_ring(point: tuple[float, float], ring: list[list[float]]) -> bool:
    x, y = point
    inside = False
    if len(ring) < 3:
        return False
    for index, start in enumerate(ring):
        end = ring[(index + 1) % len(ring)]
        if _point_on_segment(point, start, end):
            return True
        x1, y1 = float(start[0]), float(start[1])
        x2, y2 = float(end[0]), float(end[1])
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
    return inside


def _point_in_polygon(point: tuple[float, float], rings: list[list[list[float]]]) -> bool:
    return bool(rings) and _inside_ring(point, rings[0]) and not any(_inside_ring(point, ring) for ring in rings[1:])


def _polygon_parts(geometry: dict) -> list[list[list[list[float]]]]:
    if geometry.get("type") == "Polygon":
        return [geometry.get("coordinates", [])]
    if geometry.get("type") == "MultiPolygon":
        return geometry.get("coordinates", [])
    return []


def _iter_features(page_dir: Path):
    for path in sorted(page_dir.glob("flood_page_*.geojson")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        yield from payload.get("features", [])


def _write_flood_pages(output_dir: Path, points: list[tuple[float, float]]) -> tuple[int, int]:
    page_dir = output_dir / "fema_flood_pages"
    page_dir.mkdir(parents=True, exist_ok=True)
    longitudes = [point[1] for point in points]
    latitudes = [point[0] for point in points]
    margin = 0.015
    envelope = (min(longitudes) - margin, min(latitudes) - margin,
                max(longitudes) + margin, max(latitudes) + margin)
    offset = 0
    total_features = 0
    while True:
        page_path = page_dir / f"flood_page_{offset:08d}.geojson"
        if page_path.exists():
            payload = json.loads(page_path.read_text(encoding="utf-8"))
        else:
            params = {
                "where": "SFHA_TF = 'T'",
                "geometry": ",".join(f"{value:.8f}" for value in envelope),
                "geometryType": "esriGeometryEnvelope",
                "spatialRel": "esriSpatialRelIntersects",
                "inSR": 4326,
                "outSR": 4326,
                "outFields": "OBJECTID,DFIRM_ID,FLD_AR_ID,FLD_ZONE,SFHA_TF",
                "returnGeometry": "true",
                "resultOffset": offset,
                "resultRecordCount": PAGE_SIZE,
                "orderByFields": "OBJECTID ASC",
                "f": "geojson",
            }
            payload = _get_json(f"{FEMA_SOURCE}?{urlencode(params)}")
            if "error" in payload:
                raise RuntimeError(f"FEMA NFHL reduced-layer query failed: {payload['error']}")
            temporary = page_path.with_suffix(".geojson.tmp")
            temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            temporary.replace(page_path)
        rows = payload.get("features", [])
        total_features += len(rows)
        print(f"FEMA flood layer: offset {offset:,}, {len(rows):,} SFHA polygons", flush=True)
        if len(rows) < PAGE_SIZE or not payload.get("properties", {}).get("exceededTransferLimit", False):
            break
        offset += len(rows)
    return total_features, len(list(page_dir.glob("flood_page_*.geojson")))


def _flood_hits(features: list[dict[str, str]], flood_features: list[dict]) -> dict[str, bool]:
    cells: dict[tuple[int, int], list[tuple[dict, list[list[list[float]]]]]] = defaultdict(list)
    for feature in flood_features:
        geometry = feature.get("geometry") or {}
        for rings in _polygon_parts(geometry):
            if not rings or not rings[0]:
                continue
            xs = [float(point[0]) for point in rings[0]]
            ys = [float(point[1]) for point in rings[0]]
            min_cell_x, max_cell_x = math.floor((min(xs) + 180) / CELL_SIZE), math.floor((max(xs) + 180) / CELL_SIZE)
            min_cell_y, max_cell_y = math.floor((min(ys) + 90) / CELL_SIZE), math.floor((max(ys) + 90) / CELL_SIZE)
            item = (feature, rings)
            for cell_x in range(min_cell_x, max_cell_x + 1):
                for cell_y in range(min_cell_y, max_cell_y + 1):
                    cells[(cell_x, cell_y)].append(item)

    results: dict[str, bool] = {}
    for feature in features:
        asset_id = feature.get("asset_id", "")
        try:
            latitude, longitude = float(feature["latitude"]), float(feature["longitude"])
        except (KeyError, ValueError, TypeError):
            results[asset_id] = False
            continue
        key = (math.floor((longitude + 180) / CELL_SIZE), math.floor((latitude + 90) / CELL_SIZE))
        results[asset_id] = any(_point_in_polygon((longitude, latitude), rings) for _, rings in cells.get(key, []))
    return results


def build_hazard_layer(*, feature_path: Path, output_dir: Path, max_workers: int = 4) -> dict:
    features = _features(feature_path)
    points_by_bin: dict[str, tuple[float, float]] = {}
    for feature in features:
        try:
            latitude, longitude = float(feature["latitude"]), float(feature["longitude"])
        except (KeyError, ValueError, TypeError):
            continue
        if not (40.45 <= latitude <= 40.95 and -74.35 <= longitude <= -73.65):
            continue
        points_by_bin[feature["asset_id"]] = (latitude, longitude)
    if not points_by_bin:
        raise ValueError(f"No valid NYC coordinates in {feature_path}")
    grid_points = sorted({(_round_grid(lat), _round_grid(lon)) for lat, lon in points_by_bin.values()})
    output_dir.mkdir(parents=True, exist_ok=True)

    pga_by_grid: dict[tuple[float, float], float | None] = {}
    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_usgs_query, point): point for point in grid_points}
        for future in as_completed(futures):
            point, pga, error = future.result()
            pga_by_grid[point] = pga
            if error:
                errors.append({"latitude": point[0], "longitude": point[1], "error": error})
            print(f"USGS NSHM grid {point[0]:.2f},{point[1]:.2f}: {pga if pga is not None else 'missing'}", flush=True)

    flood_feature_count, page_count = _write_flood_pages(output_dir, list(points_by_bin.values()))
    flood_features = list(_iter_features(output_dir / "fema_flood_pages"))
    # Only assign a negative flood-zone flag to locations included in the
    # queried NYC envelope. Bad or out-of-area permit coordinates remain
    # missing rather than being treated as confirmed non-SFHA points.
    sfha_by_bin = _flood_hits(
        [feature for feature in features if feature.get("asset_id") in points_by_bin],
        flood_features,
    )
    rows = []
    pga_missing = 0
    for feature in features:
        asset_id = feature.get("asset_id", "")
        point = points_by_bin.get(asset_id)
        if point is None:
            pga = None
            sfha = None
        else:
            grid_point = (_round_grid(point[0]), _round_grid(point[1]))
            pga = pga_by_grid.get(grid_point)
            sfha = sfha_by_bin.get(asset_id)
        if pga is None:
            seismic_component = None
            score = None
            pga_missing += 1
        else:
            seismic_component = min(1.0, max(0.0, pga / 0.6))
            flood_component = 1.0 if sfha is True else 0.0 if sfha is False else None
            score = 0.6 * seismic_component + 0.4 * flood_component if flood_component is not None else None
        rows.append({
            "asset_id": asset_id,
            "pga_2pct_50yr_g": pga,
            "seismic_component": seismic_component,
            "fema_sfha": sfha,
            "natural_hazard_score": score,
            "usgs_grid_latitude": _round_grid(point[0]) if point else "",
            "usgs_grid_longitude": _round_grid(point[1]) if point else "",
        })
    scores_path = output_dir / "hazard_scores.csv"
    with scores_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["asset_id"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "created_at_utc": _utc_now(),
        "input_feature_file": str(feature_path),
        "building_count": len(features),
        "coordinate_count": len(points_by_bin),
        "coordinate_fraction": len(points_by_bin) / len(features) if features else None,
        "usgs": {
            "source": USGS_API,
            "model": USGS_MODEL,
            "model_release_reference": "https://doi.org/10.5066/P14VGAV4",
            "query": "nearest 0.05-degree point, site class D, PGA; log-log interpolate hazard curve at 2% probability of exceedance in 50 years",
            "grid_cells_requested": len(grid_points),
            "grid_cells_failed_or_missing": len(errors),
            "errors": errors,
            "missing_building_values": pga_missing,
        },
        "fema": {
            "source_item_id": FEMA_SOURCE_ITEM,
            "source": FEMA_SOURCE,
            "nfhl_source_vintage": FEMA_NFHL_VINTAGE,
            "query": "NYC coordinate envelope; SFHA_TF = T polygons only; point-in-polygon join",
            "sfha_polygon_count": flood_feature_count,
            "polygon_page_count": page_count,
            "sfha_building_count": sum(value is True for value in sfha_by_bin.values()),
            "coverage_note": "NYSDOS-hosted FEMA NFHL reduced layer omits low-risk X polygons and is derived from the 2021-10-13 NFHL; a polygon hit is positive SFHA evidence, while non-hit is provisional absent a separate map-availability confirmation.",
        },
        "score_formula": "0.6 * clamp(PGA_2pct_50yr_g / 0.6, 0, 1) + 0.4 * FEMA SFHA flag",
        "score_file": str(scores_path),
    }
    manifest_path = output_dir / "hazard_source_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-file", type=Path, default=Path("data/cauren_civil_nyc/normalized/features.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cauren_civil_nyc/normalized"))
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    if args.max_workers < 1 or args.max_workers > 8:
        parser.error("--max-workers must be between 1 and 8")
    build_hazard_layer(feature_path=args.feature_file, output_dir=args.output_dir, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
