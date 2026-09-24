"""
foccus_helpers.py — routines backing the FOCCUS storm-surge demonstrator.

The notebook runs on the server that hosts the data, and every figure reads
straight from the stores:

    Track A (1D, per port) .... ports/validation_<STATION>.parquet
                                ports/scores_<STATION>.json
    Track B (2D, gridded) ..... fields/<name>.nc, one field per file

Paths come from the environment, so the same notebook runs anywhere unedited:

    FOCCUS_DATA_ROOT   data stores                     (default: ./data)
    FOCCUS_FIG_ROOT    rendered figures and caches     (default: ./figs)
    FOCCUS_COASTLINE   coastline for the station map   (optional)

Heavy dependencies (pandas, xarray) are imported inside the functions
that need them, so importing the module stays cheap.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

# The fields are only ever read here. HDF5 file locking protects concurrent
# writers, and on some parallel filesystems (Lustre among them) it makes a plain
# read fail with "unable to lock file". Must be set before HDF5 is loaded.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

__version__ = "3.1"

__all__ = [
    "setup", "download_data",
    "plot_station_map", "plot_timeseries_explorer", "plot_gauge_scores",
    "plot_event_scores", "station_stats_table",
    "ecfas_scores", "plot_ecfas_scores",
    "build_map_cache", "plot_map_comparator", "plot_skill_map",
    "field_scores_2d", "storm_peak_times",
    "plot_dataflow",
]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT = Path(os.environ.get("FOCCUS_DATA_ROOT", "data"))
FIG_ROOT = Path(os.environ.get("FOCCUS_FIG_ROOT", "figs"))

PORTS_DIR = DATA_ROOT / "ports"
PORT_PATTERN = "*.parquet"
FIELDS_DIR = DATA_ROOT / "fields"
FIELD_EXT = ".nc"
#: Tolosa-SW mesh cached by convert_fields_to_netcdf.py, used for a field that
#: is indexed by element but does not carry its own topology
MESH_FILE = FIELDS_DIR / "_mesh.nc"
COASTLINE_FILE = Path(os.environ.get("FOCCUS_COASTLINE",
                                     str(DATA_ROOT / "coastline.shp")))

#: what :func:`download_data` fetches under :data:`DATA_ROOT`: folders are
#: recreated as they are, so the list follows the layout, not the file names
REMOTE_SUBFOLDER = "Shom/data/"
REMOTE_CONTENT = ["ports", "fields"]

#: built, not authored: per-station series fetched by the explorer, and the
#: rendered maps served to the comparator — both by relative path, so the
#: notebook carries paths rather than data
SERIES_CACHE = FIG_ROOT / "series"
MAP_CACHE = FIG_ROOT / "multiband"
DATAFLOW_PNG = FIG_ROOT / "D321.png"


# ---------------------------------------------------------------------------
# Stations and storms
# ---------------------------------------------------------------------------

#: Retained stations, in geographic order (Basque coast -> North Sea), so every
#: cross-station panel reads as a coastal transect.
STATIONS = [
    "BOUCAU_BAYONNE", "ARCACHON_EYRAC", "PORT_BLOC", "LA_ROCHELLE_PALLICE",
    "LES_SABLES_D_OLONNE", "LE_CROUESTY", "CONCARNEAU", "BREST", "LE_CONQUET",
    "ROSCOFF", "SAINT_MALO", "CHERBOURG", "LE_HAVRE", "BOULOGNE_SUR_MER",
    "CALAIS", "DUNKERQUE",
]

#: (lat, lon) of each station
PORTS_COORDS = {
    "BOUCAU_BAYONNE": (43.527320, -1.514830),
    "ARCACHON_EYRAC": (44.665001, -1.163550),
    "PORT_BLOC": (45.568480, -1.061555),
    "LA_ROCHELLE_PALLICE": (46.158501, -1.220650),
    "LES_SABLES_D_OLONNE": (46.497358, -1.793528),
    "LE_CROUESTY": (47.542676, -2.895167),
    "CONCARNEAU": (47.873549, -3.907207),
    "BREST": (48.382900, -4.495040),
    "LE_CONQUET": (48.359098, -4.780750),
    "ROSCOFF": (48.718426, -3.965679),
    "SAINT_MALO": (48.640812, -2.028103),
    "CHERBOURG": (49.651447, -1.635508),
    "LE_HAVRE": (49.481892, 0.105980),
    "BOULOGNE_SUR_MER": (50.727380, 1.577660),
    "CALAIS": (50.969399, 1.867720),
    "DUNKERQUE": (51.048091, 2.366698),
}

#: The four named storms of the validation winter. The date is the day the
#: storm reached the coast; the peak is searched around it, so it need only be
#: approximate.
EVENTS = [
    {"name": "Benjamin", "date": "2025-10-23"},
    {"name": "Goretti",  "date": "2026-01-09"},
    {"name": "Nils",     "date": "2026-02-12"},
    {"name": "Pedro",    "date": "2026-02-19"},
]

#: Stations each storm reached, from the reported coastal impact; drawn as a
#: ring on the station map.
EVENT_STATIONS = {
    "Benjamin": ("BOUCAU_BAYONNE", "ARCACHON_EYRAC", "PORT_BLOC",
                 "LA_ROCHELLE_PALLICE"),
    "Goretti":  ("LE_CONQUET", "BREST", "ROSCOFF", "SAINT_MALO", "CHERBOURG",
                 "LE_HAVRE"),
    "Nils":     ("ARCACHON_EYRAC", "PORT_BLOC"),
    "Pedro":    ("BOUCAU_BAYONNE", "ARCACHON_EYRAC", "PORT_BLOC"),
}

EVENT_SEARCH_H = 36     # peak search, hours either side of the event date
EVENT_WINDOW_H = 24     # scoring window, hours either side of the observed peak
EVENT_LAG_H = 6         # lag search, hours
EVENT_SMOOTH_MIN = 60   # smoothing before peak detection, minutes

#: margin around the stations when framing the station map, degrees
STATION_MAP_MARGIN = 0.9


# ---------------------------------------------------------------------------
# Series, maps, colours
# ---------------------------------------------------------------------------

#: display name -> column in the per-station Parquet
COLMAP = {
    "Observed surge": "obs",
    "Tolosa-SW (raw)": "Tolosa-SW",
    "Tolosa-SW (XGB-corrected)": "Tolosa-SW XGB-corrected",
}
REF_SERIES = "Observed surge"
RAW_SERIES = "Tolosa-SW (raw)"
COR_SERIES = "Tolosa-SW (XGB-corrected)"
TIME_COL = "time"

#: field store -> label in the comparator menus
MAP_SOURCES = {
    "bias_ref": "Bias raw (L4 − Tolosa)",
    "model": "Tolosa-SW raw",
    "correction": "Correction (EOF/CCA [20,128] d)",
    "corr_plus_model": "Tolosa-SW corrected",
}
MAP_DEFAULT_LEFT = "model"
MAP_DEFAULT_RIGHT = "corr_plus_model"

#: Sources in one group share one symmetric colour range over every date, so
#: two panels can be compared by eye and the scale does not move with the slider.
MAP_GROUPS = {
    "state":  {"sources": ("model", "corr_plus_model", "sat"), "cmap": "balance"},
    "signal": {"sources": ("bias_ref", "correction", "residual"), "cmap": "diff"},
}
SKILL_CMAP = "delta"

#: matplotlib stand-ins when cmocean is not installed
_CMAP_FALLBACK = {"balance": "RdBu_r", "diff": "PuOr_r", "delta": "PRGn_r"}

VLIM_SAMPLE = 4          # dates sampled per store for the colour range
VLIM_PERCENTILE = 98.0
VLIM_STRIDE = 10         # element subsampling before the percentile

C_OBS = "#1b1b1b"   # observation
C_RAW = "#d1495b"   # raw Tolosa-SW
C_COR = "#1f7a8c"   # corrected
C_AUX = "#e09f3e"   # storm-event subset

_DATE_RE = re.compile(r"_(\d{4})-(\d{2})-(\d{2})\.png$")
_NON_PAYLOAD = {
    "Mesh2", "Mesh2_face_nodes", "Mesh2_node_x", "Mesh2_node_y",
    "Mesh2_face_x", "Mesh2_face_y", "face_nodes", "nv",
    "node_lon", "node_lat", "clon", "clat", "lon", "lat",
    "time", "time_bnds", "crs",
}


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_SETUP_DONE = False


def _apply_style():
    global _SETUP_DONE
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 110,
        "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10,
    })
    _SETUP_DONE = True


def _ensure_setup():
    """Style the figures even when the setup cell was not run first."""
    if not _SETUP_DONE:
        _apply_style()


def setup():
    """Apply the notebook-wide figure style."""
    _apply_style()
    print("Environment ready.")


def _placeholder(ax, msg):
    """Neutral placeholder when a store is not reachable."""
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", wrap=True,
            fontsize=10, color="#666",
            bbox=dict(boxstyle="round", fc="#f4f4f4", ec="#ccc"))


def download_data(content=None, force=False):
    """Fetch from the shared bucket whatever is missing under ``data/``.

    The notebook reads the gauge records under ``ports/`` and the field
    stores under ``fields/``; ``content`` overrides that list, with folder
    names recreated as they are. Nothing already present is downloaded
    again unless ``force`` is set, so the call is safe to leave at the top
    of the notebook.
    """
    from download_from_s3 import download_files_from_s3

    wanted = list(REMOTE_CONTENT if content is None else content)
    missing = wanted if force else [c for c in wanted
                                    if not (DATA_ROOT / c).exists()]
    if not missing:
        print(f"Data already in place under {DATA_ROOT}/.")
        return
    print(f"Fetching {', '.join(missing)} → {DATA_ROOT}/ …")
    download_files_from_s3(
        files_to_download=missing,
        s3_subfolder=REMOTE_SUBFOLDER,
        local_output_dir=str(DATA_ROOT),
    )
    still = [c for c in missing if not (DATA_ROOT / c).exists()]
    if still:
        print(f"  [warn] still missing after download: {', '.join(still)}")
    else:
        print("Download complete.")


# ---------------------------------------------------------------------------
# Stations
# ---------------------------------------------------------------------------

def _norm_station(name) -> str:
    """Upper case, any non-alphanumeric run as one underscore."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name).upper()).strip("_")


def _pretty_station(name: str) -> str:
    """``LA_ROCHELLE_PALLICE`` -> ``La Rochelle Pallice``, for axis labels.

    Storage keys are upper-case with underscores; a published figure should not
    show them that way.
    """
    return str(name).replace("_", " ").title()


def stations() -> list:
    return [_norm_station(s) for s in STATIONS]


def station_coords(name):
    return PORTS_COORDS.get(_norm_station(name))


def _station_name(path: Path) -> str:
    name = path.stem
    for prefix in ("validation_", "scores_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def station_files() -> dict:
    """``{station: Path}`` for the listed stations with a file, in list order."""
    if not PORTS_DIR.exists():
        return {}
    found = {_norm_station(_station_name(f)): f
             for f in sorted(PORTS_DIR.glob(PORT_PATTERN))}
    return {name: found[name] for name in stations() if name in found}


def station_frame(path):
    """One station's file as a time-indexed DataFrame with display names.

    Series named in :data:`COLMAP` come first, in that order; any other numeric
    column follows under its own name.
    """
    import pandas as pd

    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if TIME_COL in df.columns:
        df = df.set_index(pd.to_datetime(df[TIME_COL]))
    else:
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # a repeated timestamp turns every per-instant lookup into a Series
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="first")]

    cols = {disp: df[col] for disp, col in COLMAP.items() if col in df.columns}
    used = {c for c in COLMAP.values() if c in df.columns}
    for col in df.columns:
        if col != TIME_COL and col not in used and df[col].dtype.kind in "fi":
            cols[col] = df[col]
    return pd.DataFrame(cols, index=df.index)


# ---------------------------------------------------------------------------
# Series cache for the time-series explorer
# ---------------------------------------------------------------------------

def _series_payload(path) -> dict:
    """One station at native resolution, compact enough to fetch in a browser.

    A regular time axis is written as origin + step rather than as ~260k
    timestamps, and values are rounded to the millimetre.
    """
    df = station_frame(path)
    idx = df.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    t = idx.astype("datetime64[s]").astype("int64").to_numpy()

    out = {"series": {}}
    d = t[1:] - t[:-1] if len(t) > 2 else None
    if d is not None and d.min() == d.max() and d[0] > 0:
        out["t0"], out["dt"], out["n"] = int(t[0]), int(d[0]), int(len(t))
    else:
        out["t"] = t.tolist()
    for disp in df.columns:
        out["series"][disp] = [
            None if (x is None or (isinstance(x, float) and math.isnan(x)))
            else round(float(x), 3) for x in df[disp].to_numpy()
        ]
    return out


def _series_signature() -> str:
    """Fingerprint of what decides a series file's content.

    Modification times cannot see a change of column mapping, so a cache built
    under an older mapping would look current while being wrong.
    """
    import hashlib

    payload = json.dumps({"colmap": COLMAP, "version": __version__},
                         sort_keys=True).encode()
    return hashlib.md5(payload).hexdigest()[:12]


def _export_series() -> dict:
    """Write one JSON per station under :data:`SERIES_CACHE`; ``{station: Path}``.

    A file is rewritten when its Parquet is newer or the mapping has changed.
    """
    SERIES_CACHE.mkdir(parents=True, exist_ok=True)
    sig_file = SERIES_CACHE / "_signature.txt"
    sig = _series_signature()
    try:
        stale = sig_file.read_text().strip() != sig
    except Exception:
        stale = True

    out, written, failed = {}, 0, 0
    for name, src in station_files().items():
        dest = SERIES_CACHE / f"{name}.json"
        if (not stale and dest.exists()
                and dest.stat().st_mtime >= src.stat().st_mtime):
            out[name] = dest
            continue
        try:
            with open(dest, "w") as fh:
                json.dump(_series_payload(src), fh, separators=(",", ":"))
            out[name] = dest
            written += 1
        except Exception as exc:
            failed += 1
            print(f"  [warn] {name}: {exc}")
    if written and not failed:
        sig_file.write_text(sig)
    return out


# ---------------------------------------------------------------------------
# Track A — station map
# ---------------------------------------------------------------------------

def _basemap_coastline(ax, extent, path=None):
    """Draw a coastline from a vector file. Returns a label, or None."""
    path = Path(COASTLINE_FILE if path is None else path)
    if not path.exists():
        return None
    try:
        import geopandas as gpd
    except ImportError:
        return None
    try:
        gdf = gpd.read_file(path)
        if gdf.crs is not None:
            gdf = gdf.to_crs(4326)
        if extent is not None:
            gdf = gdf.cx[extent[0]:extent[1], extent[2]:extent[3]]
        if gdf.empty:
            return None
        kinds = set(gdf.geom_type)
        if kinds & {"Polygon", "MultiPolygon"}:
            gdf.plot(ax=ax, facecolor="#eceff1", edgecolor="#90a4ae",
                     linewidth=0.6, zorder=0)
        else:
            gdf.plot(ax=ax, color="#607d8b", linewidth=0.7, zorder=0)
        return f"coastline ({path.name})"
    except Exception:
        return None


def _basemap_cartopy(ax, extent):
    """Draw Natural Earth land through cartopy's reader. Returns a label, or None.

    Only the geometries are borrowed, not the projection machinery, so this
    still draws on a plain lon/lat axes. Cartopy fetches the file on first use,
    which needs network access; that is why it is not the default.
    """
    try:
        from cartopy.io.shapereader import natural_earth, Reader
        from matplotlib.patches import Polygon as MplPolygon
    except ImportError:
        return None
    try:
        fn = natural_earth(resolution="10m", category="physical", name="land")
        for geom in Reader(fn).geometries():
            polys = getattr(geom, "geoms", [geom])
            for poly in polys:
                if extent is not None:
                    x0, y0, x1, y1 = poly.bounds
                    if (x1 < extent[0] or x0 > extent[1] or
                            y1 < extent[2] or y0 > extent[3]):
                        continue
                ax.add_patch(MplPolygon(list(poly.exterior.coords),
                                        closed=True, facecolor="#eceff1",
                                        edgecolor="#90a4ae", linewidth=0.6,
                                        zorder=0))
        return "Natural Earth 10 m (cartopy)"
    except Exception:
        return None


def _basemap_mesh(ax, extent, stride=97):
    """Draw the Tolosa-SW mesh nodes. Returns a label, or None."""
    nodes = _domain_nodes(stride)
    if nodes is None:
        return None
    lon, lat = nodes
    if extent is not None:
        keep = ((lon >= extent[0]) & (lon <= extent[1]) &
                (lat >= extent[2]) & (lat <= extent[3]))
        lon, lat = lon[keep], lat[keep]
    if not lon.size:
        return None
    ax.plot(lon, lat, ".", ms=0.6, color="#cfd8dc", alpha=0.6, zorder=0,
            rasterized=True)
    return "Tolosa-SW mesh nodes"


def _draw_basemap(ax, extent):
    """Coastline file, then Natural Earth through cartopy, then the model mesh."""
    for draw in (lambda: _basemap_coastline(ax, extent),
                 lambda: _basemap_cartopy(ax, extent),
                 lambda: _basemap_mesh(ax, extent)):
        if draw():
            return


def _domain_nodes(stride=97):
    """Subsampled ``(lon, lat)`` mesh nodes from the first readable field store.

    Only the node coordinates are read: building the triangulation of a
    2.5 M-element mesh to draw a context map would cost far more than the map is
    worth. Returns ``None`` when no store is reachable.
    """
    import numpy as np
    for name in list_fields():
        try:
            ds = open_field(name)
        except Exception:
            continue
        found = _mesh_names(ds)
        if found is None:
            continue
        _, lon_name, lat_name = found
        try:
            lon = np.asarray(ds[lon_name].values).ravel()[::stride]
            lat = np.asarray(ds[lat_name].values).ravel()[::stride]
        except Exception:
            continue
        if lon.size and lon.size == lat.size:
            return lon, lat
    return None


def plot_station_map():
    """The tide-gauge network on the French Atlantic and Channel coasts.

    Framed on the stations. A station listed but without a record is drawn
    hollow; a station reached by a validation storm carries a ring and the
    initial of each storm.
    """
    _ensure_setup()
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    want = stations()
    have = set(station_files())
    pts = [station_coords(s) for s in want if station_coords(s) is not None]
    lats = [p[0] for p in pts]; lons = [p[1] for p in pts]
    m = STATION_MAP_MARGIN
    extent = (min(lons) - m, max(lons) + m, min(lats) - m, max(lats) + m)

    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    _draw_basemap(ax, extent)

    hit = {}
    for ev, members in EVENT_STATIONS.items():
        for s in members:
            hit.setdefault(_norm_station(s), []).append(ev)

    for name in want:
        pos = station_coords(name)
        if pos is None:
            continue
        lat, lon = pos
        present = name in have
        storms = hit.get(name, [])
        if storms:
            ax.plot(lon, lat, "o", ms=14, zorder=2, mfc="none", mec=C_RAW,
                    mew=1.3, alpha=0.85)
        ax.plot(lon, lat, "o", ms=7, zorder=3,
                mfc=(C_COR if present else "none"),
                mec=(C_COR if present else "#9e9e9e"), mew=1.4)
        label = _pretty_station(name)
        if storms:
            label += "  " + "".join(e[0] for e in storms)
        # a close eastern neighbour at the same latitude (Le Conquet / Brest,
        # Calais / Dunkerque) would sit under the label: put it on the west side
        crowded = any(0 < q[1] - lon < 0.6 and abs(q[0] - lat) < 0.15
                      for q in pts)
        ax.annotate(label, (lon, lat), textcoords="offset points",
                    xytext=(-11 if crowded else 11, 0),
                    ha="right" if crowded else "left", va="center",
                    fontsize=8, zorder=4,
                    color=("#1b1b1b" if present else "#9e9e9e"))

    handles = [Line2D([], [], ls="none", marker="o", ms=7, mfc=C_COR,
                      mec=C_COR, label="station with a record"),
               Line2D([], [], ls="none", marker="o", ms=7, mfc="none",
                      mec="#9e9e9e", label="listed, no record"),
               Line2D([], [], ls="none", marker="o", ms=11, mfc="none",
                      mec=C_RAW, label="reached by a validation storm")]
    ax.legend(handles=handles, fontsize=8, loc="lower left", framealpha=0.9)

    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1.0 / max(0.2, np.cos(np.deg2rad(47.5))))
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"REFMAR tide gauges calibrated by Track A "
                 f"({len(have)} of {len(want)} with data)")
    ax.grid(alpha=0.25)
    fig.text(0.5, 0.005, "Storms: " + " · ".join(f"{e[0]} {e}" for e in EVENT_STATIONS),
             ha="center", va="bottom", fontsize=8, color="#555")
    plt.tight_layout(rect=(0, 0.02, 1, 1)); plt.show()


# ---------------------------------------------------------------------------
# Track A — time-series explorer
# ---------------------------------------------------------------------------

def plot_timeseries_explorer():
    """Interactive per-station surge explorer (uPlot).

    Drag to zoom, double-click to reset; the statistics under the plot are
    recomputed over the visible window against the observed surge. Each
    station's series is fetched from :data:`SERIES_CACHE` when selected.
    """
    _ensure_setup()
    from IPython.display import HTML, display

    cwd = Path.cwd()
    manifest = {}
    for name, dest in _export_series().items():
        try:
            url = os.path.relpath(Path(dest).resolve(), cwd)
        except ValueError:                      # different drive / mount
            url = str(Path(dest).resolve())
        manifest[name] = {"url": url}

    colors = {REF_SERIES: C_OBS, RAW_SERIES: C_RAW, COR_SERIES: C_COR}
    config = json.dumps({"ref": REF_SERIES, "height": 380, "colors": colors})
    display(HTML(_EXPLORER_TMPL
                 .replace("__PAYLOAD__", json.dumps(manifest))
                 .replace("__CONFIG__", config)))
    if not manifest:
        print(f"No station series available under {PORTS_DIR}.")


# ---------------------------------------------------------------------------
# Track A — storm by storm
# ---------------------------------------------------------------------------

def _peak_time(sr, smooth_min=EVENT_SMOOTH_MIN):
    """Time of the peak of ``sr``, smoothed first so the answer is stable.

    A bare argmax on a 10-minute record moves between neighbouring samples from
    one series to the next, which would show up as a spurious few-minute lag.
    """
    if sr.dropna().empty:
        return None
    n = max(1, int(smooth_min // 10))
    sm = sr.rolling(n, center=True, min_periods=1).mean() if n > 1 else sr
    return sm.idxmax()


def _lag_minutes(ref, other, max_h=EVENT_LAG_H):
    """Lag of ``other`` against ``ref``, in minutes, by cross-correlation.

    Positive means the model peaks *late*. Cross-correlation rather than a
    difference of peak times: it uses the whole peak rather than one sample, so
    it survives a flat or double-peaked surge, which a difference of argmax does
    not.
    """
    import numpy as np

    a = ref.to_numpy(dtype="float64")
    b = other.to_numpy(dtype="float64")
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 12:
        return None
    a = a[ok] - a[ok].mean()
    b = b[ok] - b[ok].mean()
    if not (a.any() and b.any()):
        return None
    step_min = 10
    try:
        step_min = int(round((ref.index[1] - ref.index[0]).total_seconds() / 60)) or 10
    except Exception:
        pass
    max_lag = max(1, int(max_h * 60 // step_min))
    best, best_lag = -np.inf, 0
    for k in range(-max_lag, max_lag + 1):
        if k < 0:
            x, y = a[-k:], b[:len(b) + k]
        elif k > 0:
            x, y = a[:len(a) - k], b[k:]
        else:
            x, y = a, b
        if len(x) < 12:
            continue
        denom = np.sqrt((x * x).sum() * (y * y).sum())
        if denom <= 0:
            continue
        c = float((x * y).sum() / denom)
        if c > best:
            best, best_lag = c, k
    return best_lag * step_min


def _align_tz(ts, index):
    """Put ``ts`` in the same timezone convention as ``index``.

    Validation files are often written in UTC while the event dates here are
    plain calendar days; comparing the two raises rather than returning
    something wrong, so the alignment happens once, up front.
    """
    import pandas as pd

    tz = getattr(index, "tz", None)
    ts = pd.Timestamp(ts)
    if tz is not None and ts.tzinfo is None:
        return ts.tz_localize(tz)
    if tz is None and ts.tzinfo is not None:
        return ts.tz_localize(None)
    return ts


def event_scores(impacted_only=False):
    """Per-station, per-storm scores around the observed surge peak.

    The observed peak is located within ``EVENT_SEARCH_H`` of each event date;
    every score is then computed on ``± EVENT_WINDOW_H`` around that peak.
    One row per storm, station and model series: window RMSE, peak amplitude
    ratio (modelled maximum over observed maximum), signed peak error (m,
    modelled maximum minus observed maximum) and timing lag (minutes,
    positive when the modelled peak is late). With ``impacted_only``, a storm
    is scored only at the stations listed for it in :data:`EVENT_STATIONS`.
    """
    import numpy as np
    import pandas as pd

    rows, spans, skipped_ref = [], [], 0
    models = [RAW_SERIES, COR_SERIES]
    for name, path in station_files().items():
        try:
            df = station_frame(path)
        except Exception as exc:
            print(f"  [warn] {name}: {exc}")
            continue
        if REF_SERIES not in df.columns:
            skipped_ref += 1
            print(f"  [skip] {name}: no '{REF_SERIES}' series "
                  f"(columns: {', '.join(map(str, df.columns[:8]))})")
            continue
        if len(df.index):
            spans.append((df.index[0], df.index[-1]))

        for ev in EVENTS:
            if impacted_only and name not in {
                    _norm_station(s) for s in EVENT_STATIONS.get(ev["name"], ())}:
                continue
            t0 = _align_tz(ev["date"], df.index)
            search = df.loc[t0 - pd.Timedelta(hours=EVENT_SEARCH_H):
                            t0 + pd.Timedelta(hours=EVENT_SEARCH_H)]
            if search.empty:
                continue
            tp = _peak_time(search[REF_SERIES])
            if tp is None:
                continue
            win = df.loc[tp - pd.Timedelta(hours=EVENT_WINDOW_H):
                         tp + pd.Timedelta(hours=EVENT_WINDOW_H)]
            obs = win[REF_SERIES]
            obs_peak = float(obs.max()) if obs.notna().any() else np.nan
            for m in models:
                if m not in win.columns:
                    continue
                mod = win[m]
                both = obs.notna() & mod.notna()
                if both.sum() < 12:
                    continue
                err = (mod - obs)[both]
                lag = _lag_minutes(obs[both], mod[both])
                rows.append({
                    "event": ev["name"], "station": name, "series": m,
                    "rmse": float(np.sqrt((err ** 2).mean())),
                    "peak_ratio": (float(mod.max()) / obs_peak
                                   if obs_peak and np.isfinite(obs_peak)
                                   else np.nan),
                    "peak_err": float(mod.max()) - obs_peak,
                    "lag_min": np.nan if lag is None else float(lag),
                })

    out = pd.DataFrame(rows)
    if out.empty:
        print("No event scored.")
        if spans:
            lo = min(a for a, _ in spans); hi = max(b for _, b in spans)
            print(f"  station records cover {lo:%Y-%m-%d} to {hi:%Y-%m-%d}; "
                  "events requested: "
                  + ", ".join(f"{e['name']} ({e['date']})" for e in EVENTS))
    return out


def plot_event_scores(impacted_only=False):
    """Storm by storm, averaged over stations: window RMSE, peak amplitude
    ratio and timing lag, raw against corrected. ``impacted_only`` averages
    each storm over the stations it reached only (:data:`EVENT_STATIONS`)."""
    _ensure_setup()
    import numpy as np
    import matplotlib.pyplot as plt

    df = event_scores(impacted_only=impacted_only)
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        _placeholder(ax, "Storm-by-storm scores\n\n"
                         "No event overlaps the available station records.")
        plt.tight_layout(); plt.show()
        return df

    keep = [RAW_SERIES, COR_SERIES]
    colors = {RAW_SERIES: C_RAW, COR_SERIES: C_COR}
    agg = (df.groupby(["event", "series"])
             .agg(rmse=("rmse", "mean"), peak_ratio=("peak_ratio", "mean"),
                  lag=("lag_min", "mean"))
             .reset_index())
    order = [e["name"] for e in EVENTS if e["name"] in set(agg["event"])]
    x = np.arange(len(order)); w = 0.38

    panels = [("rmse", "RMSE over ±24 h (m)", None),
              ("peak_ratio", "Peak amplitude ratio (model / observed)", 1.0),
              ("lag", "Timing lag at the peak (min)", 0.0)]
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9), sharex=True)
    for ax, (col, label, guide) in zip(axes, panels):
        for j, srs in enumerate(keep):
            v = [float(agg[(agg.event == e) & (agg.series == srs)][col].mean())
                 for e in order]
            pos = x + (j - 0.5) * w
            if col == "peak_ratio":
                # a ratio is read against 1, so stems from that line rather
                # than bars from zero
                ax.vlines(pos, 1.0, v, color=colors[srs], lw=6, alpha=0.85)
                ax.plot(pos, v, "o", ms=6, color=colors[srs], label=srs)
            else:
                ax.bar(pos, v, w, label=srs, color=colors[srs])
        if guide is not None:
            ax.axhline(guide, color=C_OBS, lw=1, ls="--")
        if col == "peak_ratio":
            allv = agg[col].dropna().tolist() + [1.0]
            pad = max(0.05, 0.15 * (max(allv) - min(allv)))
            ax.set_ylim(min(allv) - pad, max(allv) + pad)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=9)
    axes[0].set_title("Storm by storm: window error, peak amplitude and timing")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(order, rotation=20, ha="right")
    plt.tight_layout(); plt.show()
    return df


# ---------------------------------------------------------------------------
# Track A — gauge statistics and extreme-event indicators
# ---------------------------------------------------------------------------

#: ECFAS extreme-event detection (Irazoqui Apecechea et al., 2023), surge-only
#: variant: peaks over a high percentile, declustered, then paired in time
ECFAS_Q = 0.99           # percentile threshold
ECFAS_DECLUSTER_H = 72   # meteorological independence between two events
ECFAS_MATCH_H = 24       # half-window pairing a modelled event with an observed one


def _slice_block(df, start=None, end=None):
    """Rows of ``df`` between ``start`` and ``end`` (either may be None)."""
    lo = _align_tz(start, df.index) if start is not None else None
    hi = _align_tz(end, df.index) if end is not None else None
    return df.loc[lo:hi]


def _smooth(df, minutes=EVENT_SMOOTH_MIN):
    """Centred running mean over ``minutes``, for stable peak values."""
    step = 10
    try:
        step = int(round((df.index[1] - df.index[0]).total_seconds() / 60)) or 10
    except Exception:
        pass
    n = max(1, int(minutes // step))
    return df.rolling(n, center=True, min_periods=1).mean() if n > 1 else df


def _error_stats(obs, mod):
    """BIAS, MAE, RMSE and r of ``mod`` against ``obs``, in metres, with the
    standard deviation of each series (``STD`` for ``mod``, ``STD_obs``) and
    of the error itself (``STD_err``).

    The error is ``mod - obs``, so a positive BIAS is an over-prediction. The
    two series STD are read against each other: a modelled STD below the
    observed one means the variability is under-predicted.
    """
    import numpy as np

    ok = obs.notna() & mod.notna()
    n = int(ok.sum())
    if n < 12:
        return None
    o = obs[ok].to_numpy(dtype="float64")
    m = mod[ok].to_numpy(dtype="float64")
    e = m - o
    r = (float(np.corrcoef(o, m)[0, 1])
         if o.std() > 0 and m.std() > 0 else np.nan)
    return {"n": n, "BIAS": float(e.mean()), "MAE": float(np.abs(e).mean()),
            "STD": float(m.std()), "STD_obs": float(o.std()),
            "STD_err": float(e.std()),
            "RMSE": float(np.sqrt((e * e).mean())), "r": r}


def _find_key(d, key):
    """First value stored under ``key`` at any depth of a JSON object."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        items = d.values()
    elif isinstance(d, list):
        items = d
    else:
        return None
    for v in items:
        found = _find_key(v, key)
        if found is not None:
            return found
    return None


def _event_thresholds() -> dict:
    """``{station: q95_train}`` read from the ``scores_<STATION>.json`` files.

    The threshold comes from the training block, which the validation
    Parquets do not cover, so it cannot be recomputed here.
    """
    out = {}
    if not PORTS_DIR.exists():
        return out
    for p in PORTS_DIR.glob("scores_*.json"):
        try:
            q = _find_key(json.load(open(p)), "q95_train")
        except Exception:
            continue
        if q is not None:
            out[_norm_station(_station_name(p))] = float(q)
    return out


def _station_stats(start=None, end=None):
    """Tidy table of :func:`_error_stats`, per station, regime and series."""
    import pandas as pd

    thr = _event_thresholds()
    rows, no_thr = [], []
    for name, path in station_files().items():
        try:
            df = station_frame(path)
        except Exception as exc:
            print(f"  [warn] {name}: {exc}")
            continue
        if REF_SERIES not in df.columns:
            continue
        q = thr.get(name)
        if q is None:
            no_thr.append(_pretty_station(name))
        blk = _slice_block(df, start, end)
        regimes = [("All", blk)]
        if q is not None:
            regimes.append(("Events", blk[blk[REF_SERIES] > q]))
        for regime, sub in regimes:
            for srs in (RAW_SERIES, COR_SERIES):
                if srs not in sub.columns:
                    continue
                st = _error_stats(sub[REF_SERIES], sub[srs])
                if st:
                    rows.append({"station": name, "regime": regime,
                                 "series": srs, **st})
    if no_thr:
        print("  no q95_train in scores JSON, storm regime skipped: "
              + ", ".join(no_thr))
    return pd.DataFrame(rows)


def station_stats_table(regime="All", start=None, end=None):
    """BIAS, MAE, STD, RMSE (cm) and r at each gauge, raw against corrected.

    STD is the standard deviation of each series, the observed one included
    as the reference. ``regime`` is ``"All"`` (whole validation winter) or
    ``"Events"`` (observed surge above the station's ``q95_train``).
    Stations follow the
    coastal transect order; the last row is the network median.
    """
    import pandas as pd

    df = _station_stats(start, end)
    if df.empty or regime not in set(df["regime"]):
        print(f"No station scored for regime '{regime}'.")
        return pd.DataFrame()
    df = df[df["regime"] == regime].copy()
    for k in ("BIAS", "MAE", "STD", "STD_obs", "RMSE"):
        df[k] = df[k] * 100.0
    short = {RAW_SERIES: "raw", COR_SERIES: "corrected"}
    df["series"] = df["series"].map(short)
    wide = df.pivot(index="station", columns="series",
                    values=["BIAS", "MAE", "STD", "RMSE", "r"])
    wide[("STD", "observed")] = df.groupby("station")["STD_obs"].first()
    cols = [(k, s) for k in ("BIAS", "MAE") for s in ("raw", "corrected")]
    cols += [("STD", "observed"), ("STD", "raw"), ("STD", "corrected")]
    cols += [(k, s) for k in ("RMSE", "r") for s in ("raw", "corrected")]
    wide = wide[[c for c in cols if c in wide.columns]]
    wide = wide.reindex([s for s in stations() if s in wide.index])
    wide.index = [_pretty_station(s) for s in wide.index]
    wide.loc["Network median"] = wide.median()
    return wide.round(2)


def _pot_events(sr, thr, decluster_h=ECFAS_DECLUSTER_H):
    """Peaks over ``thr``, declustered: ``[(time, value), ...]``.

    Exceedances closer than ``decluster_h`` belong to one event, whose peak
    is its maximum.
    """
    import numpy as np
    import pandas as pd

    sr = sr.dropna()
    above = sr[sr > thr]
    if above.empty:
        return []
    t = above.index.to_series()
    cid = (t.diff() > pd.Timedelta(hours=decluster_h)).cumsum().to_numpy()
    out = []
    for k in np.unique(cid):
        tk = above.index[cid == k]
        seg = sr.loc[tk[0]:tk[-1]]
        out.append((seg.idxmax(), float(seg.max())))
    return out


def ecfas_scores(q=ECFAS_Q, start=None, end=None,
                 absolute=False, detail=False):
    """ECFAS extreme-event indicators at each gauge, raw against corrected.

    Following Irazoqui Apecechea et al. (2023), surge-only variant (ECFAS
    detects events jointly on total water level and surge; the validation
    records hold the surge only): extreme events (EEs) are peaks over the
    ``q`` percentile of the observed surge over the validation winter,
    declustered over :data:`ECFAS_DECLUSTER_H`, on 60-min smoothed series.
    For each detected EE the modelled peak is read within ± ECFAS_MATCH_H:
    peak error (cm, and % of the observed peak) and timing error (min,
    positive when late) — the latter by cross-correlation, as in
    :func:`event_scores`, rather than by a difference of peak times, which a
    broad or double-peaked surge makes unstable. The model *captures* an EE
    when its own
    peak-over-threshold event falls within ± ECFAS_MATCH_H of it; a modelled
    event with no observed counterpart is *false*.

    The model threshold is its own ``q`` percentile, as in ECFAS, so the
    detection skill does not simply reflect the bias; ``absolute=True`` uses
    the observed threshold instead, closer to a fixed warning level.
    ``detail=True`` returns the per-event table rather than the summary.
    """
    import numpy as np
    import pandas as pd

    win = pd.Timedelta(hours=ECFAS_MATCH_H)
    events, summary = [], []
    for name, path in station_files().items():
        try:
            df = station_frame(path)
        except Exception as exc:
            print(f"  [warn] {name}: {exc}")
            continue
        if REF_SERIES not in df.columns:
            continue
        blk = _smooth(_slice_block(df, start, end))
        obs = blk[REF_SERIES].dropna()
        if obs.empty:
            continue
        thr_o = float(obs.quantile(q))
        ev_o = _pot_events(obs, thr_o)
        for srs in (RAW_SERIES, COR_SERIES):
            if srs not in blk.columns:
                continue
            mod = blk[srs]
            thr_m = thr_o if absolute else float(mod.quantile(q))
            ev_m = _pot_events(mod, thr_m)
            t_m = [t for t, _ in ev_m]
            n_cap = 0
            for to, vo in ev_o:
                w = mod.loc[to - win:to + win].dropna()
                if w.empty:
                    continue
                vm = float(w.max())
                ow = obs.loc[to - win:to + win]
                both = ow.index.intersection(w.index)
                lag = _lag_minutes(ow.loc[both], w.loc[both])
                cap = any(abs(t - to) <= win for t in t_m)
                n_cap += cap
                events.append({
                    "station": name, "series": srs, "time": to,
                    "obs_peak": vo, "mod_peak": vm,
                    "peak_err_cm": 100.0 * (vm - vo),
                    "peak_err_pct": 100.0 * (vm - vo) / vo if vo else np.nan,
                    "timing_min": np.nan if lag is None else float(lag),
                    "captured": bool(cap),
                })
            n_false = sum(1 for t in t_m
                          if not any(abs(t - to) <= win for to, _ in ev_o))
            summary.append({
                "station": name, "series": srs,
                "EEs detected": len(ev_o),
                "Captured (%)": 100.0 * n_cap / len(ev_o) if ev_o else np.nan,
                "False (%)": 100.0 * n_false / len(ev_m) if ev_m else np.nan,
            })

    ev = pd.DataFrame(events)
    if detail:
        return ev
    out = pd.DataFrame(summary)
    if out.empty:
        print("No extreme event detected.")
        return out
    if not ev.empty:
        agg = (ev.groupby(["station", "series"])
                 .agg(**{"Peak error (cm)": ("peak_err_cm", "mean"),
                         "Peak error (%)": ("peak_err_pct", "mean"),
                         "|Timing| (min)": ("timing_min",
                                            lambda x: x.abs().mean())})
                 .reset_index())
        out = out.merge(agg, on=["station", "series"], how="left")
    order = {s: i for i, s in enumerate(stations())}
    out["_o"] = out["station"].map(order)
    out = out.sort_values(["_o", "series"]).drop(columns="_o")
    out["station"] = out["station"].map(_pretty_station)
    out["series"] = out["series"].map({RAW_SERIES: "raw",
                                       COR_SERIES: "corrected"})
    return out.set_index(["station", "series"]).round(1)


def plot_gauge_scores(start=None, end=None):
    """The gauge scores of the validation winter in one figure.

    Six panels, stations in coastal transect order: RMSE raw against
    corrected with the storm values overlaid, the share of error removed
    overall and on storms, the bias, the mean absolute error, the standard
    deviation of the error — its spread once the bias is removed — and the
    correlation with the observed surge. The same numbers are available as
    tables from :func:`station_stats_table`.
    """
    _ensure_setup()
    import numpy as np
    import matplotlib.pyplot as plt

    df = _station_stats(start, end)
    if df.empty:
        print("No station scored.")
        return
    names = [s for s in stations() if s in set(df["station"])]
    x = np.arange(len(names))
    w = 0.38

    def val(regime, series, col, scale=100.0):
        out = []
        for n in names:
            row = df[(df.station == n) & (df.regime == regime)
                     & (df.series == series)]
            out.append(scale * float(row[col].iloc[0]) if len(row) else np.nan)
        return np.array(out)

    rmse_r = val("All", RAW_SERIES, "RMSE")
    rmse_c = val("All", COR_SERIES, "RMSE")
    ev_r = val("Events", RAW_SERIES, "RMSE")
    ev_c = val("Events", COR_SERIES, "RMSE")
    with np.errstate(invalid="ignore", divide="ignore"):
        red_all = 100 * (1 - rmse_c / rmse_r)
        red_ev = 100 * (1 - ev_c / ev_r)

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)

    ax = axes[0, 0]
    ax.bar(x - w / 2, rmse_r, w, color=C_RAW, label="Raw (all)")
    ax.bar(x + w / 2, rmse_c, w, color=C_COR, label="Corrected (all)")
    if np.isfinite(ev_r).any():
        ax.plot(x - w / 2, ev_r, "o", ms=4.5, color="#7d2230",
                label="Raw (storms)")
        ax.plot(x + w / 2, ev_c, "o", ms=4.5, color="#0d3b45",
                label="Corrected (storms)")
    ax.set_title("Surge RMSE (cm)", fontsize=10.5)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[0, 1]
    ax.bar(x - w / 2, red_all, w, color=C_COR, label="All data")
    if np.isfinite(red_ev).any():
        ax.bar(x + w / 2, red_ev, w, color="#e3a008", label="Storm events")
    ax.axhline(0.0, color="#444", lw=0.8)
    ax.set_title("Error removed by the correction (%)", fontsize=10.5)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.bar(x - w / 2, val("All", RAW_SERIES, "BIAS"), w, color=C_RAW,
           label="Raw")
    ax.bar(x + w / 2, val("All", COR_SERIES, "BIAS"), w, color=C_COR,
           label="Corrected")
    ax.axhline(0.0, color="#444", lw=0.8)
    ax.set_title("Bias, model − observation (cm)", fontsize=10.5)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.bar(x - w / 2, val("All", RAW_SERIES, "MAE"), w, color=C_RAW,
           label="Raw")
    ax.bar(x + w / 2, val("All", COR_SERIES, "MAE"), w, color=C_COR,
           label="Corrected")
    ax.set_title("Mean absolute error (cm)", fontsize=10.5)
    ax.legend(fontsize=8)

    ax = axes[2, 0]
    ax.bar(x - w / 2, val("All", RAW_SERIES, "STD_err"), w, color=C_RAW,
           label="Raw")
    ax.bar(x + w / 2, val("All", COR_SERIES, "STD_err"), w, color=C_COR,
           label="Corrected")
    ax.set_title("Standard deviation of the error (cm)", fontsize=10.5)
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    ax.bar(x - w / 2, val("All", RAW_SERIES, "r", 1.0), w, color=C_RAW,
           label="Raw")
    ax.bar(x + w / 2, val("All", COR_SERIES, "r", 1.0), w, color=C_COR,
           label="Corrected")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Correlation with the observed surge", fontsize=10.5)
    ax.legend(fontsize=8, loc="lower right")

    for ax in axes.ravel():
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[2]:
        ax.set_xticks(x)
        ax.set_xticklabels([_pretty_station(n) for n in names],
                           rotation=45, ha="right", fontsize=8.5)
    fig.suptitle("Surge errors at each gauge over the validation winter",
                 fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()


def plot_ecfas_scores(q=ECFAS_Q, start=None, end=None, absolute=False):
    """The ECFAS indicators of :func:`ecfas_scores` as four panels.

    Detection on the left — share of observed extreme events captured, share
    of modelled events with no observed counterpart — and accuracy on the
    right: peak error as a share of the observed peak, and absolute timing
    error. Stations follow the coastal transect order.
    """
    _ensure_setup()
    import numpy as np
    import matplotlib.pyplot as plt

    df = ecfas_scores(q=q, start=start, end=end, absolute=absolute)
    if df.empty:
        return
    names = list(dict.fromkeys(df.index.get_level_values(0)))
    x = np.arange(len(names))
    w = 0.38
    panels = (("Captured (%)", "Events captured (%)", 100.0),
              ("False (%)", "False events (%)", 0.0),
              ("Peak error (%)", "Peak error (% of observed peak)", 0.0),
              ("|Timing| (min)", "Absolute timing error (min)", 0.0))
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for ax, (col, title, target) in zip(axes.ravel(), panels):
        for k, (series, colour) in enumerate((("raw", C_RAW),
                                              ("corrected", C_COR))):
            vals = [df.loc[(n, series), col]
                    if (n, series) in df.index else np.nan for n in names]
            ax.bar(x + (k - 0.5) * w, vals, w, color=colour,
                   label=f"Tolosa-SW ({series})")
        if col in ("Peak error (%)", "|Timing| (min)"):
            ax.axhline(0.0, color="#444", lw=0.8)
        if col == "Captured (%)":
            ax.axhline(target, color="#444", lw=0.8, ls=":")
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[1]:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8.5)
    fig.suptitle(f"Extreme-event indicators (ECFAS), surge over the "
                 f"{100 * q:g}th percentile", fontsize=12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 0.955), frameon=False, fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.show()


# ---------------------------------------------------------------------------
# Track B — scores over the domain
# ---------------------------------------------------------------------------

def field_scores_2d(step=16):
    """Error of the 2D field over the Track B test period, raw and corrected.

    Read from the exported stores, which do not share a grid: the raw error
    is ``bias_ref`` and the corrected one ``residual``, both on the
    predictor grid, so everything is derived from those two and from the
    correction they imply, ``bias_ref - residual``. Metrics are computed
    cell by cell over time, then averaged over the cells covered at least
    80 % of the period.

    ``EV`` is the share of the reference bias variance the correction
    removes. The skill of the corrected field is the mean of the exported
    ``skill`` field, computed by the calibration chain against its own
    bias-climatology baseline, and is nil for the raw field by
    construction.
    """
    _ensure_setup()
    import numpy as np
    import pandas as pd

    try:
        ds_b, ds_r = open_field("bias_ref"), open_field("residual")
    except Exception as exc:
        print(f"  [warn] {exc}")
        return pd.DataFrame()
    da_b, da_r = ds_b[_payload_var(ds_b)], ds_r[_payload_var(ds_r)]
    nt = int(da_b.sizes["time"])
    acc = None
    for k in range(0, nt, step):
        sl = slice(k, k + step)
        b = np.asarray(da_b.isel(time=sl).values, dtype="float64")
        r = np.asarray(da_r.isel(time=sl).values, dtype="float64")
        b, r = b.reshape(b.shape[0], -1), r.reshape(r.shape[0], -1)
        if b.shape[1] != r.shape[1]:
            print(f"  [warn] bias_ref and residual are not on the same grid "
                  f"({b.shape[1]} vs {r.shape[1]} cells)")
            return pd.DataFrame()
        ok = np.isfinite(b) & np.isfinite(r)
        b, r = np.where(ok, b, 0.0), np.where(ok, r, 0.0)
        if acc is None:
            acc = {n: np.zeros(b.shape[1]) for n in
                   ("n", "b", "bb", "ab", "r", "rr", "ar")}
        acc["n"] += ok.sum(axis=0)
        acc["b"] += b.sum(axis=0)
        acc["bb"] += (b * b).sum(axis=0)
        acc["ab"] += np.abs(b).sum(axis=0)
        acc["r"] += r.sum(axis=0)
        acc["rr"] += (r * r).sum(axis=0)
        acc["ar"] += np.abs(r).sum(axis=0)

    keep = acc["n"] > 0.8 * nt
    if not keep.any():
        print("No cell covered over the period.")
        return pd.DataFrame()
    n = acc["n"][keep]
    var_b = acc["bb"][keep] / n - (acc["b"][keep] / n) ** 2
    var_r = acc["rr"][keep] / n - (acc["r"][keep] / n) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        ev = np.where(var_b > 0, 1.0 - var_r / var_b, np.nan)
    rows = {
        "Tolosa-SW raw": {
            "RMSE (cm)": 100 * float(np.nanmean(np.sqrt(acc["bb"][keep] / n))),
            "MAE (cm)": 100 * float(np.nanmean(acc["ab"][keep] / n)),
            "EV": 0.0, "Skill": 0.0,
        },
        "Tolosa-SW corrected": {
            "RMSE (cm)": 100 * float(np.nanmean(np.sqrt(acc["rr"][keep] / n))),
            "MAE (cm)": 100 * float(np.nanmean(acc["ar"][keep] / n)),
            "EV": float(np.nanmean(ev)),
        },
    }
    try:
        ds_k = open_field("skill")
        rows["Tolosa-SW corrected"]["Skill"] = float(np.nanmean(
            np.asarray(ds_k[_payload_var(ds_k)].values, dtype="float64")))
    except Exception as exc:
        print(f"  [warn] skill: {exc}")
        rows["Tolosa-SW corrected"]["Skill"] = float("nan")
    out = pd.DataFrame(rows).T[["RMSE (cm)", "MAE (cm)", "EV", "Skill"]]
    out = out.rename(columns={"EV": "EV (bias variance removed)"})
    out.index.name = f"{int(keep.sum())} cells, {nt} days"
    return out.round(3)


# ---------------------------------------------------------------------------
# Track B — field stores
# ---------------------------------------------------------------------------

def list_fields() -> list:
    """Names of the field files under :data:`FIELDS_DIR`."""
    if not FIELDS_DIR.exists():
        return []
    # names starting with "_" are support files (the cached mesh), not fields
    return sorted(p.stem for p in FIELDS_DIR.glob(f"*{FIELD_EXT}")
                  if not p.stem.startswith("_"))


def open_field(name):
    """Open ``<FIELDS_DIR>/<name>.nc``; each file carries its own mesh.

    Opened without dask on purpose. xarray's lazy indexing then reads only the
    slab that is asked for — one date for a map, a window for the storm table —
    whether the variable is chunked on disk or stored contiguously. Through
    dask, a contiguous variable would be a single chunk, and reading one date
    would pull the whole record into memory.
    """
    path = FIELDS_DIR / f"{name}{FIELD_EXT}"
    if not path.exists():
        raise FileNotFoundError(
            f"field file not found: {path} "
            f"(available: {', '.join(list_fields()) or 'none'})")
    import xarray as xr
    return xr.open_dataset(path)


def _payload_var(ds, prefer=None):
    """Name of the data variable to plot in a single-field store.

    The stores are one-variable-per-store, but the variable inside is not
    necessarily named after the store, so pick the first data variable that is
    not mesh topology or a coordinate.
    """
    names = list(getattr(ds, "data_vars", ds.variables))
    if prefer and prefer in names:
        return prefer
    for n in names:
        if n in _NON_PAYLOAD:
            continue
        if getattr(ds[n], "ndim", 1) >= 1:
            return n
    raise KeyError(f"no plottable variable found (saw: {', '.join(names)})")


def _mesh_names(ds):
    """``(connectivity, node x, node y)`` variable names in ``ds``, or None.

    The UGRID convention is read first: the variable carrying
    ``cf_role = "mesh_topology"`` names its connectivity in
    ``face_node_connectivity`` and its node coordinates in ``node_coordinates``,
    whatever those variables are called. Only when no such variable exists are
    the usual names tried.
    """
    names = set(ds.variables)
    for v in names:
        a = ds[v].attrs
        if a.get("cf_role") != "mesh_topology" and "face_node_connectivity" not in a:
            continue
        conn = a.get("face_node_connectivity")
        coords = str(a.get("node_coordinates", "")).split()
        if conn in names and len(coords) == 2 and set(coords) <= names:
            return conn, coords[0], coords[1]
    conn = next((v for v in ("Mesh2_face_nodes", "mesh2d_face_nodes",
                             "face_nodes", "nv") if v in names), None)
    x = next((v for v in ("Mesh2_node_x", "mesh2d_node_x", "node_lon", "node_x",
                          "lon") if v in names), None)
    y = next((v for v in ("Mesh2_node_y", "mesh2d_node_y", "node_lat", "node_y",
                          "lat") if v in names), None)
    return (conn, x, y) if conn and x and y else None


def _mesh_from(ds):
    """Build a matplotlib Triangulation from the UGRID topology in ``ds``.

    Purely triangular mesh; the connectivity may be stored as
    ``(three, nelements)`` and start at 0 or 1. Raises a ValueError naming the
    variables actually present when no mesh can be found, rather than letting
    matplotlib fail later on an unrelated-looking unpacking error.
    """
    import numpy as np
    from matplotlib.tri import Triangulation

    found = _mesh_names(ds)
    if found is None:
        listing = ", ".join(f"{v}{tuple(ds[v].dims)}" for v in ds.variables)
        raise ValueError(f"no triangular mesh found (no UGRID mesh_topology "
                         f"variable, no known connectivity name); variables: "
                         f"{listing}")
    fn_name, lon_name, lat_name = found

    fn = ds[fn_name]
    tri = np.asarray(fn.values)
    # An integer connectivity carrying a _FillValue is decoded by xarray as
    # float with NaN; cast back when it is complete, fail clearly when it is not
    if tri.dtype.kind == "f":
        if not np.isfinite(tri).all():
            raise ValueError(f"{fn_name} holds missing node indices; "
                             "the mesh must be purely triangular")
        tri = tri.astype("int64")
    if tri.shape[0] == 3 and tri.shape[1] != 3:      # (three, nelements)
        tri = tri.T
    start = int(fn.attrs.get("start_index", 1))
    tri = tri.astype("int64") - start
    return Triangulation(np.asarray(ds[lon_name].values),
                         np.asarray(ds[lat_name].values), tri)


def _cmap(name):
    """Resolve a cmocean colormap by name, falling back to a matplotlib one.

    ``cmocean`` maps are perceptually uniform and are the house standard for
    ocean fields; the fallback keeps the module usable where it is not
    installed.
    """
    import matplotlib.pyplot as plt
    try:
        import cmocean
        return getattr(cmocean.cm, name)
    except Exception:
        return plt.get_cmap(_CMAP_FALLBACK.get(name, name))


def _group_of(source):
    """Name of the colour group a source belongs to, or None."""
    for gname, g in MAP_GROUPS.items():
        if source in g["sources"]:
            return gname
    return None


def _used_extent(triang, pad=0.02):
    """Bounding box of the nodes actually referenced by the connectivity.

    Matters when the element dimension has been cropped while ``node_lon`` /
    ``node_lat`` were left complete — the correct arrangement, since cropping
    the nodes too would mean reindexing the whole connectivity. matplotlib's
    ``tripcolor`` derives the data limits from the *full* node arrays, so
    without this the map would be framed on the uncropped domain and a large
    part of every PNG would be empty.
    """
    import numpy as np
    # min/max over the referenced nodes: np.unique would sort 3*nelements
    # indices for the same bounding box, ~20x slower on a 2.5M-element mesh
    tri = triang.triangles
    xs, ys = triang.x[tri], triang.y[tri]
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return (x0 - dx, x1 + dx, y0 - dy, y1 + dy)


# ---------------------------------------------------------------------------
# Track B — skill over the domain
# ---------------------------------------------------------------------------

def plot_skill_map():
    """2D skill score over the Track B test period, point by point.

    The colour scale is centred on zero but not symmetric: its bounds are the
    1st and 99th percentiles, zero being held at the neutral colour by a
    two-slope norm. Forcing ``±max`` would spend half the range on negative
    values that barely occur.
    """
    _ensure_setup()
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    try:
        ds = open_field("skill")
    except Exception as exc:
        fig, ax = plt.subplots(figsize=(8, 3))
        _placeholder(ax, f"2D skill map\n\n{exc}")
        plt.tight_layout(); plt.show()
        return

    var = _payload_var(ds)
    # masked arrays and fill values reach here from the store; coerce once
    values = np.squeeze(np.ma.filled(np.ma.masked_invalid(
        np.asarray(ds[var].values, dtype="float64")), np.nan))
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        _placeholder(ax, f"2D skill map\n\n'{var}' holds no finite value.")
        plt.tight_layout(); plt.show()
        return

    lo = min(float(np.percentile(finite, 1)), -0.02)
    hi = max(float(np.percentile(finite, 99)), 0.02)
    norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)

    fig, ax = plt.subplots(figsize=(7.4, 6.8))
    try:
        coll, _ = _draw_field(ax, ds, values, SKILL_CMAP, norm=norm)
    except Exception as exc:
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 3))
        _placeholder(ax, f"2D skill map\n\n{exc}")
        plt.tight_layout(); plt.show()
        return
    fig.colorbar(coll, ax=ax, shrink=0.85, label="skill score")
    ax.set_title("2D skill of the correction over the test period", pad=22)
    ax.text(0.5, 1.005,
            f"mean {finite.mean():+.3f} · median {np.median(finite):+.3f} · "
            f"{100.0 * float((finite > 0).mean()):.0f} % of the domain above 0",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5,
            color="#666")
    plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------------
# Track B — storm dates, for the external diagnostics
# ---------------------------------------------------------------------------

def storm_peak_times() -> dict:
    """``{storm: time}`` of the observed surge peak, to date the 2D windows.

    A gridded field has no peak of its own, so each storm is dated from the
    gauges: the median time of the observed peaks at the stations it reached
    (:data:`EVENT_STATIONS`), each located as in :func:`event_scores`. A storm
    with no gauge record keeps its nominal date.
    """
    import pandas as pd

    files = station_files()
    out = {}
    for ev in EVENTS:
        peaks = []
        for s in EVENT_STATIONS.get(ev["name"], ()):
            name = _norm_station(s)
            if name not in files:
                continue
            try:
                df = station_frame(files[name])
            except Exception:
                continue
            if REF_SERIES not in df.columns:
                continue
            t0 = _align_tz(ev["date"], df.index)
            tp = _peak_time(df.loc[t0 - pd.Timedelta(hours=EVENT_SEARCH_H):
                                   t0 + pd.Timedelta(hours=EVENT_SEARCH_H)]
                            [REF_SERIES])
            if tp is None:
                continue
            tp = pd.Timestamp(tp)
            if tp.tzinfo is not None:
                tp = tp.tz_convert("UTC").tz_localize(None)
            peaks.append(tp)
        out[ev["name"]] = (pd.Series(peaks).median() if peaks
                           else pd.Timestamp(ev["date"]))
    return out


# ---------------------------------------------------------------------------
# Track B — map cache and comparator
# ---------------------------------------------------------------------------

def _slice_values(ds, var, date):
    import numpy as np
    da = ds[var]
    if "time" in getattr(da, "dims", ()):
        da = da.sel(time=str(date), method="nearest")
        if "time" in getattr(da, "dims", ()):       # a whole-day slice
            da = da.mean("time")
    return np.asarray(da.values).squeeze()


def _auto_vlim(values):
    import numpy as np
    finite = values[np.isfinite(values)]
    m = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
    return (-m, m) if m > 0 else (-1.0, 1.0)


def _grid_coords(ds):
    """``(lon, lat)`` coordinate names of a regular grid, or None."""
    lon = next((v for v in ("lon", "longitude", "nav_lon") if v in ds.variables), None)
    lat = next((v for v in ("lat", "latitude", "nav_lat") if v in ds.variables), None)
    return (lon, lat) if lon and lat else None


def _field_mesh(ds, values):
    """The dataset holding the mesh ``values`` live on, or None.

    The field's own topology when it has one; otherwise, for values indexed by
    element alone, the cached :data:`MESH_FILE` — but only if its face count
    equals the number of values, since a mesh of another size would put every
    value in the wrong place without any error.
    """
    import numpy as np

    if _mesh_names(ds) is not None:
        return ds
    if np.ndim(values) != 1 or not MESH_FILE.exists():
        return None
    import xarray as xr
    mesh = xr.open_dataset(MESH_FILE)
    found = _mesh_names(mesh)
    if found is None:
        return None
    conn = mesh[found[0]]
    n_faces = max(conn.shape) if min(conn.shape) <= 4 else conn.shape[0]
    if n_faces != np.size(values):
        raise ValueError(f"{MESH_FILE.name} has {n_faces} faces but the field "
                         f"has {np.size(values)} values: not the same mesh")
    return mesh


def _draw_field(ax, ds, values, cmap, **scale):
    """Draw ``values`` on the field's own geometry; return an updater.

    A triangular mesh is drawn with ``tripcolor`` — from the field's own UGRID
    topology, or from the cached mesh for a field indexed by element alone — and
    a regular lon/lat grid with ``pcolormesh``. ``scale`` is ``vmin``/``vmax``
    or ``norm``. The returned function swaps the data of the same collection,
    so a sequence of dates pays for the geometry once.
    """
    import numpy as np

    mesh_ds = _field_mesh(ds, values)
    if mesh_ds is not None:
        triang = _mesh_from(mesh_ds)
        coll = ax.tripcolor(triang, facecolors=np.asarray(values).ravel(),
                            cmap=_cmap(cmap), shading="flat", **scale)
        x0, x1, y0, y1 = _used_extent(triang)
        ax.set_aspect("equal", adjustable="box")
        update = lambda v: coll.set_array(np.asarray(v).ravel())
    else:
        names = _grid_coords(ds)
        if names is None and np.ndim(values) == 1:
            raise ValueError(
                f"field indexed by element but carries no mesh, and no "
                f"{MESH_FILE.name} in {MESH_FILE.parent}. Create it once with: "
                f"python convert_fields_to_netcdf.py --mesh-source "
                f"<file holding the Tolosa-SW mesh>")
        if names is None:
            _mesh_from(ds)          # raises, listing what the file contains
        lon = np.asarray(ds[names[0]].values, dtype="float64")
        lat = np.asarray(ds[names[1]].values, dtype="float64")
        # data stored (lon, lat) rather than (lat, lon): transpose once here
        flip = (lon.ndim == 1 and np.shape(values) == (lon.size, lat.size)
                and lon.size != lat.size)
        prep = (lambda v: np.ma.masked_invalid(np.asarray(v, dtype="float64").T)) \
            if flip else (lambda v: np.ma.masked_invalid(np.asarray(v, dtype="float64")))
        coll = ax.pcolormesh(lon, lat, prep(values), cmap=_cmap(cmap),
                             shading="auto", **scale)
        x0, x1 = float(np.nanmin(lon)), float(np.nanmax(lon))
        y0, y1 = float(np.nanmin(lat)), float(np.nanmax(lat))
        # degrees of longitude shrink with latitude; without this the map is
        # stretched east-west by up to 70 % at the top of the domain
        ax.set_aspect(1.0 / max(0.2, np.cos(np.deg2rad(0.5 * (y0 + y1)))),
                      adjustable="box")
        update = lambda v: coll.set_array(prep(v))
    coll.set_rasterized(True)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    return coll, update


def _open_axes(ds, values0, vlim, cmap):
    """Figure and colour-mapped collection, built once per chunk of dates.

    The geometry — tessellation of a mesh, or the cell edges of a grid — does
    not depend on the date, so later dates only swap the data array.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(7.2, 6.0), dpi=110)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    coll, update = _draw_field(ax, ds, values0, cmap, vmin=vlim[0], vmax=vlim[1])
    fig.colorbar(coll, ax=ax, shrink=0.85, label="m")
    return fig, ax, update


def _render_task(args):
    """Worker: open one store and render a chunk of its dates."""
    source, label, dates, vlim, cmap = args
    try:
        ds = open_field(source)
        var = _payload_var(ds)
        out_dir = MAP_CACHE / source
        out_dir.mkdir(parents=True, exist_ok=True)
        first = _slice_values(ds, var, dates[0])
        fig, ax, update = _open_axes(ds, first, vlim or _auto_vlim(first), cmap)
        made = failed = 0
        for i, d in enumerate(dates):
            try:
                values = first if i == 0 else _slice_values(ds, var, d)
                update(values)
                ax.set_title(f"{label} — {d}", fontsize=11)
                fig.tight_layout()
                fig.savefig(out_dir / f"{source}_{d}.png", bbox_inches="tight")
                made += 1
            except Exception:
                failed += 1
        return source, made, failed, None
    except Exception as exc:
        return source, 0, len(dates), str(exc)


def _vlim_task(args):
    """Worker: robust amplitude of one field on one date."""
    import numpy as np
    src, date = args
    try:
        ds = open_field(src)
        v = _slice_values(ds, _payload_var(ds), date).ravel()[::VLIM_STRIDE]
        v = v[np.isfinite(v)]
        return src, (float(np.percentile(np.abs(v), VLIM_PERCENTILE))
                     if v.size else None)
    except Exception:
        return src, None


def _pool_map(fn, jobs, workers):
    """``map`` over a process pool, falling back to a single process."""
    if workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        try:
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
                return list(ex.map(fn, jobs))
        except Exception:
            pass
    return [fn(j) for j in jobs]


def _group_vlim(dates_by_src, workers):
    """One symmetric colour range per group, pooled over its sources and a
    sample of dates, cached in ``<MAP_CACHE>/_vlim.json``."""
    cache_file = MAP_CACHE / "_vlim.json"
    if cache_file.exists():
        try:
            return {g: tuple(v) for g, v in json.load(open(cache_file)).items()}
        except Exception:
            pass

    jobs = []
    for src, want in dates_by_src.items():
        if _group_of(src) is None or not want:
            continue
        step = max(1, len(want) // VLIM_SAMPLE)
        jobs += [(src, d) for d in want[::step][:VLIM_SAMPLE]]

    per_group = {}
    for src, val in _pool_map(_vlim_task, jobs, workers):
        if val is not None:
            per_group.setdefault(_group_of(src), []).append(val)
    vlims = {g: ((-max(v), max(v)) if max(v) > 0 else (-1.0, 1.0))
             for g, v in per_group.items()}
    if vlims:
        MAP_CACHE.mkdir(parents=True, exist_ok=True)
        json.dump({g: list(v) for g, v in vlims.items()}, open(cache_file, "w"))
    return vlims


def build_map_cache():
    """Render the comparator PNGs that are not yet on disk.

    Each date is rendered once and cached; the mesh is tessellated once per
    chunk of dates, and chunks are spread over the cores this process may use.
    Colour ranges are pooled per group so two panels stay comparable.
    """
    _ensure_setup()
    import pandas as pd

    try:
        workers = len(os.sched_getaffinity(0))
    except AttributeError:
        workers = os.cpu_count() or 1

    tasks = []
    for src, label in MAP_SOURCES.items():
        try:
            ds = open_field(src)
            want = sorted({d.strftime("%Y-%m-%d")
                           for d in pd.to_datetime(ds["time"].values)})
        except Exception as exc:
            print(f"  [warn] {src}: {exc}")
            continue
        todo = [d for d in want
                if not (MAP_CACHE / src / f"{src}_{d}.png").exists()]
        if todo:
            tasks.append((src, label, todo, want))
    if not tasks:
        return

    group_vlim = _group_vlim({src: want for src, _, _, want in tasks}, workers)
    n_maps = sum(len(t[2]) for t in tasks)
    per_chunk = max(1, math.ceil(n_maps / max(1, workers)))
    jobs = []
    for src, label, todo, _ in tasks:
        g = _group_of(src)
        cmap = MAP_GROUPS[g]["cmap"] if g else "balance"
        for i in range(0, len(todo), per_chunk):
            jobs.append((src, label, todo[i:i + per_chunk],
                         group_vlim.get(g), cmap))

    print(f"Rendering {n_maps} map(s) on {min(workers, len(jobs))} worker(s)…")
    failed, errors = 0, {}
    for src, made, fail, err in _pool_map(_render_task, jobs, workers):
        failed += fail
        if err:
            errors.setdefault((src, err), 0)
            errors[(src, err)] += fail
    for (src, err), n in errors.items():
        print(f"  [warn] {src} ({n} map(s)): {err}")
    if failed:
        print(f"  {failed} map(s) failed to render.")


def _discover_maps():
    """``(pngs per source and date, dates common to every non-empty source)``."""
    found = {}
    for src in MAP_SOURCES:
        hits = {}
        for p in sorted((MAP_CACHE / src).glob(f"{src}_*.png")):
            m = _DATE_RE.search(p.name)
            if m:
                hits["-".join(m.groups())] = p
        found[src] = hits
    non_empty = [s for s, h in found.items() if h]
    dates = (sorted(set.intersection(*(set(found[s]) for s in non_empty)))
             if non_empty else [])
    return found, dates


def plot_map_comparator():
    """Side-by-side 2D maps with a date slider, served from the PNG cache."""
    _ensure_setup()
    from IPython.display import HTML, display

    found, dates = _discover_maps()
    cwd = Path.cwd()
    payload_sources = []
    for src, label in MAP_SOURCES.items():
        imgs = {}
        for d in dates:
            p = found[src].get(d)
            if p is None:
                imgs[d] = None
                continue
            try:
                imgs[d] = os.path.relpath(p.resolve(), cwd)
            except ValueError:                  # different drive / mount
                imgs[d] = str(p.resolve())
        payload_sources.append({"key": src, "label": label, "imgs": imgs,
                                "missing": not found[src]})
    payload = json.dumps({"sources": payload_sources, "dates": dates,
                          "left": MAP_DEFAULT_LEFT, "right": MAP_DEFAULT_RIGHT,
                          "cache": str(MAP_CACHE)})
    display(HTML(_COMPARATOR_TMPL.replace("__PAYLOAD__", payload)))
    if not dates:
        print(f"No cached maps under {MAP_CACHE}; run build_map_cache() first.")


# ---------------------------------------------------------------------------
# Data flow
# ---------------------------------------------------------------------------

def plot_dataflow():
    """The application data-flow schematic, from :data:`DATAFLOW_PNG`."""
    from IPython.display import Image, display
    if DATAFLOW_PNG.exists():
        display(Image(str(DATAFLOW_PNG), width=900))
    else:
        print(f"Data-flow schematic not found: {DATAFLOW_PNG}")


# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------

_EXPLORER_TMPL = r"""
<link rel="stylesheet" href="https://unpkg.com/uplot@1.6.31/dist/uPlot.min.css">
<div id="xgb-wrap" style="font-family:system-ui,-apple-system,sans-serif;color:#1b1b1b;">
  <style>
    #xgb-wrap *{box-sizing:border-box}
    #xgb-wrap .row{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-bottom:10px}
    #xgb-wrap label{font-size:13px;color:#444;font-weight:500}
    #xgb-wrap select{font-size:13px;padding:6px 8px;border:1px solid #ccc;border-radius:8px;background:#fff;min-width:180px}
    #xgb-wrap .legend{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:6px 0 10px}
    #xgb-wrap .chip{display:inline-flex;align-items:center;gap:6px;font-size:13px;cursor:pointer;
        border:1px solid #ddd;border-radius:16px;padding:4px 10px;user-select:none}
    #xgb-wrap .chip.off{opacity:.4}
    #xgb-wrap .sw{width:12px;height:12px;border-radius:3px;display:inline-block}
    #xgb-wrap .hint{font-size:12px;color:#888;margin-left:auto}
    #xgb-wrap .plot{width:100%;height:380px}
    #xgb-wrap .empty{padding:40px;border:1px dashed #ccc;border-radius:8px;color:#888;font-size:13px;background:#fafafa;text-align:center}
    #xgb-wrap .stats{margin-top:12px;overflow-x:auto}
    #xgb-wrap table{border-collapse:collapse;font-size:12.5px;min-width:520px}
    #xgb-wrap th,#xgb-wrap td{padding:6px 12px;text-align:right;border-bottom:1px solid #eee}
    #xgb-wrap th:first-child,#xgb-wrap td:first-child{text-align:left}
    #xgb-wrap thead th{color:#666;font-weight:600;border-bottom:1px solid #ccc}
    #xgb-wrap td.name{display:flex;align-items:center;gap:7px}
    #xgb-wrap .win{font-size:12px;color:#888;margin:4px 0 2px}
    #xgb-wrap .best{font-weight:600;color:#1f7a8c}
  </style>

  <div class="row">
    <span><label>Station&nbsp;</label><select id="station"></select></span>
    <span class="hint">Drag to zoom · double-click to reset</span>
  </div>
  <div class="legend" id="legend"></div>
  <div class="plot" id="plot"></div>

  <div class="stats">
    <div class="win" id="win"></div>
    <table>
      <thead><tr>
        <th>Series</th><th>N</th><th>RMSE (m)</th><th>MAE (m)</th><th>Bias (m)</th><th>Corr</th>
      </tr></thead>
      <tbody id="statsbody"></tbody>
    </table>
    <div class="win" style="margin-top:6px">Statistics vs <b id="refname"></b>, computed on the visible window; recomputed on zoom.</div>
    <div class="win" id="diag" style="margin-top:4px;color:#b05a2c"></div>
  </div>
</div>

<script>
(function(){
  var MANIFEST = __PAYLOAD__;   // {station: {url}} — the series live on disk
  var DATA = {};                // fetched stations, kept for the session
  var CFG  = __CONFIG__;
  var H    = CFG.height || 380;
  var COLORS = ["#d1495b","#1f7a8c","#e09f3e","#3d348b","#2a9d8f","#9c6644"];
  // Fixed colours are keyed by series name, not by position: the observation
  // must stay black whatever else a file happens to contain, and the raw and
  // corrected series must keep the colours used by every other figure here.
  var FIXED = CFG.colors || {};
  function colorOf(name, i){ return FIXED[name] || COLORS[i % COLORS.length]; }
  var root = document.getElementById("xgb-wrap");
  var names = Object.keys(MANIFEST);
  var stationSel = root.querySelector("#station");
  var legend = root.querySelector("#legend");
  var plotEl = root.querySelector("#plot");
  var statsBody = root.querySelector("#statsbody");
  var winEl = root.querySelector("#win");
  var diagEl = root.querySelector("#diag");

  var uplot=null, current=null, visible={}, refName=null;

  function resolveRef(seriesKeys){
    if(!seriesKeys || seriesKeys.length===0) return null;
    if(CFG.ref && seriesKeys.indexOf(CFG.ref)>=0) return CFG.ref;
    for(var i=0;i<seriesKeys.length;i++){ if(/obs/i.test(seriesKeys[i])) return seriesKeys[i]; }
    return seriesKeys[0];
  }

  if(names.length===0){
    plotEl.style.display="none";
    legend.innerHTML='<div class="empty">No station series available. Check FOCCUS_DATA_ROOT / PORTS_DIR, run fh.export_station_series(), and re-run this cell.</div>';
    root.querySelector(".stats").style.display="none";
    return;
  }
  names.forEach(function(n){var o=document.createElement("option");o.value=n;o.textContent=n;stationSel.appendChild(o);});

  function ensureUplot(cb){
    if(window.uPlot) return cb();
    var s=document.createElement("script");
    s.src="https://unpkg.com/uplot@1.6.31/dist/uPlot.iife.min.js";
    s.onload=cb;
    s.onerror=function(){legend.innerHTML='<div class="empty">Could not load uPlot from CDN (offline?). Serve the library locally and point the script tag at it.</div>';};
    document.head.appendChild(s);
  }
  function seriesNames(){var s=DATA[current];return s&&s.series?Object.keys(s.series):[];}

  function buildLegend(){
    legend.innerHTML="";
    var s=DATA[current];
    if(s.error){legend.innerHTML='<div class="empty">Error reading this station: '+s.error+'</div>';return;}
    seriesNames().forEach(function(name,i){
      if(visible[name]===undefined) visible[name]=true;
      var chip=document.createElement("span");
      chip.className="chip"+(visible[name]?"":" off");
      chip.innerHTML='<span class="sw" style="background:'+colorOf(name,i)+'"></span>'+name;
      chip.onclick=function(){
        visible[name]=!visible[name];
        chip.classList.toggle("off");
        if(uplot) uplot.setSeries(i+1,{show:visible[name]});
        updateStats();
      };
      legend.appendChild(chip);
    });
  }
  function buildData(){
    var s=DATA[current], arr=[s.t];
    seriesNames().forEach(function(name){arr.push(s.series[name]);});
    return arr;
  }
  function makeOpts(){
    var ser=[{}];
    seriesNames().forEach(function(name,i){
      ser.push({label:name,stroke:colorOf(name,i),width:1.6,
                show:visible[name]!==false,spanGaps:false,
                value:(u,v)=> v==null?"–":v.toFixed(3)+" m"});
    });
    return {
      width: plotEl.clientWidth||900, height:H,
      cursor:{drag:{x:true,y:false}},
      scales:{x:{time:true}},
      axes:[
        {grid:{stroke:"#f0f0f0"},ticks:{stroke:"#ccc"},stroke:"#666"},
        {grid:{stroke:"#f0f0f0"},ticks:{stroke:"#ccc"},stroke:"#666",
         size:56,label:"Surge height (m)",labelSize:26,
         values:(u,vals)=> vals.map(function(v){return v.toFixed(2);})}
      ],
      series: ser,
      legend:{show:false},   // built-in uPlot legend off: the chips above are the only toggle
      hooks:{ setScale:[function(u){ updateStats(); }] }
    };
  }

  function fmt(x,d){ return (x==null||isNaN(x))?"–":x.toFixed(d===undefined?3:d); }
  function computeStats(){
    var s=DATA[current]; if(!s||s.error) return {rows:[],lo:null,hi:null,diag:"station error"};
    var t=s.t, ref=s.series[refName];
    var tmin=t[0], tmax=t[t.length-1];
    var lo=tmin, hi=tmax, unit="s";
    if(uplot && uplot.scales && uplot.scales.x){
      var xmin=uplot.scales.x.min, xmax=uplot.scales.x.max;
      if(xmin!=null && xmax!=null && isFinite(xmin) && isFinite(xmax) && xmax>xmin){
        var factor = 1;
        if(tmin > 0){
          var mag = xmin / tmin;                 // ~1 if seconds, ~1000 if ms
          if(mag > 100)      { factor = 1/1000; unit="ms"; }
          else if(mag < 0.01){ factor = 1000;  unit="ks"; }
        }
        lo = xmin*factor; hi = xmax*factor;
        if(lo < tmin) lo = tmin; if(hi > tmax) hi = tmax;
        if(!(hi > lo)){ lo = tmin; hi = tmax; }
      }
    }
    var rows=[], refMissing = (ref===undefined);
    seriesNames().forEach(function(name){
      if(name===refName) return;
      if(visible[name]===false) return;
      var arr=s.series[name];
      var n=0,se=0,ae=0,be=0, sx=0,sy=0,sxx=0,syy=0,sxy=0;
      for(var j=0;j<t.length;j++){
        if(t[j]<lo||t[j]>hi) continue;
        var a=ref?ref[j]:null, b=arr[j];
        if(a==null||b==null) continue;
        var e=b-a; n++; se+=e*e; ae+=Math.abs(e); be+=e;
        sx+=a; sy+=b; sxx+=a*a; syy+=b*b; sxy+=a*b;
      }
      var rmse=n?Math.sqrt(se/n):null, mae=n?ae/n:null, bias=n?be/n:null;
      var cov=n?(sxy/n-(sx/n)*(sy/n)):0;
      var vx=n?(sxx/n-(sx/n)*(sx/n)):0, vy=n?(syy/n-(sy/n)*(sy/n)):0;
      var corr=(n&&vx>0&&vy>0)?cov/Math.sqrt(vx*vy):null;
      rows.push({name:name,n:n,rmse:rmse,mae:mae,bias:bias,corr:corr});
    });
    var diag="";
    if(refMissing) diag="Reference series ‘"+refName+"’ not found in this file.";
    else if(rows.length && rows.every(function(r){return r.n===0;}))
      diag="0 overlapping points with the reference (check for all-NaN columns or misaligned timestamps). Window unit detected: "+unit+".";
    return {rows:rows,lo:lo,hi:hi,diag:diag};
  }
  function updateStats(){
    var res=computeStats();
    diagEl.textContent = res.diag || "";
    if(res.lo!=null){
      var d0=new Date(res.lo*1000), d1=new Date(res.hi*1000);
      var opt={year:"numeric",month:"short",day:"numeric",hour:"2-digit",minute:"2-digit"};
      winEl.textContent="Window: "+d0.toLocaleString(undefined,opt)+"  →  "+d1.toLocaleString(undefined,opt);
    } else winEl.textContent="";
    var best={rmse:Infinity, mae:Infinity, bias:Infinity, corr:-Infinity};
    res.rows.forEach(function(r){
      if(r.rmse!=null && r.rmse<best.rmse) best.rmse=r.rmse;
      if(r.mae !=null && r.mae <best.mae ) best.mae =r.mae;
      if(r.bias!=null && Math.abs(r.bias)<best.bias) best.bias=Math.abs(r.bias);
      if(r.corr!=null && r.corr>best.corr) best.corr=r.corr;
    });
    var nBest = res.rows.length>1;
    function cell(val, isBest, digits){
      var txt=fmt(val, digits);
      return (nBest && isBest) ? '<td class="best">'+txt+'</td>' : '<td>'+txt+'</td>';
    }
    statsBody.innerHTML="";
    var si=seriesNames();
    res.rows.forEach(function(r){
      var ci=si.indexOf(r.name);
      var tr=document.createElement("tr");
      var isR = r.rmse!=null && Math.abs(r.rmse-best.rmse)<1e-12;
      var isM = r.mae !=null && Math.abs(r.mae -best.mae )<1e-12;
      var isB = r.bias!=null && Math.abs(Math.abs(r.bias)-best.bias)<1e-12;
      var isC = r.corr!=null && Math.abs(r.corr-best.corr)<1e-12;
      tr.innerHTML='<td class="name"><span class="sw" style="background:'+colorOf(r.name,ci)+'"></span>'+r.name+'</td>'+
                   '<td>'+r.n+'</td>'+
                   cell(r.rmse, isR)+
                   cell(r.mae,  isM)+
                   cell(r.bias, isB)+
                   cell(r.corr, isC, 3);
      statsBody.appendChild(tr);
    });
    if(res.rows.length===0){
      statsBody.innerHTML='<tr><td colspan="6" style="text-align:center;color:#888">No comparable series visible (need the reference plus at least one other).</td></tr>';
    }
  }

  function render(){
    var s=DATA[current];
    if(s.error){ if(uplot){uplot.destroy();uplot=null;} plotEl.innerHTML=""; updateStats(); return; }
    if(uplot){uplot.destroy();uplot=null;}
    plotEl.innerHTML="";
    uplot=new uPlot(makeOpts(), buildData(), plotEl);
    updateStats();
  }
  function populateRefSelector(){
    var keys=seriesNames();
    refName = resolveRef(keys);
    var rn=root.querySelector("#refname"); if(rn) rn.textContent = refName || "—";
  }
  // A regular time axis travels as origin + step; rebuild it once on arrival.
  function decode(j){
    if(j.t) return j;
    if(j.t0===undefined) return j;
    var n=j.n, t0=j.t0, dt=j.dt, t=new Array(n);
    for(var i=0;i<n;i++) t[i]=t0+i*dt;
    j.t=t; return j;
  }
  function finishSelect(){
    buildLegend();
    populateRefSelector();
    ensureUplot(render);
  }
  function selectStation(n){
    current=n; visible={};
    if(DATA[n]){ finishSelect(); return; }
    legend.innerHTML='<div class="empty">Loading '+n+'…</div>';
    plotEl.innerHTML=""; statsBody.innerHTML=""; winEl.textContent="";
    var m=MANIFEST[n];
    if(!m||!m.url){ DATA[n]={error:"no series file for this station"}; finishSelect(); return; }
    fetch(m.url)
      .then(function(r){ if(!r.ok) throw new Error(r.status+" "+r.statusText); return r.json(); })
      .then(function(j){ DATA[n]=decode(j); if(current===n) finishSelect(); })
      .catch(function(e){
        DATA[n]={error:"could not load "+m.url+" ("+e.message+"). "
                      +"The file is served by path relative to the notebook, "
                      +"like the map cache."};
        if(current===n) finishSelect();
      });
  }
  stationSel.onchange=function(){selectStation(stationSel.value);};
  window.addEventListener("resize",function(){ if(uplot) uplot.setSize({width:plotEl.clientWidth||900,height:H}); });
  selectStation(names[0]);
})();
</script>
"""


_COMPARATOR_TMPL = r"""
<div id="ibi-wrap" style="font-family:system-ui,-apple-system,sans-serif;color:#1b1b1b;">
  <style>
    #ibi-wrap *{box-sizing:border-box}
    #ibi-wrap .row{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-bottom:12px}
    #ibi-wrap label{font-size:13px;color:#444;font-weight:500}
    #ibi-wrap select{font-size:13px;padding:6px 8px;border:1px solid #ccc;border-radius:8px;background:#fff;min-width:230px}
    #ibi-wrap .stage{display:flex;gap:12px;justify-content:center;align-items:flex-start;flex-wrap:wrap}
    #ibi-wrap .panel{flex:1 1 0;min-width:280px;text-align:center}
    #ibi-wrap .panel h4{font-size:13px;font-weight:500;color:#1f7a8c;margin:0 0 6px}
    #ibi-wrap img{max-width:100%;border:1px solid #eee;border-radius:8px}
    #ibi-wrap .missing{padding:40px;border:1px dashed #ccc;border-radius:8px;color:#888;font-size:13px;background:#fafafa}
    #ibi-wrap .timebar{display:flex;gap:12px;align-items:center;margin:4px 0 14px;
        padding:10px 14px;border:1px solid #eee;border-radius:10px;background:#fafafa}
    #ibi-wrap input[type=range]{flex:1 1 auto;accent-color:#1f7a8c;cursor:pointer}
    #ibi-wrap .date{font-variant-numeric:tabular-nums;font-weight:600;font-size:14px;
        color:#1f7a8c;min-width:110px;text-align:center}
    #ibi-wrap .step{border:1px solid #ccc;background:#fff;border-radius:6px;
        width:30px;height:28px;cursor:pointer;font-size:14px;line-height:1}
    #ibi-wrap .count{font-size:12px;color:#888;min-width:64px;text-align:right}
  </style>

  <div class="row">
    <span><label>Left&nbsp;</label><select id="left"></select></span>
    <span><label>Right&nbsp;</label><select id="right"></select></span>
  </div>

  <div class="timebar">
    <button class="step" id="prev" title="Previous date">&#9664;</button>
    <button class="step" id="play" title="Play / pause">&#9654;</button>
    <button class="step" id="next" title="Next date">&#9654;&#9654;</button>
    <input type="range" id="slider" min="0" value="0" step="1">
    <span class="date" id="datelbl">—</span>
    <span class="count" id="countlbl"></span>
  </div>

  <div class="stage" id="stage"></div>
</div>

<script>
(function(){
  var D = __PAYLOAD__;
  var S = D.sources, DATES = D.dates;
  var root = document.getElementById("ibi-wrap");
  var ti = 0, timer = null;

  function idxOfKey(k, fb){ for(var i=0;i<S.length;i++){ if(S[i].key===k) return i; } return fb; }

  function fill(sel, def){
    S.forEach(function(p,i){
      var o=document.createElement("option"); o.value=i; o.textContent=p.label; sel.appendChild(o);
    });
    sel.value = def;
  }
  var left=root.querySelector("#left"), right=root.querySelector("#right");
  fill(left, idxOfKey(D.left, 0)); fill(right, idxOfKey(D.right, 1));

  var slider=root.querySelector("#slider"),
      datelbl=root.querySelector("#datelbl"),
      countlbl=root.querySelector("#countlbl");
  slider.max = Math.max(0, DATES.length-1);
  slider.disabled = DATES.length < 2;

  function imgHTML(p){
    var src = p.imgs ? p.imgs[DATES[ti]] : null;
    if(!src) return '<div class="missing">No cached map for:<br><b>'+p.label+'</b>'
                    + (DATES.length ? '<br>'+DATES[ti] : '') + '</div>';
    return '<img src="'+src+'" alt="'+p.label+' '+DATES[ti]+'" loading="lazy">';
  }
  function panelHTML(p){ return '<div class="panel"><h4>'+p.label+'</h4>'+imgHTML(p)+'</div>'; }

  function render(){
    var stage=root.querySelector("#stage");
    if(!DATES.length){
      stage.innerHTML = '<div class="missing">No cached maps under <code>'+D.cache+'</code>.<br>'
                      + 'Run <code>fh.build_map_cache()</code> first.</div>';
      datelbl.textContent = "—"; countlbl.textContent = "";
      return;
    }
    datelbl.textContent  = DATES[ti];
    countlbl.textContent = (ti+1) + " / " + DATES.length;
    slider.value = ti;
    stage.innerHTML = panelHTML(S[+left.value]) + panelHTML(S[+right.value]);
  }
  function goto(i){ if(!DATES.length) return; ti = (i + DATES.length) % DATES.length; render(); }

  function stop(){ if(timer){ clearInterval(timer); timer=null; }
                   root.querySelector("#play").innerHTML="&#9654;"; }
  function play(){
    if(timer){ stop(); return; }
    if(DATES.length<2) return;
    root.querySelector("#play").innerHTML="&#10074;&#10074;";
    timer=setInterval(function(){ goto(ti+1); }, 900);
  }

  root.querySelector("#prev").onclick=function(){stop(); goto(ti-1)};
  root.querySelector("#next").onclick=function(){stop(); goto(ti+1)};
  root.querySelector("#play").onclick=play;
  slider.oninput=function(){ stop(); goto(+slider.value); };
  slider.onkeydown=function(e){
    if(e.key==="ArrowLeft"){ stop(); goto(ti-1); e.preventDefault(); }
    if(e.key==="ArrowRight"){ stop(); goto(ti+1); e.preventDefault(); }
  };
  [left,right].forEach(function(s){s.onchange=render});
  render();
})();
</script>
"""
