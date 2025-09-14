
"""
climate_pipeline.py (updated)
Index-agnostic pipeline to aggregate raster data (ASC, TIF, NetCDF)
to arbitrary admin units — explicitly selectable (VG250 levels or custom GDF).
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Dict, Union

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import xarray as xr
import rioxarray as rxr
from rasterstats import zonal_stats
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

# ----------------------------
# Configuration dataclasses
# ----------------------------
@dataclass
class SourceSpec:
    name: str                 # short identifier used as column prefix
    path: str                 # raster path (.asc, .tif, .nc, ...)
    year: Optional[str] = None
    band: int = 1             # for multi-band rasters
    var: Optional[str] = None # for NetCDF: variable name. If None, first data var
    fallback_crs: Optional[str] = None  # e.g. "EPSG:31467" if file lacks CRS
    nodata: Optional[float] = None      # override nodata if needed

@dataclass
class VG250Spec:
    gpkg_path: str
    layer_sta: str = "VG250_STA"  # Staat
    layer_krs: str = "VG250_KRS"  # kreis
    layer_gem: str = "VG250_GEM"  # Gemeine
    layer_lan: str = "VG250_LAN"  # Land
    layer_rbz: str = "VG250_RBZ"  # optional in some packages
    layer_vwg: str = "VG250_VWG"  # verwaltungsgemeinschaften
    layer_pk: str = "VG250_PK"   # Punkte für Kern der Gemeinde
    id_col: str = "ARS"           # default robust key for admin units



# ----------------------------
# Bundesländer
# ----------------------------
STATES = {
  '01': 'Schleswig-Holstein',
  '02': 'Hamburg',
  '03': 'Niedersachsen',
  '04': 'Bremen',
  '05': 'Nordrhein-Westfalen',
  '06': 'Hessen',
  '07': 'Rheinland-Pfalz',
  '08': 'Baden-Württemberg',
  '09': 'Bayern',
  '10': 'Saarland',
  '11': 'Berlin',
  '12': 'Brandenburg',
  '13': 'Mecklenburg-Vorpommern',
  '14': 'Sachsen',
  '15': 'Sachsen-Anhalt',
  '16': 'Thüringen',
  'Schleswig-Holstein'     :'01',
  'Hamburg'                :'02',
  'Niedersachsen'          :'03',
  'Bremen'                 :'04',
  'Nordrhein-Westfalen'    :'05',
  'Hessen'                 :'06',
  'Rheinland-Pfalz'        :'07',
  'Baden-Württemberg'      :'08',
  'Bayern'                 :'09',
  'Saarland'               :'10',
  'Berlin'                 :'11',
  'Brandenburg'            :'12',
  'Mecklenburg-Vorpommern' :'13',
  'Sachsen'                :'14',
  'Sachsen-Anhalt'         :'15',
  'Thüringen'              :'16',  
}

KUERZEL = {
  'Schleswig-Holstein'     :'SH',
  'Hamburg'                :'HH',
  'Niedersachsen'          :'NI',
  'Bremen'                 :'HB',
  'Nordrhein-Westfalen'    :'NW',
  'Hessen'                 :'HE',
  'Rheinland-Pfalz'        :'RP',
  'Baden-Württemberg'      :'BW',
  'Bayern'                 :'BY',
  'Saarland'               :'SL',
  'Berlin'                 :'BE',
  'Brandenburg'            :'BB',
  'Mecklenburg-Vorpommern' :'MV',
  'Sachsen'                :'SN',
  'Sachsen-Anhalt'         :'SA',
  'Thüringen'              :'TH',  
  '01':'SH',
  '02':'HH',
  '03':'NI',
  '04':'HB',
  '05':'NW',
  '06':'HE',
  '07':'RP',
  '08':'BW',
  '09':'BY',
  '10':'SL',
  '11':'BE',
  '12':'BB',
  '13':'MV',
  '14':'SN',
  '15':'SA',
  '16':'TH',  
  'SH':'01',
  'HH':'02',
  'NI':'03',
  'HB':'04',
  'NW':'05',
  'HE':'06',
  'RP':'07',
  'BW':'08',
  'BY':'09',
  'SL':'10',
  'BE':'11',
  'BB':'12',
  'MV':'13',
  'SN':'14',
  'SA':'15',
  'TH':'16',  
}


# ----------------------------
# CRS helpers
# ----------------------------
def _infer_epsg_from_xy(xmin: float, xmax: float, ymin: float, ymax: float) -> str:
    """Heuristic for common German grids. Falls back to EPSG:4326 if nothing matches."""
    if -180 <= xmin <= 180 and -180 <= xmax <= 180 and -90 <= ymin <= 90 and -90 <= ymax <= 90:
        return "EPSG:4326"
    if 3_000_000 <= xmin < 4_000_000:
        return "EPSG:31467"  # GK3
    if 4_000_000 <= xmin < 5_000_000:
        return "EPSG:31468"  # GK4
    if 5_000_000 <= xmin < 6_000_000:
        return "EPSG:31469"  # GK5
    if 200_000 <= xmin <= 900_000 and 5_000_000 <= ymin <= 6_500_000:
        return "EPSG:25832"  # UTM32N
    return "EPSG:4326"

def _ensure_crs(da: xr.DataArray, fallback_crs: Optional[str] = None) -> xr.DataArray:
    if da.rio.crs is not None:
        return da
    try:
        xmin, xmax = float(da.x.min()), float(da.x.max())
        ymin, ymax = float(da.y.min()), float(da.y.max())
        epsg = fallback_crs or _infer_epsg_from_xy(xmin, xmax, ymin, ymax)
    except Exception:
        epsg = fallback_crs or "EPSG:4326"
    return da.rio.write_crs(epsg)

# ----------------------------
# Raster loading
# ----------------------------
def open_raster_any(spec: SourceSpec) -> xr.DataArray:
    """
    Open ASC/TIF/NetCDF. Return a 2D DataArray with x/y dims and CRS.
    - ASC/TIF: via rioxarray.open_rasterio -> [band, y, x]; select band
    - NetCDF: via xarray; pick variable and ensure dims named x/y
    """
    ext = os.path.splitext(spec.path)[1].lower()
    if ext in {".asc", ".tif", ".tiff"}:
        da = rxr.open_rasterio(spec.path, masked=True)
        if "band" in da.dims:
            da = da.sel(band=spec.band)
        da = da.squeeze()
        da = _ensure_crs(da, spec.fallback_crs)
        if spec.nodata is not None:
            da = da.rio.write_nodata(spec.nodata)
        return da

    if ext in {".nc", ".nc4", ".cdf"}:
        ds = xr.open_dataset(spec.path)
        var = spec.var
        if var is None:
            data_vars = list(ds.data_vars)
            if not data_vars:
                raise ValueError("NetCDF contains no data variables.")
            var = data_vars[0]
        da = ds[var]
        if "time" in da.dims:
            da = da.isel(time=0)
        ren = {}
        if "lon" in da.dims: ren["lon"] = "x"
        if "latitude" in da.dims: ren["latitude"] = "y"
        if "longitude" in da.dims: ren["longitude"] = "x"
        if "lat" in da.dims: ren["lat"] = "y"
        da = da.rename(ren)
        da = da.rio.write_crs(spec.fallback_crs or "EPSG:4326")
        out_path = _temp_path(f"{spec.name}_nc_to_tif.tif")
        da.rio.to_raster(out_path)
        da2 = rxr.open_rasterio(out_path, masked=True)
        if "band" in da2.dims:
            da2 = da2.sel(band=1).squeeze()
        return da2

    raise ValueError(f"Unsupported raster format: {ext}")

# ----------------------------
# Temporary files and saving
# ----------------------------
def _temp_path(name: str) -> str:
    tmp_dir = os.path.join("tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    return os.path.join(tmp_dir, name)

def write_geotiff(da: xr.DataArray, out_name: Optional[str] = None) -> Tuple[str, Optional[float]]:
    if out_name is None:
        out_name = f"{uuid.uuid4().hex}.tif"
    out_path = _temp_path(out_name)
    da.rio.to_raster(out_path)
    with rasterio.open(out_path) as src:
        nodata = src.nodata
    return out_path, nodata

def save_map(fig: plt.Figure, filename: str, out_dir: str = "exports", *, dpi: int = 300) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path

# ----------------------------
# Vector loading and helpers
# ----------------------------
def load_vg250(spec: VG250Spec) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    kreise = gpd.read_file(spec.gpkg_path, layer=spec.layer_krs)
    laender = gpd.read_file(spec.gpkg_path, layer=spec.layer_lan)
    return kreise, laender

def load_level(spec: VG250Spec, level: str) -> gpd.GeoDataFrame:
    """Load a specific admin level from VG250 by string code: 'STA', 'KRS', 'GEM', 'LAN', 'RBZ'."""
    level = level.upper()
    if level == "STA":           # Staat
        layer = spec.layer_sta   
    elif level == "KRS":           # Landkreis
        layer = spec.layer_krs
    elif level == "GEM":         # Gemeinde
        layer = spec.layer_gem
    elif level == "LAN":         # Bundesland
        layer = spec.layer_lan
    elif level == "RBZ":         # Regierungsbezirk
        layer = spec.layer_rbz
    elif level == "VWG":         # verwaltungsgemeinschaft
        layer = spec.layer_vwg
    elif level == "PK":
        layer = spec.layer_pk
    else:
        raise ValueError(f"Unknown level '{level}'. Use one of: KRS, GEM, LAN, RBZ, VWG.")
    
    rf = gpd.read_file(spec.gpkg_path, layer=layer)
    # um duplikate zu vermeiden, nur geofaktor 4
    # siehe Anlage B für informationen
    if layer != spec.layer_pk:
        rf = rf[rf['GF']==4]
    
    return rf

def detect_id_col(gdf: gpd.GeoDataFrame, preferred: str = "AGS") -> str:
    candidates = [preferred, "AGS", "ARS", "RS", "RS_0", "AGS_0", "KRS_CODE", "NUTS"]
    for c in candidates:
        if c in gdf.columns:
            return c
    # conservative fallback
    for c in gdf.columns:
        if c != "geometry":
            try:
                if gdf[c].dtype == object and gdf[c].notna().mean() > 0.95:
                    return c
            except Exception:
                pass
    raise KeyError(f"No likely ID column found. Columns: {list(gdf.columns)}")

def filter_admin(gdf: gpd.GeoDataFrame, column: str, include: Optional[Sequence[str]] = None,
                 exclude: Optional[Sequence[str]] = None) -> gpd.GeoDataFrame:
    out = gdf
    if include is not None:
        out = out[out[column].isin(include)]
    if exclude is not None:
        out = out[~out[column].isin(exclude)]
    return out.copy()

# ----------------------------
# Zonal statistics core
# ----------------------------
def zonal_stats_to_gdf(
    raster_tif: str,
    regions_gdf: gpd.GeoDataFrame,
    stats: Sequence[str] = ("mean",),
    all_touched: bool = False
) -> Tuple[gpd.GeoDataFrame, Dict[str, np.ndarray]]:
    with rasterio.open(raster_tif) as src:
        poly = regions_gdf.to_crs(src.crs)
        zs = zonal_stats(poly, raster_tif, stats=list(stats), all_touched=all_touched, nodata=src.nodata)
    out = poly.copy()
    arrays = {s: np.array([d.get(s) for d in zs], dtype=float) for s in stats}
    for s, arr in arrays.items():
        out[s] = arr
    return out, arrays

# ----------------------------
# Multi-source processing (explicit level or custom GDF)
# ----------------------------
def process_sources(
    sources: Iterable[SourceSpec],
    vg: Optional[VG250Spec] = None,
    stats: Sequence[str] = ("mean",),
    all_touched: bool = False,
    *,  # explicit-only parameters below
    target_level: str = "KRS",                 # "KRS" (default), "GEM", "LAN", "RBZ"
    target_gdf: Optional[gpd.GeoDataFrame] = None,
    target_id_col: Optional[str] = None,
    target_filter: Optional[Dict[str, Sequence[str]]] = None,  # e.g. {"BEZ": ["Landkreis"]}
) -> gpd.GeoDataFrame:
    """
    Aggregate multiple rasters to a chosen admin level or a custom region set.

    Use either:
    - target_gdf (and target_id_col)  -> custom polygons you provide, OR
    - vg + target_level               -> load polygons from VG250Spec

    Returns: GeoDataFrame (EPSG:4326) with columns {id_col, geometry, <name>_<stat>...}
    """
    # 1) Resolve target polygons
    if target_gdf is not None:
        regions = target_gdf.copy()
        id_col = 'AGS' # target_id_col or detect_id_col(regions, preferred="AGS")
    else:
        if vg is None:
            raise ValueError("Either provide target_gdf + target_id_col, or a VG250Spec (vg).")
        regions = load_level(vg, target_level)
        id_col = target_id_col or detect_id_col(regions, preferred=vg.id_col)
    # optional filter (e.g., only Landkreise)
    if target_filter:
        for col, vals in target_filter.items():
            regions = filter_admin(regions, col, include=vals)

    # normalize id column name to id_col and cast to string for stable joins
    if id_col not in regions.columns:
        raise KeyError(f"ID column '{id_col}' not found in target polygons.")
    if (vg is not None) and (vg.id_col != id_col):
        regions = regions.rename(columns={id_col: vg.id_col})
        id_col = vg.id_col
    try:
        regions[id_col] = regions[id_col].astype(str)
    except Exception:
        pass

    gdf_accum: Optional[gpd.GeoDataFrame] = None

    # 2) Loop over sources and compute zonal stats
    for spec in sources:
        da = open_raster_any(spec)
        da = _ensure_crs(da, spec.fallback_crs)
        tif_path, _ = write_geotiff(da, out_name=f"{spec.name}.tif")

        tmp_gdf, _ = zonal_stats_to_gdf(tif_path, regions, stats=stats, all_touched=all_touched)
        if id_col not in tmp_gdf.columns:
            raise KeyError(f"Expected ID column '{id_col}' missing after zonal_stats. Columns: {list(tmp_gdf.columns)}")

        rename_map = {s: f"{spec.name}_{s}" for s in stats}
        tmp_gdf = tmp_gdf.rename(columns=rename_map)
        keep_cols = [id_col, "geometry"] + list(rename_map.values())
        tmp_gdf = tmp_gdf[[c for c in keep_cols if c in tmp_gdf.columns]].copy()

        try:
            tmp_gdf[id_col] = tmp_gdf[id_col].astype(str)
        except Exception:
            pass

        if gdf_accum is None:
            gdf_accum = tmp_gdf
        else:
            tmp_gdf = tmp_gdf.set_crs(gdf_accum.crs) if gdf_accum.crs is not None else tmp_gdf
            gdf_accum = gdf_accum.merge(tmp_gdf.drop(columns=["geometry"]), on=id_col, how="left")

    assert gdf_accum is not None
    return gdf_accum.to_crs(4326)

# ----------------------------
# Post-processing utilities
# ----------------------------
def add_percent_change(gdf: gpd.GeoDataFrame, newer_col: str, older_col: str, out_col: Optional[str] = None) -> gpd.GeoDataFrame:
    if out_col is None:
        out_col = f"{newer_col}_vs_{older_col}_pct"
    gdf[out_col] = (gdf[newer_col] - gdf[older_col]) / gdf[older_col] * 100.0
    return gdf

def attach_excel(
    gdf: gpd.GeoDataFrame,
    excel_path: str,
    *,
    sheet: Optional[str] = None,
    left_on: str = "AGS",
    right_on: Optional[str] = None
) -> gpd.GeoDataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet)
    if right_on is None:
        right_on = left_on
    df[right_on] = df[right_on].astype(gdf[left_on].dtype, errors="ignore")
    merged = gdf.merge(df, how="left", left_on=left_on, right_on=right_on)
    return gpd.GeoDataFrame(merged, geometry=gdf.geometry, crs=gdf.crs)

# ----------------------------
# Classification helpers (generic)
# ----------------------------
def classify_discrete(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    *,
    bins: Sequence[float],
    labels: Sequence[str],
    class_col: Optional[str] = None,
) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    if class_col is None:
        class_col = f"{value_col}_class"
    gdf[class_col] = pd.cut(
        gdf[value_col].astype(float), bins=bins, labels=labels, right=False, include_lowest=True
    )
    return gdf

def build_colormap_and_norm(
    *,
    bins: Sequence[float],
    labels: Sequence[str],
    colors: Optional[Sequence[str]] = None,
) -> Tuple[ListedColormap, BoundaryNorm, Sequence[str]]:
    n = len(labels)
    if colors is None:
        base = plt.cm.viridis(np.linspace(0, 1, n))
        cmap = ListedColormap(base)
    else:
        if len(colors) != n:
            raise ValueError("Length of colors must match labels.")
        cmap = ListedColormap(colors)
    norm = BoundaryNorm(bins, ncolors=len(cmap.colors), clip=True)
    return cmap, norm, labels

# ----------------------------
# Plotting (generic)
# ----------------------------
def plot_choropleth_discrete(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    *,
    bins: Sequence[float],
    labels: Sequence[str],
    class_col: Optional[str] = None,
    colors: Optional[Sequence[str]] = None,
    laender: Optional[gpd.GeoDataFrame] = None,
    states : Optional[list] = None,
    title: Optional[str] = None,
    legend_offset_cm: float = 1.2,
    figsize: Tuple[float, float] = (8, 9),
) -> Tuple[plt.Figure, plt.Axes]:

    if class_col is None:
        class_col = f"{value_col}_class"
    if class_col not in gdf.columns:
        gdf = classify_discrete(gdf, value_col, bins=bins, labels=labels, class_col=class_col)

    cmap, norm, labels_used = build_colormap_and_norm(bins=bins, labels=labels, colors=colors)

    # Auswahl von Bundesländern
    if states:
        gdf = gdf[gdf['ARS'].str[:2].isin(states)].copy()
    
    if laender is not None and states:
        laender = laender[laender['ARS'].str[:2].isin(states)].copy()
        
    fig, ax = plt.subplots(figsize=figsize)
    gdf.plot(column=value_col, cmap=cmap, norm=norm, linewidth=0, ax=ax, legend=False)

    if laender is not None:
        laender.to_crs(gdf.crs).boundary.plot(ax=ax, linewidth=0.6, color="#ffffff", alpha=0.6)

    present = list(gdf[class_col].dropna().unique())
    present_sorted = [lab for lab in labels_used if lab in present]
    patches = [Patch(facecolor=cmap.colors[labels_used.index(lab)], edgecolor="none", label=lab) for lab in present_sorted]

    cm = -legend_offset_cm
    inch = cm / 2.54
    fig_w, fig_h = fig.get_size_inches()
    pos = ax.get_position()
    axes_w_in = fig_w * pos.width
    dx = inch / axes_w_in

    ax.legend(
        handles=patches,
        loc="lower right",
        fontsize=8,
        title_fontsize=9,
        frameon=True,
        bbox_to_anchor=(1 - dx, 0),
        bbox_transform=ax.transAxes,
    )

    if title:
        ax.set_title(title)
    ax.set_axis_off()
    return fig, ax

def plot_choropleth_continuous(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    *,
    cmap_name: str = "viridis",
    laender: Optional[gpd.GeoDataFrame] = None,
    states : Optional[list] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 9),
    norm = None
) -> Tuple[plt.Figure, plt.Axes]:
    laender.to_crs(gdf.crs)

    # Auswahl von Bundesländern
    if states:
        gdf = gdf[gdf['ARS'].str[:2].isin(states)].copy()
    
    if laender is not None and states:
        laender = laender[laender['ARS'].str[:2].isin(states)].copy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    gdf.plot(
        column=value_col, 
        cmap=cmap_name,
        linewidth=0,
        alpha = 1.0,
        ax=ax,
        legend=True,
        norm = norm
    )
    
    if laender is not None:
        laender.boundary.plot(
            ax=ax,
            linewidth=0.8,
            color="#ffffff",
            alpha=0.6
        )
    
    if title:
        ax.set_title(title, loc = "center")

    if states:
        tmp = ""
        for state in states:
            tmp = tmp + STATES[state]+", "
        
        ax.text(0.5, -0.12, "Ausschnitt: "+tmp,
            transform=ax.transAxes, ha="center", va="bottom", fontsize=10)
        # Tipp: bei Bedarf Platz schaffen
        ax.figure.tight_layout(rect=[0,0,1,0.96])
        
    ax.set_axis_off()
    return fig, ax


# Topologie helper
def list_statecodes():
    for code, state in STATES.items():
        print(code, state)

def identify_statecode(names):
    for name in names:
        for code, state in STATES.items():
            if name in state or name in code:
                print(code, state)

# ---- PALETTEN / COLORMAPS -----------------------------------------------

DISCRETE_PRESETS = {
    # farbfehlsicht-freundlich
    "okabe_ito": [  
        "#000000",
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7"
    ],
    # deine Dürre/Feuchte-Skala (7 Klassen)
    "dmi_7": [
        "#b30000", "#e34a33", "#fdae6b", "#fee08b", "#a6d96a", "#1a9850", "#006837"
    ],
    # Beispiele
    "blues_5":  ["#eff3ff","#bdd7e7","#6baed6","#3182bd","#08519c"],
    "greens_5": ["#edf8e9","#bae4b3","#74c476","#31a354","#006d2c"],
    "oranges_5":["#fff5eb","#fed9a6","#fcae6b","#f16913","#7f2704"],
    "rdylgn_7": ["#a50026","#d73027","#f46d43","#fdae61","#a6d96a","#1a9850","#006837"],
    "rdbu_7":   ["#b2182b","#ef8a62","#fddbc7","#f7f7f7","#d1e5f0","#67a9cf","#2166ac"],
    "burd_7":   ["#2166ac","#67a9cf","#d1e5f0","#f7f7f7","#fddbc7","#ef8a62","#b2182b"],
}

def colors_from_mpl(name: str, n: int, *, reverse: bool=False, clip: float=0.05):
    cmap = plt.get_cmap(name)
    xs = np.linspace(clip, 1.0 - clip, n)
    cols = [mcolors.to_hex(cmap(x)) for x in xs]
    return list(reversed(cols)) if reverse else cols

def get_listed_cmap(preset_or_mpl: str, n: int|None=None, *, reverse: bool=False) -> ListedColormap:
    if preset_or_mpl in DISCRETE_PRESETS:
        cols = DISCRETE_PRESETS[preset_or_mpl]
        if n is not None:
            cols = (cols[:n]) if n <= len(cols) else cols + colors_from_mpl("viridis", n-len(cols))
        return ListedColormap(cols[::-1] if reverse else cols)
    if n is None:
        raise ValueError("Für MPL-Maps bitte n angeben.")
    return ListedColormap(colors_from_mpl(preset_or_mpl, n, reverse=reverse))

# --- z-score and rescaling ----------------------------------------------------------------------
# robuste z-transform
# robust wegen MAD statt med
def robust_z(s: pd.Series, clip=3.0) -> pd.Series:
    s = s.astype(float)
    med = np.nanmedian(s)
    mad = np.nanmedian(np.abs(s - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale == 0:  # Fallbacks
        q75, q25 = np.nanpercentile(s, [75, 25])
        iqr = q75 - q25
        scale = iqr / 1.349 if iqr > 0 else np.nanstd(s)
    if not np.isfinite(scale) or scale == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)  # alles gleich
    z = (s - med) / scale
    return z.clip(-clip, clip)

# index Skalierung
def scale_0_100(x: pd.Series, p_low=1, p_high=99) -> pd.Series:
    pl, ph = np.nanpercentile(x, [p_low, p_high])
    denom = (ph - pl) if ph > pl else (np.nanmax(x)-np.nanmin(x) or 1.0)
    return ((x - pl) / denom * 100).clip(0, 100)



# ----- Annotate points in a plot -------------------------------
def plot_pk(ax, gdf, name_col="GEN", dx=6, dy=6, color = 'black'):
    for _, row in gdf.iterrows():
        x, y = row.geometry.x, row.geometry.y
        label = str(row[name_col])

        gdf.plot(
            ax=ax,
            marker="o",
            color=color,
            markersize=20,
            zorder=10
        )
        
        ax.annotate(
            label, xy=(x, y),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=9, ha="left", va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="none",
                alpha=0.4),
            zorder=12,
        )