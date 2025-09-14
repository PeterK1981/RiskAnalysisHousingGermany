"""
Workflow: Hitze-Kennzahlen je Gemeinde (ein Jahr, Tages-NetCDF)

Berechnet aus einem HYRAS-/DWD-Tagesraster Tmax (°C):
- H30: Anzahl Tage mit Tmax > 30 °C (Gemeinde-**Flächenmittel**, Mai–Sep)
- H32: Anzahl Tage mit Tmax > 32 °C (Gemeinde-**Flächenmittel**, Mai–Sep)
- H35: Anzahl Tage mit Tmax > 35 °C (Gemeinde-**Flächenmittel**, Mai–Sep)
- **H30_any / H32_any / H35_any**: Tage, an denen **mind. 1 Pixel** der Gemeinde die Schwelle überschreitet
- **H30_frac10 / H32_frac10 / H35_frac10**: Tage, an denen **≥10 % der Gemeindefläche** die Schwelle überschreitet
- I30: Graddauer über 30 °C, Summe (Mai–Sep)
- D30: Persistenz. Summe der Tage, die zu Episoden mit Länge ≥ 3 und Tmax > 30 °C gehören (Mai–Sep)

Speichert:
- cache/tmax_gem_daily_{year}.parquet   (date, ARS, tmax_gem, gt30_any, gt30_frac10, gt32_any, gt32_frac10, gt35_any, gt35_frac10)
- cache/hitze_metrics_gem_year_{year}.parquet (year, ARS, H30, H32, H35, H30_any, H32_any, H35_any, H30_frac10, H32_frac10, H35_frac10, I30, D30)

Design & Robustheit:
- VG250 wird über dein Paket geladen; GF==4-Filter dort. Layer **GEM**, Schlüssel **ARS** (1:1 durchgereicht, keine Typ-/Längenänderung).
- Keine händische Projektion: GEM wird automatisch ins Raster-CRS gebracht.
- Performance: Einmaliges **Label-Raster** (1..N) + `np.bincount` → keine Polygon-Rasterisierung pro Tag.
- Georeferenz-Fallbacks bei Mismatch: (1) Transform direkt aus Raster, (2) Reproject Summer→GEM-CRS, (3) **XY-Shift** der Transform-Matrix (Mittelpunktabgleich).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr  # noqa: F401

from rasterio.transform import array_bounds, Affine
from rasterio.enums import Resampling
import rasterio.features as rfeatures

# Importe aus deinem Paket
from choropleth_pipeline import VG250Spec, load_level, detect_id_col
from choropleth_pipeline import _ensure_crs as cp_ensure_crs

# ------------------------- Konfiguration -------------------------
MONTH_START = 5
MONTH_END = 9
THRESH_HOT = 30.0
THRESH_HOT2 = 32.0
THRESH_EXTREME = 35.0
ALL_TOUCHED = True
CACHE_DIR = Path("cache")
OUT_DIR = Path("out")

# ------------------------- Hilfsfunktionen -----------------------

def _detect_var(ds: xr.Dataset) -> str:
    """Finde die Temperaturvariable mit Zeitdimension."""
    candidates = []
    for v in ds.data_vars:
        dims = set(ds[v].dims)
        if "time" in dims and any(d in dims for d in ("x","lon","rlon")) and any(d in dims for d in ("y","lat","rlat")):
            candidates.append(v)
    if not candidates:
        raise ValueError("Keine geeignete Variable mit Dimensionen (time, y, x / lat, lon) gefunden.")
    for name in ("tasmax", "tmax", "Tmax", "tx", "air_temperature_max"):
        if name in candidates:
            return name
    return candidates[0]


def _ensure_xy(da: xr.DataArray) -> xr.DataArray:
    rename = {}
    if "x" not in da.dims:
        for cand in ("lon","rlon"):
            if cand in da.dims:
                rename[cand] = "x"; break
    if "y" not in da.dims:
        for cand in ("lat","rlat"):
            if cand in da.dims:
                rename[cand] = "y"; break
    if rename:
        da = da.rename(rename)
    if not {"x","y"}.issubset(da.dims):
        raise ValueError("Räumliche Dims nicht gefunden. Erwartet x/y oder lat/lon.")
    return da


def _to_celsius(da: xr.DataArray) -> xr.DataArray:
    units = str(da.attrs.get("units", "")).lower()
    if units.startswith("k"):
        da = da - 273.15
        da.attrs["units"] = "degC"
    return da


def _summer_slice(da: xr.DataArray, year: int) -> xr.DataArray:
    """Robustes Sommerfenster: erst nach Jahr filtern, dann Monate (Mai–Sep)."""
    if "time" not in da.dims:
        raise ValueError("Variable besitzt keine 'time'-Dimension.")
    da_year = da.sel(time=da["time"].dt.year == year)
    da_summer = da_year.sel(time=da_year["time"].dt.month.isin(range(MONTH_START, MONTH_END + 1)))
    return da_summer


# ------------------------- Kernschritte --------------------------

def daily_gem_mean_from_nc(
    nc_path: str | os.PathLike,
    year: int,
    vg: VG250Spec,
    all_touched: bool = ALL_TOUCHED,
    debug: bool = False,
    buffer_m: Optional[float] = None,
) -> pd.DataFrame:
    """Berechne Tages-Gemeindemittel (Mai–Sep) für ein Jahres-NetCDF.

    Gibt DataFrame (date, ARS, tmax_gem, gt*_any/frac10) zurück.
    Performance: Einmaliges Label-Raster + np.bincount.
    """
    ds = xr.open_dataset(nc_path, decode_cf=True)
    varname = _detect_var(ds)
    da = ds[varname]
    da = _ensure_xy(da)
    da = _to_celsius(da)

    if debug:
        tmin = pd.to_datetime(ds["time"].values.min()).strftime("%Y-%m-%d")
        tmax = pd.to_datetime(ds["time"].values.max()).strftime("%Y-%m-%d")
        print(f"Time coverage in file: {tmin} .. {tmax}; n={ds.sizes.get('time', 'NA')}")

    summer = _summer_slice(da, year)
    if debug:
        print(f"Year {year}: summer steps = {summer.sizes.get('time', 0)}")
    if summer.sizes.get("time", 0) == 0:
        raise RuntimeError(f"Kein Sommerzeitraum (Mai–Sep) im NetCDF für Jahr {year} gefunden.")

    # Gemeinden laden (GF==4 im Paket gefiltert)
    gem = load_level(vg, "GEM").copy()
    if "ARS" not in gem.columns:
        k = detect_id_col(gem, preferred="ARS")
        gem = gem.rename(columns={k: "ARS"})
    if buffer_m and buffer_m != 0:
        try:
            gem["geometry"] = gem.buffer(buffer_m)
        except Exception:
            pass
    # ARS 1:1 übernehmen
    ars_values = gem["ARS"].to_numpy(copy=True)

    # Spatial-Dims bestimmen
    x_name = next((d for d in ("x","rlon","lon") if d in summer.dims), summer.dims[-1])
    y_name = next((d for d in ("y","rlat","lat") if d in summer.dims), summer.dims[-2])

    # rioxarray über Raumachsen/CRS informieren
    try:
        summer = summer.rio.set_spatial_dims(x_dim=x_name, y_dim=y_name, inplace=False)
    except Exception:
        pass
    if summer.rio.crs is None:
        summer = cp_ensure_crs(summer)

    # GEM ins Raster-CRS
    if getattr(gem, "crs", None) is not None and gem.crs != summer.rio.crs:
        gem = gem.to_crs(summer.rio.crs)

    # Geometrien validieren
    try:
        from shapely.validation import make_valid
        gem["geometry"] = gem["geometry"].apply(make_valid)
    except Exception:
        try:
            gem["geometry"] = gem.buffer(0)
        except Exception:
            pass

    # Transform aus Raster
    transform = summer.isel(time=0).rio.transform(recalc=True)

    # Debug: Bounds
    if debug:
        h, w = int(summer.sizes[y_name]), int(summer.sizes[x_name])
        left, bottom, right, top = array_bounds(h, w, transform)
        print(f"Raster bounds: L={left:.1f} B={bottom:.1f} R={right:.1f} T={top:.1f} CRS={summer.rio.crs}")
        print(f"GEM bounds:    L={gem.total_bounds[0]:.1f} B={gem.total_bounds[1]:.1f} R={gem.total_bounds[2]:.1f} T={gem.total_bounds[3]:.1f} CRS={gem.crs}")

    # Labelraster 1. Versuch
    labels = rfeatures.rasterize(
        ((geom, i+1) for i, geom in enumerate(gem.geometry)),
        out_shape=(int(summer.sizes[y_name]), int(summer.sizes[x_name])),
        transform=transform,
        all_touched=all_touched,
        fill=0,
        dtype="int32",
    )

    N = len(gem)
    n_labels = N
    if debug:
        _present = np.unique(labels)
        covered = (_present[_present>0].size / N) if N>0 else 0
        print(f"Label coverage: {_present[_present>0].size}/{N} ({covered:.1%})")

    # Fallback 1: reproject Summer → GEM-CRS
    if covered == 0.0:
        if debug:
            print("Coverage 0% → reproject summer to GEM CRS and rebuild labels …")
        summer = summer.rio.reproject(gem.crs, resampling=Resampling.nearest)
        x_name, y_name = "x", "y"
        transform = summer.isel(time=0).rio.transform(recalc=True)
        labels = rfeatures.rasterize(
            ((geom, i+1) for i, geom in enumerate(gem.geometry)),
            out_shape=(int(summer.sizes[y_name]), int(summer.sizes[x_name])),
            transform=transform,
            all_touched=all_touched,
            fill=0,
            dtype="int32",
        )
        if debug:
            _present = np.unique(labels)
            covered = (_present[_present>0].size / N) if N>0 else 0
            print(f"Label coverage (after reproject): {_present[_present>0].size}/{N} ({covered:.1%})")

    # Fallback 2: XY-Shift (Mittelpunktabgleich)
    if covered == 0.0:
        if debug:
            print("Coverage still 0% → try constant XY-offset alignment …")
        h, w = int(summer.sizes[y_name]), int(summer.sizes[x_name])
        l, b, r, t = array_bounds(h, w, transform)
        rast_mid_x = 0.5 * (l + r)
        rast_mid_y = 0.5 * (t + b)
        gem_mid_x  = 0.5 * (gem.total_bounds[0] + gem.total_bounds[2])
        gem_mid_y  = 0.5 * (gem.total_bounds[1] + gem.total_bounds[3])
        dx_shift   = gem_mid_x - rast_mid_x
        dy_shift   = gem_mid_y - rast_mid_y
        transform = Affine(transform.a, transform.b, transform.c + dx_shift,
                           transform.d, transform.e, transform.f + dy_shift)
        labels = rfeatures.rasterize(
            ((geom, i+1) for i, geom in enumerate(gem.geometry)),
            out_shape=(int(summer.sizes[y_name]), int(summer.sizes[x_name])),
            transform=transform,
            all_touched=all_touched,
            fill=0,
            dtype="int32",
        )
        if debug:
            _present = np.unique(labels)
            covered = (_present[_present>0].size / N) if N>0 else 0
            print(f"Label coverage (after XY-shift): {_present[_present>0].size}/{N} ({covered:.1%}) • dx={dx_shift:.1f}, dy={dy_shift:.1f}")

    dates = pd.to_datetime(summer.time.values)
    rows = []

    # Tages-Loop: Mittel/Anteile per bincount
    for t, dt in enumerate(dates):
        day = summer.isel(time=t).squeeze()
        try:
            day = day.rio.set_spatial_dims(x_dim=x_name, y_dim=y_name, inplace=False)
        except Exception:
            pass
        if day.rio.crs is None:
            day = cp_ensure_crs(day)

        arr = day.values
        mask = ~np.isnan(arr)
        lab = labels[mask].ravel()
        vals = arr[mask].ravel()

        # Mittelwerte pro Label
        num = np.bincount(lab, weights=vals, minlength=n_labels+1)
        den = np.bincount(lab, minlength=n_labels+1)
        mean_per_label = np.divide(num, den, out=np.full(n_labels+1, np.nan), where=den>0)
        tmax_today = mean_per_label[1:n_labels+1]

        # any / frac10 pro Label (für 30/32/35)
        def _any_frac10(thr: float):
            c = np.bincount(lab, weights=(vals > thr).astype(np.int32), minlength=n_labels+1)
            any_lab = (c > 0)[1:n_labels+1].astype(np.uint8)
            frac_lab = np.divide(c, den, out=np.zeros(n_labels+1, dtype=float), where=den>0)[1:n_labels+1]
            frac10_lab = (frac_lab >= 0.10).astype(np.uint8)
            return any_lab, frac10_lab
        any30, frac10_30 = _any_frac10(THRESH_HOT)
        any32, frac10_32 = _any_frac10(THRESH_HOT2)
        any35, frac10_35 = _any_frac10(THRESH_EXTREME)

        rows.append(pd.DataFrame({
            "date": dt.date(),
            "ARS": gem["ARS"].to_numpy(copy=False),
            "tmax_gem": tmax_today,
            "gt30_any": any30,
            "gt30_frac10": frac10_30,
            "gt32_any": any32,
            "gt32_frac10": frac10_32,
            "gt35_any": any35,
            "gt35_frac10": frac10_35,
        }))

    out = pd.concat(rows, ignore_index=True)
    return out


def yearly_metrics_from_daily(df_daily: pd.DataFrame, year: int) -> pd.DataFrame:
    """Jahreskennzahlen je Gemeinde:
    - Mittel-basiert: H30/H32/H35, I30, D30
    - Flächenanteilsbasiert: H30_any/H32_any/H35_any und H30_frac10/H32_frac10/H35_frac10
    """
    df = df_daily.copy()

    # Mittel-basierte Schwellen (Gemeinde-Flächenmittel)
    df["H30"] = (df["tmax_gem"] > THRESH_HOT).astype(int)
    df["H32"] = (df["tmax_gem"] > THRESH_HOT2).astype(int)
    df["H35"] = (df["tmax_gem"] > THRESH_EXTREME).astype(int)
    df["I30"] = (df["tmax_gem"] - THRESH_HOT).clip(lower=0)

    # Tages-Indikatoren sicherstellen (falls alte Caches geladen werden)
    for c in ["gt30_any","gt32_any","gt35_any","gt30_frac10","gt32_frac10","gt35_frac10"]:
        if c not in df.columns:
            df[c] = 0

    # Jahres-Summen
    agg = df.groupby(["ARS"], as_index=False).agg(
        H30=("H30", "sum"),
        H32=("H32", "sum"),
        H35=("H35", "sum"),
        I30=("I30", "sum"),
        H30_any=("gt30_any", "sum"),
        H32_any=("gt32_any", "sum"),
        H35_any=("gt35_any", "sum"),
        H30_frac10=("gt30_frac10", "sum"),
        H32_frac10=("gt32_frac10", "sum"),
        H35_frac10=("gt35_frac10", "sum"),
    )

    # D30: Summe der Tage in Episoden ≥3 Tage >30°C (basierend auf Gemeinde-Mittel)
    def _d30_from_boolean(series: pd.Series, min_len: int = 3) -> int:
        v = series.to_numpy(dtype=np.int8)
        edges = np.diff(np.r_[0, v, 0])
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        lengths = ends - starts
        return int(lengths[lengths >= min_len].sum())

    d30_list = []
    for ars, sub in df.sort_values(["ARS", "date"]).groupby(["ARS"]):
        d30 = _d30_from_boolean(sub["tmax_gem"] > THRESH_HOT, min_len=3)
        d30_list.append((ars, d30))
    d30_df = pd.DataFrame(d30_list, columns=["ARS", "D30"])

    out = agg.merge(d30_df, on=["ARS"], how="left")
    out.insert(0, "year", int(year))
    return out

# ------------------------- Hauptfunktion -------------------------

def process_year(
    nc_path: str | os.PathLike,
    vg250_gpkg: str,
    year: int,
    cache_dir: Path = CACHE_DIR,
    out_dir: Path = OUT_DIR,
    save_daily: bool = True,
    debug: bool = False,
    buffer_m: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Verarbeite ein NetCDF-Jahresfile zu Tagesreihen & Jahreskennzahlen.

    Returns
    -------
    df_daily : DataFrame [date, ARS, tmax_gem, gt30_any, gt30_frac10, gt32_any, gt32_frac10, gt35_any, gt35_frac10]
    df_year  : DataFrame [year, ARS, H30, H32, H35, H30_any, H32_any, H35_any, H30_frac10, H32_frac10, H35_frac10, I30, D30]
    """
    cache_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)

    vg = VG250Spec(gpkg_path=vg250_gpkg, id_col="ARS")

    df_daily = daily_gem_mean_from_nc(
        nc_path, year, vg, all_touched=ALL_TOUCHED, debug=debug, buffer_m=buffer_m
    )

    if save_daily:
        p_daily = cache_dir / f"tmax_gem_daily_{year}.parquet"
        df_daily.to_parquet(p_daily, index=False)

    df_year = yearly_metrics_from_daily(df_daily, year)
    p_year = cache_dir / f"hitze_metrics_gem_year_{year}.parquet"
    df_year.to_parquet(p_year, index=False)

    return df_daily, df_year

# ------------------------- CLI -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hitze-Workflow je Jahr (Gemeindeebene)")
    parser.add_argument("nc", help="Pfad zum NetCDF mit Tages-Tmax (ein Jahr)")
    parser.add_argument("vg250_gpkg", help="Pfad zur VG250 Geopackage-Datei")
    parser.add_argument("year", type=int, help="Jahr der Verarbeitung")
    parser.add_argument("--save-daily", action="store_true", default=False, help="Tägliche Werte cachen")
    parser.add_argument("--all-touched", action="store_true", default=False, help="Alle berührten Zellen berücksichtigen")
    parser.add_argument("--debug", action="store_true", default=False, help="Zeit/CRS/Label-Debug-Ausgabe")
    parser.add_argument("--buffer-m", type=float, default=None, help="Optionale Geometrie-Pufferung (Meter) nur fürs Labelraster")
    args = parser.parse_args()

    if args.all_touched:
        ALL_TOUCHED = True

    daily, yearly = process_year(
        args.nc, args.vg250_gpkg, args.year,
        save_daily=args.save_daily,
        debug=args.debug,
        buffer_m=args.buffer_m,
    )
    print("Fertig.")
    print(yearly.head())
