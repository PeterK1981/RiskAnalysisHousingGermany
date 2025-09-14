#!/usr/bin/env python3
"""
Erzeuge eine Excel-Datei mit Kreisnamen (GEN) als Schlüssel und Dummywerten.
- Liest VG250_KRS aus einem GeoPackage
- Bildet einen stabilen Text-Key:
    GEN_key = GEN (BEZ)  falls GEN nicht eindeutig ist,
    sonst GEN_key = GEN
- Schreibt Excel: Sheet "kreise" mit Spalten [GEN, GEN_key, <value_col>]
"""

import argparse
import os
import numpy as np
import pandas as pd
import geopandas as gpd


def make_gen_key(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Liefert einen eindeutigen Textschlüssel:
    - Primär 'GEN'
    - Falls doppelt: 'GEN (BEZ)' wenn BEZ existiert
      Fallbacks: 'GEN (NUTS)' oder 'GEN (#<laufende_nummer>)'
    """
    if "GEN" not in gdf.columns:
        raise KeyError("Feld 'GEN' fehlt im VG250-Layer.")

    gen = gdf["GEN"].astype(str)

    if not gen.duplicated().any():
        return gen  # bereits eindeutig

    # Versuche mit BEZ zu disambiguieren
    if "BEZ" in gdf.columns:
        bez = gdf["BEZ"].astype(str)
        gen_key = gen + " (" + bez + ")"
        if not gen_key.duplicated().any():
            return gen_key

    # Versuche mit NUTS
    if "NUTS" in gdf.columns:
        nuts = gdf["NUTS"].astype(str)
        gen_key = gen + " (" + nuts + ")"
        if not gen_key.duplicated().any():
            return gen_key

    # Letzter Fallback: laufende Nummer für Duplikate
    gen_key = gen.copy()
    counts = {}
    for i, val in gen_key.items():
        counts[val] = counts.get(val, 0) + 1
        if counts[val] > 1:
            gen_key.iloc[i] = f"{val} (#{counts[val]})"
    if gen_key.duplicated().any():
        raise RuntimeError("GEN konnte nicht eindeutig gemacht werden.")
    return gen_key


def main(vg_path: str,
         layer: str,
         out_xlsx: str,
         value_col: str,
         seed: int,
         mean: float,
         sd: float) -> None:

    if not os.path.exists(vg_path):
        raise FileNotFoundError(f"VG250 nicht gefunden: {vg_path}")

    gdf = gpd.read_file(vg_path, layer=layer)
    # Nur Attribute behalten (Geometry fliegt für Excel raus)
    attrs = gdf.drop(columns=[c for c in gdf.columns if c.lower() == "geometry"], errors="ignore").copy()

    # GEN_key bilden
    attrs["GEN_key"] = make_gen_key(attrs)

    # Dummywerte erzeugen
    rng = np.random.default_rng(seed)
    values = np.round(rng.normal(loc=mean, scale=sd, size=len(attrs)), 2)

    out = pd.DataFrame({
        "GEN": attrs["GEN"].astype(str),
        "GEN_key": attrs["GEN_key"].astype(str),
        value_col: values
    })

    # Excel schreiben
    out.to_excel(out_xlsx, sheet_name="kreise", index=False)
    print(f"OK: {out_xlsx} geschrieben – Zeilen: {len(out)}, Spalten: {list(out.columns)}")
    print("Hinweis: Für Joins bevorzugt 'GEN_key' verwenden (eindeutig).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Erzeuge Excel mit GEN/GEN_key und Dummywerten je Kreis aus VG250_KRS."
    )
    ap.add_argument("--vg", required=True, help="Pfad zu DE_VG250.gpkg")
    ap.add_argument("--layer", default="VG250_KRS", help="Layername (Default: VG250_KRS)")
    ap.add_argument("--out", default="kreiswerte_from_vg_GEN.xlsx", help="Output-Excel (Default: kreiswerte_from_vg_GEN.xlsx)")
    ap.add_argument("--value_col", default="my_metric", help="Spaltenname der Dummy-Messgröße (Default: my_metric)")
    ap.add_argument("--seed", type=int, default=42, help="Zufalls-Seed (Default: 42)")
    ap.add_argument("--mean", type=float, default=50.0, help="Mittelwert der Dummyverteilung (Default: 50.0)")
    ap.add_argument("--sd", type=float, default=15.0, help="Standardabweichung der Dummyverteilung (Default: 15.0)")
    args = ap.parse_args()

    main(args.vg, args.layer, args.out, args.value_col, args.seed, args.mean, args.sd)
