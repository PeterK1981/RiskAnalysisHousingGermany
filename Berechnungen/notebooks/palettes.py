# ---- PALETTEN / COLORMAPS -----------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

# Katalog (Namen = Matplotlib-Colormaps)
CMAPS = {
    "sequential": [
        "viridis", "plasma", "magma", "inferno", "cividis",
        "YlGnBu", "OrRd", "Greens", "Blues", "BuPu", "PuBuGn"
    ],
    "diverging": [
        "RdYlGn", "RdBu", "BrBG", "PiYG", "PRGn", "PuOr",
        "coolwarm", "Spectral", "seismic"
    ],
    "qualitative": [
        "tab10", "tab20", "Set2", "Set3", "Paired", "Accent"
    ],
}

# Handverlesene DISKRETE Paletten (Hex), farbfehlsicht-tauglich
DISCRETE_PRESETS = {
    # Okabe–Ito (8-Colors, colorblind-friendly)
    "okabe_ito": [
        "#000000", "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
    ],
    # Deine DMI-Klassenpalette (7 Klassen; rot → grün)
    "dmi_7": [
        "#b30000", "#e34a33", "#fdae6b",
        "#fee08b", "#a6d96a", "#1a9850", "#006837"
    ],
    # Beispiele: sequentiell, 5 Klassen
    "blues_5":  ["#eff3ff", "#bdd7e7", "#6baed6", "#3182bd", "#08519c"],
    "greens_5": ["#edf8e9", "#bae4b3", "#74c476", "#31a354", "#006d2c"],
    "oranges_5":["#fff5eb", "#fed9a6", "#fcae6b", "#f16913", "#7f2704"],
    # Divergierend (neutraler Mittelbereich), 7 Klassen
    "rdylgn_7": ["#a50026", "#d73027", "#f46d43", "#fdae61", "#a6d96a", "#1a9850", "#006837"],
    "rdbu_7":   ["#b2182b", "#ef8a62", "#fddbc7", "#f7f7f7", "#d1e5f0", "#67a9cf", "#2166ac"],
}

def colors_from_mpl(name: str, n: int, *, reverse: bool=False, clip: float=0.05):
    """
    Diskrete Farben aus einer Matplotlib-Colormap ziehen.
    clip: schneidet die äußeren 5% ab, damit sehr helle/dunkle Enden vermieden werden.
    """
    cmap = plt.get_cmap(name)
    xs = np.linspace(clip, 1.0 - clip, n)
    cols = [mcolors.to_hex(cmap(x)) for x in xs]
    return list(reversed(cols)) if reverse else cols

def get_listed_cmap(preset_or_mpl: str, n: int|None=None, *, reverse: bool=False) -> ListedColormap:
    """
    Liefert eine ListedColormap:
      - wenn preset_or_mpl in DISCRETE_PRESETS: exakt diese Farben (ggf. gekürzt/verlängert)
      - sonst: sampled aus Matplotlib-CMAP (n muss dann gesetzt werden)
    """
    if preset_or_mpl in DISCRETE_PRESETS:
        cols = DISCRETE_PRESETS[preset_or_mpl]
        if n is not None:
            if n <= len(cols):
                cols = cols[:n]
            else:
                # zyklisch auffüllen (oder besser: neu samplen)
                more = colors_from_mpl("viridis", n - len(cols))
                cols = cols + more
        return ListedColormap(cols[::-1] if reverse else cols)
    # sonst: Matplotlib-Cmap diskretisieren
    if n is None:
        raise ValueError("Bitte n angeben, wenn aus Matplotlib-Cmap gesampled werden soll.")
    cols = colors_from_mpl(preset_or_mpl, n, reverse=reverse)
    return ListedColormap(cols)
