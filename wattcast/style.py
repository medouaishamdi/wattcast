"""Shared chart colors (one palette for report figures and the dashboard)."""

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Categorical series - fixed order, never cycled
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Status colors - always shown together with a text label
RISK_COLORS = {"High": "#d03b3b", "Medium": "#ec835a", "Low": "#0ca30c"}

# Diverging: reduces risk (blue) <-> increases risk (red)
REDUCES = "#2a78d6"
INCREASES = "#e34948"


def apply_matplotlib_style():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": TEXT_SECONDARY,
            "ytick.labelcolor": TEXT_SECONDARY,
            "legend.frameon": False,
            "legend.labelcolor": TEXT_SECONDARY,
            "lines.linewidth": 2,
            "font.size": 10.5,
        }
    )
