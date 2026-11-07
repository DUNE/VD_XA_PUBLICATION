"""Shared figure sizing, margins, and font-size conventions.

Mirrors the layout used by the LOWE_RECONSTRUCTION_PUBLICATION scripts
(see src/lib/__init__.py and src/lib/plot.py there) so figures produced by
this repo's macros share the same look: fixed absolute-inch margins
(instead of tight_layout, which shifts per-plot) and the same figure size
and font-size scale.
"""
import os

import matplotlib.pyplot as plt

# LOWE_RECONSTRUCTION_PUBLICATION explicitly bumps the base font size from
# matplotlib's default (10) to 11 after loading the DUNE style (see
# src/lib/__init__.py there). All the "xx-large"/"x-large"/etc. sizes below
# are relative to this, so without it every relative size renders ~10%
# smaller than in that repo.
plt.rcParams.update({"font.size": 11})

FIGURE_BASE_WIDTH = 8
FIGURE_PANEL_WIDTH = 5
FIGURE_HEIGHT = 6

FIGURE_MARGINS_INCHES = {
    "left": 0.96,
    "right": 0.40,
    "bottom": 0.72,
    "top": 0.60,
}

TITLE_FONTSIZE = "xx-large"
SUBTITLE_FONTSIZE = "x-large"
LABEL_FONTSIZE = "xx-large"
SUBLABEL_FONTSIZE = "x-large"
LEGEND_FONTSIZE = "x-large"
LEGEND_TITLE_FONTSIZE = "xx-large"


def get_figsize(ncols=1, nrows=1):
    width = FIGURE_BASE_WIDTH + FIGURE_PANEL_WIDTH * (max(1, ncols) - 1)
    height = FIGURE_HEIGHT * max(1, nrows)
    return (float(width), float(height))


def new_figure(ncols=1, nrows=1):
    """Create (and make current) a figure with the shared figsize."""
    return plt.figure(figsize=get_figsize(ncols=ncols, nrows=nrows))


def get_output_path(filename, subfolder=None, base_dir="images"):
    """Build the export path for a plot, creating the directory if needed."""
    directory = os.path.join(base_dir, subfolder) if subfolder else base_dir
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, filename)


def apply_common_margins(fig=None, **overrides):
    """Apply the shared fixed-inch margins, converted to normalized coords.

    Pass overrides (e.g. wspace=0) for panel-specific spacing needs.
    """
    fig = fig or plt.gcf()
    figwidth = fig.get_figwidth()
    figheight = fig.get_figheight()
    margins = {
        "left": FIGURE_MARGINS_INCHES["left"] / figwidth,
        "right": 1 - FIGURE_MARGINS_INCHES["right"] / figwidth,
        "bottom": FIGURE_MARGINS_INCHES["bottom"] / figheight,
        "top": 1 - FIGURE_MARGINS_INCHES["top"] / figheight,
    }
    margins.update(overrides)
    fig.subplots_adjust(**margins)


def get_title_x(fig=None):
    """Figure-fraction x that centers a suptitle over the axes box.

    The shared left/right margins are asymmetric (left is wider, to leave
    room for y-axis tick labels and the ylabel), so the axes box itself
    isn't centered on the figure: matplotlib's default suptitle x=0.5 reads
    as off-center relative to the plot frame. This computes the actual
    horizontal center of the margined axes area instead.
    """
    fig = fig or plt.gcf()
    figwidth = fig.get_figwidth()
    left = FIGURE_MARGINS_INCHES["left"] / figwidth
    right = 1 - FIGURE_MARGINS_INCHES["right"] / figwidth
    return (left + right) / 2


def suptitle(text, fig=None, **kwargs):
    """fig.suptitle() centered on the axes box rather than the full figure."""
    fig = fig or plt.gcf()
    kwargs.setdefault("x", get_title_x(fig))
    return fig.suptitle(text, **kwargs)
