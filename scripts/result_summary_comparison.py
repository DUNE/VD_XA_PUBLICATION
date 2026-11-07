import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import dunestyle.matplotlib as dunestyle
from _bootstrap import ensure_src_path
ensure_src_path()
from lib.plot_style import (
    get_output_path,
    get_figsize,
    apply_common_margins,
    suptitle,
    TITLE_FONTSIZE,
    SUBTITLE_FONTSIZE,
    LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
)


def weighted_mean_and_error(y, yerr):
    """Inverse-variance weighted mean and its propagated standard error."""
    weights = 1 / np.square(yerr)
    mean = np.average(y, weights=weights)
    err = np.sqrt(1 / np.sum(weights))
    return mean, err


# Create -d flag to run this script with terminal output
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', '--institution', type=str, help='Institution name', default="Ciemat")
parser.add_argument('-m', '--mean-only', action='store_true', help='Show only the weighted mean of each series instead of every OV data point')
parser.add_argument('-p', '--plot', action='store_true', help='Show plot')
parser.add_argument('-sf', '--subfolder', type=str, default=None, help='Save the plot to a subfolder within the export path')
parser.add_argument('-o', '--output-dir', type=str, default='images', help='Base directory to save the plot into (default: images)')
parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
args = parser.parse_args()

# Step 1: Load data from text file
with open(f'data/xa_{args.institution.split(" ")[-1].lower()}_pde_key_comparisons.txt', 'r') as file:
    data = file.read()
    # Ignore lines that start with '#'
    data = '\n'.join(line for line in data.splitlines() if not line.startswith('#'))
    if args.debug:
        print("Loaded data from text file:")
        print(data)

# Step 2: Parse the data (assuming it's space-separated values with 1 header row)
header = data.split('\n')[0].strip()  # Get the header line
data_lines = data.strip().split('\n')[1:]  # Skip the first line if it's a header
name = np.array([list(map(str, line.split()[:1])) for line in data_lines])
values = np.array([list(map(float, line.split()[1:])) for line in data_lines])

# Add name as a header for the data
data = np.column_stack((name, values))

# Convert data to df
data = pd.DataFrame(data, columns=header.split())
for col in data.columns[1:]:
    data[col] = pd.to_numeric(data[col], errors='coerce')
if args.debug:
    print("Parsed data:")
    print(data)

# Step 3: Define the three panels of interest and, within each, the series to overlay
panels = [
    {
        "title": "",
        "series": {"DFvsnoDF_SS": "SS", "DFvsnoDF_DS": "DS"},
        "formula": "(noDF-DF)/DF",
    },
    {
        "title": "",
        "series": {"SSvsDS_DF": "DF", "SSvsDS_noDF": "noDF"},
        "formula": "(DS-SS)/SS",
    },
    {
        "title": "",
        "series": {"ConcvsStd_noDF": "SS, noDF"},
        "formula": "(24mg-80mg)/80mg",
    },
]
legend_locs = ["lower center", "lower center", "lower center"]

fig, axs = plt.subplots(1, 3, figsize=get_figsize(), sharey=True, gridspec_kw={'wspace': 0})

for panel_idx, (ax, panel, legend_loc) in enumerate(zip(axs, panels, legend_locs)):
    legend_handles = []
    n_series = len(panel["series"])
    # Small symmetric offsets so, with no OV axis to spread them across, the
    # per-series mean points don't sit exactly on top of one another.
    mean_offsets = np.linspace(-0.15, 0.15, n_series) if n_series > 1 else np.array([0.0])
    for idx, (name, label) in enumerate(panel["series"].items()):
        subset = data[data["Name"] == name]
        if subset.empty:
            if args.debug:
                print(f"No data found for {name}. Skipping...")
            continue
        y = subset["VALUE"].values
        yerr = subset["ERROR"].values
        color = f"C{idx}"

        if args.mean_only:
            mean_y, mean_err = weighted_mean_and_error(y, yerr)
            ax.errorbar([mean_offsets[idx]], [mean_y], yerr=[mean_err], fmt='o', color=color, capsize=3)
        else:
            x = subset["OV"].values
            # No `label=` on the errorbar itself: matplotlib's default legend
            # handle for errorbar() includes the error-bar caps, which -- once
            # isolated in empty axes space -- reads as a spurious floating data
            # point. Use a plain marker-only proxy artist for the legend instead.
            ax.errorbar(x, y, yerr=yerr, fmt='o', color=color, capsize=3)
            mean_y, _ = weighted_mean_and_error(y, yerr)
            ax.axhline(mean_y, color=color, ls='--', linewidth=1.5)

        if label is not None:
            legend_handles.append(Line2D([], [], marker='o', linestyle='none', color=color, label=label))

    ax.axhline(0, color='grey', ls=':', linewidth=1, zorder=0)
    if args.mean_only:
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_xlabel(panel["formula"], fontsize=LABEL_FONTSIZE, linespacing=1.4, labelpad=12)
    else:
        ax.set_xlim(2.5, 8)
        ax.set_xticks([3.5, 4.5, 7.0])
    ax.set_title(panel["title"], fontsize=SUBTITLE_FONTSIZE)
    if legend_handles:
        ax.legend(handles=legend_handles, loc=legend_loc, fontsize=LEGEND_FONTSIZE, frameon=False)
    if panel_idx > 0:
        # Shared y-axis: only the leftmost section needs tick labels, so the
        # three sections read as one split plot rather than three separate ones.
        ax.tick_params(labelleft=False)

axs[0].set_ylabel('Relative PDE variation')
axs[0].set_ylim(-0.15, 0.25)
axs[0].yaxis.set_major_formatter(lambda val, pos: f"{val*100:.0f}%")
if not args.mean_only:
    fig.supxlabel('Overvoltage (V)', fontsize=LABEL_FONTSIZE)

suptitle('Key XA PDE Comparisons', fig=fig, fontsize=TITLE_FONTSIZE)

# dunestyle.Preliminary(x=0.02, fontsize="xx-large")

# Step 4: Save the plot
apply_common_margins(fig, wspace=0)
out_path = get_output_path(f"RESULT_SUMMARY_COMPARISON{'_MEAN' if args.mean_only else ''}.png", args.subfolder, args.output_dir)
plt.savefig(out_path, dpi=300)
if args.debug:
    print(f"Plot saved as '{out_path}'")

if args.plot:
    plt.show()
