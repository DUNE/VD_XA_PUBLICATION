import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dunestyle.matplotlib as dunestyle
from _bootstrap import ensure_src_path
ensure_src_path()
from lib.plot_style import (
    get_output_path,
    new_figure,
    apply_common_margins,
    suptitle,
    LABEL_FONTSIZE,
    TITLE_FONTSIZE,
    LEGEND_FONTSIZE,
    LEGEND_TITLE_FONTSIZE,
)


# Create -d flag to run this script with terminal output
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-p', '--plot', action='store_true', help='Show plot')
parser.add_argument('-sf', '--subfolder', type=str, default=None, help='Save the plot to a subfolder within the export path')
parser.add_argument('-o', '--output-dir', type=str, default='images', help='Base directory to save the plot into (default: images)')
parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
args = parser.parse_args()

new_figure()

# Step 1: Load data from YAML file
with open(f'data/transmittance.txt', 'r') as file:
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
data = np.array([list(map(float, line.split()[1:])) for line in data_lines])

# Add name as a header for the data
data = np.column_stack((name, data))

# Convert data to df 
data = pd.DataFrame(data, columns=header.split())
# Convert all columns except 'Name' to numeric
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    data[col] = data[col].astype(float)
if args.debug:
    print("Parsed data:")
    print(data)

# Step 3: Plot each column against the first column
bin_centers = data[data.columns[0]]
bins = (bin_centers[:-1] + np.diff(bin_centers) / 2).to_numpy(dtype=float).tolist()  # Calculate bin centers for histogram
y_series = []
for col in data.columns[1:]:
    # plt.scatter(data[data.columns[0]], data[col], label=f"{col}°", alpha=0.7)
    plt.hist(data[data.columns[0]], bins=bins, weights=data[col], label=f"{col}°", histtype='step')
    y_series.append(data[col].to_numpy(dtype=float))
# Add vline at 410 nm (reference cutoff value for 45º angle)
# plt.axvline(x=400, color='r', linestyle='--', label='Cutoff', zorder=3)

# Step 4: Customize the plot
# plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
plt.xlabel('Wavelength (nm)', fontsize=LABEL_FONTSIZE)
# plt.xlim(data[data.columns[0]].min(), data[data.columns[0]].max())  # Set x-axis limits
plt.ylabel('Transmittance (%)', fontsize=LABEL_FONTSIZE)
plt.ylim(0, 115)  # Leave headroom at the top for the legend
suptitle('Transmittance Spectrum', fontsize=TITLE_FONTSIZE)
legend_loc = 'upper center'
legend = plt.legend(
    title='Incidence Angle', ncol=6, fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE,
    loc=legend_loc, frameon=False,
    handlelength=1.0, handletextpad=0.4, columnspacing=0.8, labelspacing=0.3, borderaxespad=0.3,
)

# Step 5: Save the plot
apply_common_margins()
out_path = get_output_path('TRANSMITTANCE.png', args.subfolder, args.output_dir)
plt.savefig(out_path, dpi=300)
# Step 6: Show the plot
if args.debug:
    print(f"Plot saved as '{out_path}'")

if args.plot:
    plt.show()