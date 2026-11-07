import argparse
import numpy as np
import pandas as pd
from itertools import product
from scipy import interpolate
import matplotlib.pyplot as plt
import dunestyle.matplotlib as dunestyle
from typing import Any
from _bootstrap import ensure_src_path
ensure_src_path()
from lib.plot_style import new_figure, apply_common_margins, get_output_path, suptitle, TITLE_FONTSIZE, LEGEND_FONTSIZE

# Create -d flag to run this script with terminal output
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
parser.add_argument('-e', '--exclusive', action='store_true', help='Include or exclude name in the plot')
parser.add_argument('-i', '--institution', type=str, help='Institution names', default=["Ciemat"], nargs='+')
parser.add_argument('-n', '--name', type=str, help='XA name', default="ALL")
parser.add_argument('-o', '--OV', type=float, help='Channel numbers', default=[3.5,4.5,7.0], nargs='+')
parser.add_argument('-p', '--plot', action='store_true', help='Show plot')
parser.add_argument('-sf', '--subfolder', type=str, default=None, help='Save the plot to a subfolder within the export path')
parser.add_argument('-od', '--output-dir', type=str, default='images', help='Base directory to save the plot into (default: images)')
parser.add_argument('-s', '--shift', action='store_true', help='Enable marker shift')
args = parser.parse_args()

new_figure()


def get_offsets(total):
    if total <= 1:
        return np.array([0.0])
    if total == 2:
        return np.array([-0.2, 0.2])
    return np.linspace(-0.2, 0.2, total)


# Step 1: Load data from YAML file
for institution in args.institution:
    with open(f'data/xa_{institution.split(" ")[-1].lower()}_pde.txt', 'r') as file:
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
    for col in data.columns[1:]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        if col == 'SET':
            data[col] = data[col].astype(int)
    if args.debug:
        print("Parsed data:")
        print(data)

    # Step 3: Plot the data dividing by name
    if args.exclusive:
        selection = data[data['Name'] != args.name]
    else:
        selection = data[data['Name'] == args.name]
    
    if selection.empty:
        if args.debug:
            print(f"No data found for {args.name} in {institution}. Skipping...")
        continue
    
    variation_dict = {}
    ov_colors = {ov: f"C{idx}" for idx, ov in enumerate(args.OV)}
    legend_added = set()
    name_list = selection['Name'].unique()
    # Slightly offset each arapuca configuration along the x-axis so
    # overlapping points at the same SET are easier to tell apart.
    name_offsets = dict(zip(name_list, get_offsets(len(name_list)))) if args.exclusive else {}
    for idx, (name, ov) in enumerate(product(selection['Name'].unique(), args.OV)):
        if name not in variation_dict:
            variation_dict[name] = np.array([])
        # find jdx where name is in selection['Name']
        jdx = selection['Name'].unique().tolist().index(name)
        subset = selection[(selection['Name'] == name) & (selection['OV'] == ov)]
        x = subset['SET'].values
        y = subset['VALUE'].values

        # Add values to variation_dict that are not NaN
        variation_dict[name] = np.concatenate((variation_dict[name], y))

        marker: Any = ("v" if ov < 4.5 else "^" if ov > 4.5 else "o") if len(args.OV) > 1 else "o"
        if args.exclusive:
            x = x + name_offsets.get(name, 0.0)
        elif ov < 4.5 and args.shift:
            x = x - 0.1  # Adjust x for OV < 4.5
        elif ov > 4.5 and args.shift:
            x = x + 0.1

        # When split by configuration, the marker shape (annotated separately)
        # already conveys the OV, so color/legend are keyed by name instead.
        color = f"C{jdx}" if args.exclusive else ov_colors.get(ov, f"C{jdx}")
        label = None if args.exclusive else (f"{ov} V" if ov not in legend_added else None)
        plt.scatter(x, y, marker=marker, color=color, label=label, zorder=len(selection['Name'].unique()) - jdx)
        if label is not None:
            legend_added.add(ov)

    if args.exclusive:
        for jdx, name in enumerate(variation_dict):
            mean_value = np.nanmean(variation_dict[name])
            plt.axhline(y=mean_value, color=f"C{jdx}", ls='--', zorder=0, label=f"{name} Average")
    else:
        for ov in args.OV:
            subset = selection[selection['OV'] == ov]
            if subset.empty:
                continue
            mean_value = np.nanmean(subset['VALUE'].values)
            plt.axhline(y=mean_value, color=ov_colors.get(ov, 'red'), ls='--', zorder=0)

# dunestyle.Preliminary(x=0.02, fontsize="xx-large")
plt.xlabel('SET')
# Set x axis to integer
plt.xlim(0.5, 3.5)
plt.xticks([1, 2, 3])

plt.ylabel('PDE (SET1 - SETi) / SET1')

# Set y axis to percentage
plt.ylim(-0.12, 0.12)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
# Use less y ticks
# plt.yticks(np.arange(-0.05, 0.16, 0.05), [f'{int(100*x)}%' for x in np.arange(-0.05, 0.16, 0.05)])

suptitle('XA PDE Divergence', fontsize=TITLE_FONTSIZE)
handles, labels = plt.gca().get_legend_handles_labels()
if handles:
    plt.legend(ncol=2, loc="lower center", fontsize=LEGEND_FONTSIZE, frameon=False)


# Add annotation for OV markers
if len(args.OV) > 1 and len(args.name) > 1:
    # Add a legend for the OV markers
    ov_markers = {'3.5': '▼', '4.5': '●', '7.0': '▲'}
    for idx, (ov, marker) in enumerate(ov_markers.items()):
        plt.annotate(f'{marker}: {ov} (V)', xy=(2.5, 0.05), xytext=(1.025 + 0.75*idx, 0.06), fontsize='x-large')


title = "XA_PDE"
if args.exclusive:
    title += "_EXCLUSIVE"   
if len(args.OV) == 1:
    title += f"_{args.OV[0]}V"
    suptitle(f'XA PDE Divergence ({args.OV[0]} V)', fontsize=TITLE_FONTSIZE)

# Step 5: Save the plot
apply_common_margins()
out_path = get_output_path(f'{title}.png', args.subfolder, args.output_dir)
plt.savefig(out_path, dpi=300)
# Step 6: Show the plot
if args.debug:
    print(f"Plot saved as '{out_path}'")

if args.plot:
    plt.show()