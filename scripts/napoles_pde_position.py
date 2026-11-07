import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dunestyle.matplotlib as dunestyle
from _bootstrap import ensure_src_path
ensure_src_path()
from lib.plot_style import new_figure, apply_common_margins, get_output_path, suptitle, TITLE_FONTSIZE, LEGEND_FONTSIZE

parser = argparse.ArgumentParser(description='Plot the INFN Naples PDE as a function of source position and height.')
parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
parser.add_argument('-p', '--plot', action='store_true', help='Show plot')
parser.add_argument('-sf', '--subfolder', type=str, default=None, help='Save the plot to a subfolder within the export path')
parser.add_argument('-od', '--output-dir', type=str, default='images', help='Base directory to save the plot into (default: images)')
args = parser.parse_args()

new_figure()

POSITIONS = ['1', '2', 'C', '4']
COLORS = {'5cm': 'black', '10cm': '#D2691E', '15cm': '#55B9EB'}
MARKERS = {'5cm': 'o', '10cm': 'o', '15cm': 'o'}


def get_offsets(total):
    if total <= 1:
        return np.array([0.0])
    return np.linspace(-0.1, 0.1, total)


# Step 1: Load data
with open('data/xa_naples_pde_position.txt', 'r') as file:
    data = file.read()
    data = '\n'.join(line for line in data.splitlines() if not line.startswith('#'))
    if args.debug:
        print("Loaded data from text file:")
        print(data)

header = data.split('\n')[0].strip()
data_lines = data.strip().split('\n')[1:]
name = np.array([list(map(str, line.split()[:1])) for line in data_lines])
values = np.array([list(map(str, line.split()[1:])) for line in data_lines])
data = np.column_stack((name, values))
data = pd.DataFrame(data, columns=header.split())
for col in ['VALUE', 'ERROR']:
    data[col] = 100 * pd.to_numeric(data[col], errors='coerce')
if args.debug:
    print("Parsed data:")
    print(data)

# Step 2: Plot the data, one dodged series per source height
names = data['Name'].unique()
x_offsets = dict(zip(names, get_offsets(len(names))))
all_values = data['VALUE'].to_numpy()
ymin, ymax = all_values.min(), all_values.max()

for name in names:
    subset = data[data['Name'] == name]
    x = np.array([POSITIONS.index(p) for p in subset['Position']]) + x_offsets[name]
    plt.errorbar(
        x,
        subset['VALUE'],
        yerr=subset['ERROR'],
        marker=MARKERS[name],
        color=COLORS[name],
        linestyle='none',
        label=f'OV: 4.5 - Heigth: {name}',
    )

# Step 3: Overall average line (plain mean across all positions/heights)
average = all_values.mean()
plt.axhline(average, color='red', ls='--', label='OV: 4.5 - Average')

plt.xlabel('Position')
plt.xlim(-0.5, len(POSITIONS) - 0.5)
plt.xticks(range(len(POSITIONS)), POSITIONS)

plt.ylabel('PDE (%)')
plt.ylim(ymin - 0.9, ymax + 0.6)

plt.legend(loc='lower center', ncol=2, fontsize=LEGEND_FONTSIZE, frameon=False)

suptitle('XA PDE Position Variation', fontsize=TITLE_FONTSIZE)

# Step 4: Save the plot
apply_common_margins()
out_path = get_output_path('NAPLES_PDE_POSITION.png', args.subfolder, args.output_dir)
plt.savefig(out_path, dpi=300)
if args.debug:
    print(f"Plot saved as '{out_path}'")

if args.plot:
    plt.show()
