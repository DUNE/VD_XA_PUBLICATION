import argparse
import numpy as np
import pandas as pd
from itertools import product
from scipy import interpolate
import matplotlib.pyplot as plt
import dunestyle.matplotlib as dunestyle
from _bootstrap import ensure_src_path
ensure_src_path()
from lib.plot_style import new_figure, apply_common_margins, get_output_path, suptitle, TITLE_FONTSIZE, LEGEND_FONTSIZE

# Create -d flag to run this script with terminal output
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
parser.add_argument('-e', '--exclusive', action='store_true', help='Include or exclude name in the plot')
parser.add_argument('-i', '--institution', type=str, help='Institution names', default=["Ciemat"], nargs='+')
parser.add_argument('-n', '--name', type=str, help='XA name', default="DF-XA")
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
        return np.array([-0.1, 0.1])
    return np.linspace(-0.1, 0.1, total)

# Step 1: Load data from YAML file
for institution in args.institution:
    with open(f'data/xa_{institution.split(" ")[-1].lower()}_pde_box.txt', 'r') as file:
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
        if col == 'BOX':
            data[col] = data[col].astype(int)
    if args.debug:
        print("Parsed data:")
        print(data)

    # Step 3: Plot the data dividing by name
    if args.exclusive:
        # Remove entries with names that end with '_1' or '_2'
        selection = data[~data['Name'].str.endswith(('_1', '_2'))]
    else:
        selection = data[data['Name'] == args.name]

    if selection.empty:
        if args.debug:
            print(f"No data found for {args.name} in {institution}. Skipping...")
        continue
    
    variation_dict = {}
    ov_offsets = dict(zip(args.OV, get_offsets(len(args.OV))))
    for idx, (name, ov) in enumerate(product(selection['Name'].unique(), args.OV)):
        if name not in variation_dict:
            variation_dict[name] = np.array([])
        # find jdx where name is in selection['Name']
        jdx = selection['Name'].unique().tolist().index(name)
        subset = selection[(selection['Name'] == name) & (selection['OV'] == ov)]
        x = subset['BOX'].values
        y = 100*subset['TOTAL'].values
        x_offset = ov_offsets.get(ov, 0.0) if (len(args.OV) > 1 or args.shift) else 0.0
        x = x + x_offset
        
        # Add values to variation_dict that are not NaN
        variation_dict[name] = np.concatenate((variation_dict[name], (y - y[0]) / y[0]))

        marker = "v" if ov < 4.5 else "^" if ov > 4.5 else "o"
        if args.shift or len(args.OV) > 1:
            plt.scatter(
                x,
                (y - y[0]) / y[0],
                marker=marker,
                color=f"C{jdx}",
                zorder=len(selection['Name'].unique()) - jdx,
                label=f"{name}" if ov == 4.5 else None
            )
        else:
            plt.scatter(x, (y - y[0]) / y[0], marker=marker, color=f"C{jdx}", label=f"{name}", zorder=len(selection['Name'].unique()) - jdx)

    for jdx, key in enumerate(variation_dict):
        y = variation_dict[key]
        x = np.array([1, 2, 3] * (len(y) // 3))
        # Reshape the arrays
        x = np.asarray(x).reshape(-1, 3)
        y = np.asarray(y).reshape(-1, 3)
        # Plot in sets of 3
        mean_x = []
        mean_y = []
        for i in range(0, len(y[0])):
            if args.debug:
                print(f"STD for {key} at BOX {i+1}: {np.mean(y[:,i], where=~np.isnan(y[:,i])):.2f}%")
            # Pick up the ith values of each set of 3
            mean_x.append(np.mean(x[:, i]))
            mean_y.append(np.mean(y[:, i], where=~np.isnan(y[:, i])))
        plt.plot(mean_x, mean_y, color=f"C{jdx}", ls='--', zorder=len(selection['Name'].unique()) - jdx)

# dunestyle.Preliminary(x=0.02, fontsize="xx-large")
plt.xlabel('BOX')
# Set x axis to integer
plt.xlim(0.75, 3.25)
plt.xticks([1, 2, 3])

plt.ylabel(r'PDE (BOX$_\text{i}$ - BOX$_1$) / BOX$_1$')

# Set y axis to percentage
plt.ylim(-0.05, 0.16)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
# Use less y ticks
# plt.yticks(np.arange(-0.05, 0.16, 0.05), [f'{int(100*x)}%' for x in np.arange(-0.05, 0.16, 0.05)])

plt.legend(ncol=2, loc="lower center", fontsize=LEGEND_FONTSIZE, frameon=False)

file_title = "XA_BOX_PDE"
plot_title = "XA PDE BOX Ratio"
if args.exclusive:
    file_title += "_EXCLUSIVE"
if len(args.OV) == 1:
    file_title += f"_{args.OV[0]}V"
    plot_title += f" ({args.OV[0]} V)"

suptitle(plot_title, fontsize=TITLE_FONTSIZE)

# Step 5: Save the plot
apply_common_margins()
out_path = get_output_path(f'{file_title}_RATIO.png', args.subfolder, args.output_dir)
plt.savefig(out_path, dpi=300)
# Step 6: Show the plot
if args.debug:
    print(f"Plot saved as '{out_path}'")

if args.plot:
    plt.show()