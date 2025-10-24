import os
import argparse
import numpy as np
import pandas as pd
from itertools import product
from scipy import interpolate
import matplotlib.pyplot as plt
import dunestyle.matplotlib as dunestyle

# Create -d flag to run this script with terminal output
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
parser.add_argument('-e', '--exclusive', action='store_true', help='Include or exclude name in the plot')
parser.add_argument('-i', '--institution', type=str, help='Institution names', default=["Ciemat"], nargs='+')
parser.add_argument('-n', '--name', type=str, help='XA name', default="DF-XA")
parser.add_argument('-o', '--OV', type=float, help='Channel numbers', default=[3.5,4.5,7.0], nargs='+')
parser.add_argument('-p', '--plot', action='store_true', help='Show plot')
parser.add_argument('-s', '--shift', action='store_true', help='Enable marker shift')
args = parser.parse_args()

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
        if col == 'TOTAL':
            data[col] = 100*data[col]
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
    
    ymin = np.inf
    ymax = -np.inf
    variation_dict = {}
    for idx, (name, ov) in enumerate(product(selection['Name'].unique(), args.OV)):
        if name not in variation_dict:
            variation_dict[name] = np.array([])
        # find jdx where name is in selection['Name']
        jdx = selection['Name'].unique().tolist().index(name)
        subset = selection[(selection['Name'] == name) & (selection['OV'] == ov)]
        x = subset['BOX'].values
        y = subset['TOTAL'].values
        if np.min(y) < ymin:
            ymin = np.min(y)
        if np.max(y) > ymax:
            ymax = np.max(y)
        
        # Add values to variation_dict that are not NaN
        variation_dict[name] = np.concatenate((variation_dict[name], y))

        marker = "v" if ov < 4.5 else "^" if ov > 4.5 else "o"
        error = 0.09 * y  # Calculate 9% error
        if args.shift:
            if ov < 4.5:
                x = x - 0.1  # Adjust x for OV < 4.5
                plt.errorbar(x, y, yerr=error, marker=marker, color=f"C{jdx}", zorder=len(selection['Name'].unique()) - jdx, fmt='o')
            elif ov > 4.5:
                x = x + 0.1
                plt.errorbar(x, y, yerr=error, marker=marker, color=f"C{jdx}", zorder=len(selection['Name'].unique()) - jdx, fmt='o')
            else:
                x = x
                plt.errorbar(x, y, yerr=error, marker=marker, color=f"C{jdx}", zorder=len(selection['Name'].unique()) - jdx, label=f"{name}", fmt='o')

        else:
            plt.errorbar(x, y, yerr=error, marker=None, color=f"C{idx}", label=f"OV: {ov}" if ~args.exclusive else f"{name}" if idx == 0 else None, zorder=len(selection['Name'].unique()) - jdx, fmt='o')

    for jdx, key in enumerate(variation_dict):
        y = variation_dict[key]
        y = np.asarray(y).reshape(-1, 3)
        y[:,1] = 2*y[:,1] # Double the values for the second box for proper weighting

        # Plot in sets of 3
        color = f"C{jdx}"
        if args.exclusive == False:
            color = "red"

        for idx, i in enumerate(y):
            if idx == 0:
                plt.axhline(np.sum(i)/4, color=color, ls='--', zorder=len(selection['Name'].unique()) - jdx, label=f"{key} Average" if args.exclusive else "Average")
            else:
                plt.axhline(np.sum(i)/4, color=color, ls='--', zorder=len(selection['Name'].unique()) - jdx)
            plt.fill_between([0.5,3.5], np.sum(i)/4*0.91, np.sum(i)/4*1.09, color=color, alpha=0.2, edgecolor='none')

# dunestyle.Preliminary(x=0.02, fontsize="xx-large")
plt.xlabel('BOX Position')
# Set x axis to integer
plt.xlim(0.75, 3.25)
plt.xticks([1, 2, 3])

plt.ylabel(f'PDE (%)')

# Set y axis to percentage
plt.ylim(ymin - 1, ymax + 1.5 if args.exclusive else ymax + 1)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
# Use less y ticks
# plt.yticks(np.arange(ymin - 0.01, ymax + 0.015, 0.01), [f'{int(100*x)}%' for x in np.arange(ymin - 0.01, ymax + 0.015, 0.01)])

plt.legend(ncol=2, loc="lower center")

# Add annotation for OV markers
if len(args.OV) > 1 and args.exclusive:
    # Add a legend for the OV markers
    ov_markers = {'3.5': '▼', '4.5': '●', '7.0': '▲'}
    for idx, (ov, marker) in enumerate(ov_markers.items()):
        plt.annotate(f'{marker}: {ov} (V)', xy=(2.5, ymax+1), xytext=(1.025 + 0.75*idx, ymax+0.5), fontsize='x-large')

file_title = "XA_BOX_PDE"
plot_title = "XA PDE BOX Variation"
if args.exclusive:
    file_title += "_EXCLUSIVE"   

if len(args.OV) == 1:
    file_title += f"_{args.OV[0]}V"
    plot_title += f" ({args.OV[0]} V)"

plt.title(plot_title, fontsize="xx-large")

# Step 5: Save the plot
if not os.path.exists('images'):
    os.makedirs('images')
plt.tight_layout()
plt.savefig(f'images/{file_title}.png', dpi=300)
# Step 6: Show the plot
if args.debug:
    print(f"Plot saved as 'images/{file_title}.png'")

if args.plot:
    plt.show()