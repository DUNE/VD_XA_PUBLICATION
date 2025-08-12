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
parser.add_argument('-n', '--name', type=str, help='XA name', default="ALL")
parser.add_argument('-o', '--OV', type=float, help='Channel numbers', default=[3.5,4.5,7.0], nargs='+')
parser.add_argument('-p', '--plot', action='store_true', help='Show plot')
parser.add_argument('-s', '--shift', action='store_true', help='Enable marker shift')
args = parser.parse_args()

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

        marker = "v" if ov < 4.5 else "^" if ov > 4.5 else "o"
        if args.shift or len(args.OV) > 1:
            if ov < 4.5:
                x = x - 0.1  # Adjust x for OV < 4.5
                plt.scatter(x, y, marker=marker, color=f"C{jdx}", zorder=len(selection['Name'].unique()) - jdx)
            elif ov > 4.5:
                x = x + 0.1
                plt.scatter(x, y, marker=marker, color=f"C{jdx}", zorder=len(selection['Name'].unique()) - jdx)
            else:
                x = x
                plt.scatter(x, y, marker=marker, color=f"C{jdx}", label=f"{name}", zorder=len(selection['Name'].unique()) - jdx)
        
        else:
            plt.scatter(x, y, marker=marker, color=f"C{jdx}", label=f"{name}", zorder=len(selection['Name'].unique()) - jdx)

    for jdx, key in enumerate(variation_dict):
        y = variation_dict[key]
        plt.axhline(y=np.mean(y, where=~np.isnan(y)), color=f"C{jdx}", ls='--', zorder=len(selection['Name'].unique()) - jdx)

dunestyle.Preliminary(x=0.02, fontsize="xx-large")
plt.xlabel('SET')
# Set x axis to integer
plt.xlim(0.5, 3.5)
plt.xticks([1, 2, 3])

plt.ylabel(r'PDE (SET$_1$ - SET$_\text{i}$) / SET$_1$')

# Set y axis to percentage
plt.ylim(-0.12, 0.12)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
# Use less y ticks
plt.yticks(np.arange(-0.05, 0.16, 0.05), [f'{int(100*x)}%' for x in np.arange(-0.05, 0.16, 0.05)])

plt.title('XA PDE Divergence', fontsize="xx-large")
plt.legend(ncol=2, loc="lower center")


# Add annotation for OV markers
if len(args.OV) > 1:
    # Add a legend for the OV markers
    ov_markers = {'3.5': '▼', '4.5': '●', '7.0': '▲'}
    for idx, (ov, marker) in enumerate(ov_markers.items()):
        plt.annotate(f'{marker}: {ov} (V)', xy=(2.5, 0.05), xytext=(1.025 + 0.75*idx, 0.06), fontsize='x-large')


title = "XA_PDE"
if args.exclusive:
    title += "_EXCLUSIVE"   
if len(args.OV) == 1:
    title += f"_{args.OV[0]}V"
    plt.title(f'XA BOX PDE Variation ({args.OV[0]} V)', fontsize="xx-large")

# Step 5: Save the plot
if not os.path.exists('images'):
    os.makedirs('images')
plt.savefig(f'images/{title}.png', dpi=300)
# Step 6: Show the plot
if args.debug:
    print(f"Plot saved as 'images/{title}.png'")

if args.plot:
    plt.show()