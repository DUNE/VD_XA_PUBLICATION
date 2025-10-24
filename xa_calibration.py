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
parser.add_argument('-c', '--channel', type=int, help='Channel numbers', default=["ALL"], nargs='+', required=False)
parser.add_argument('-e', '--exclusive', action='store_true', help='Include or exclude name in the plot')
parser.add_argument('-i', '--institution', type=str, help='Institution names', default=["Ciemat", "INFN Naples"], nargs='+')
parser.add_argument('-n', '--name', type=str, help='XA name', default="ALL")
parser.add_argument('-p', '--plot', action='store_true', help='Show plot')
parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
args = parser.parse_args()

# Step 1: Load data from YAML file
for institution, channel in product(args.institution, args.channel):
    path = f'data/xa_ch{channel}_{institution.split(" ")[-1].lower()}_calibration.txt'
    if channel not in [0, 1]:
        path = f'data/xa_{institution.split(" ")[-1].lower()}_calibration.txt'
    with open(path, 'r') as file:
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
            print(f"No data found for {args.name} in {institution} channel {channel}. Skipping...")
        continue
    
    for idx, name in enumerate(selection['Name'].unique()):
        label=f"{institution}"
        if channel in [0, 1]:
            label += f" CH {channel}"

        if len(selection['Name'].unique()) > 1:
            label += f" {name}"

        if len(selection['Name'].unique()) == 1 and len(args.institution) == 1:
            label = "Data"
        
        subset = selection[selection['Name'] == name]
        x = subset['OV'].values
        dx = 0.02 * subset['OV'].values
        y = subset['Gain'].values
        dy = subset['DGain'].values
        x_new = np.linspace(0, 10, 100)
        y_new = interpolate.interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
        
        if len(args.channel) > 1:
            plt.errorbar(x, y, xerr=dx, yerr=dy, ls='none', mfc='w' if channel == 0 else f'C{idx}' if institution == "Ciemat" else f"C{idx+1}", color=f"C{idx}" if institution == "Ciemat" else f"C{idx+1}", marker='o', label=label, zorder = len(args.name) - idx)
            plt.plot(x_new, y_new(x_new), ls=':' if channel == 0 else '--', color=f'C{idx}' if institution == "Ciemat" else f"C{idx+1}", zorder = len(args.name) - idx)
        else:
            plt.errorbar(x, y, xerr=dx, yerr=dy, ls='none', color=f"C{idx}" if institution == "Ciemat" else f"C{idx+1}", marker='o', label=label, zorder = len(args.name) - idx)
            plt.plot(x_new, y_new(x_new), ls='--', color=f'C{idx}' if institution == "Ciemat" else f"C{idx+1}", zorder = len(args.name) - idx)

# dunestyle.Preliminary(x=0.02, fontsize="xx-large")
plt.xlabel('Overvoltage (V)')
plt.xlim(0, 10)
# plt.xscale('log')
plt.ylabel('Gain')
plt.ylim(0, 1e6)
# plt.yscale('linear')
if len(args.channel) == 1 and args.channel[0] in [0, 1]:
    plt.title(f'XA Calibration (CH {args.channel[0]})', fontsize="xx-large")
else:
    plt.title('XA Calibration', fontsize="xx-large")
plt.legend()
# Place the legend in the lower right corner
plt.legend(loc='lower right')

title = "XA_GAIN"
if len(args.channel) == 1:
    title += f"_CH{args.channel[0]}"

if args.exclusive:
    title += "_EXCLUSIVE"   

# Step 5: Save the plot
if not os.path.exists('images'):
    os.makedirs('images')
plt.savefig(f'images/{title}.png', dpi=300)
# Step 6: Show the plot
if args.debug:
    print(f"Plot saved as 'images/{title}.png'")

if args.plot:
    plt.show()