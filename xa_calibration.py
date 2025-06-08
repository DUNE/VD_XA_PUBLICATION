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
parser.add_argument('-c', '--channel', type=list, help='Channel numbers', default=[0,1])
parser.add_argument('-i', '--institution', type=list, help='Institution names', default=["Ciemat", "INFN Naples"])
parser.add_argument('-n', '--name', type=str, help='XA name', default="ALL")
parser.add_argument('-e', '--exclusive', action='store_true', help='Include or exclude name in the plot')
parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
args = parser.parse_args()

# Step 1: Load data from YAML file
for institution, channel in product(args.institution, args.channel):
    with open(f'data/xa_ch{channel}_{institution.split(" ")[-1].lower()}_calibration.txt', 'r') as file:
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
    
    label=f"{institution} CH {channel}"
    for idx, name in enumerate(selection['Name'].unique()):
        if len(selection['Name'].unique()) > 1:
            label = f"{institution} {name}"
        
        subset = selection[selection['Name'] == name]
        x = subset['OV'].values
        dx = 0.02 * subset['OV'].values
        y = subset['Gain'].values
        dy = subset['DGain'].values
        plt.errorbar(x, y, xerr=dx, yerr=dy, ls='none', mfc='w' if channel == 0 else f'C{idx}' if institution == "Ciemat" else f"C{idx+1}", color=f"C{idx}" if institution == "Ciemat" else f"C{idx+1}", marker='o', label=label)

        x_new = np.linspace(0, 10, 100)
        y_new = interpolate.interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
        plt.plot(x_new, y_new(x_new), ls=':' if channel == 0 else '--', color=f'C{idx}' if institution == "Ciemat" else f"C{idx+1}")

dunestyle.Preliminary(x=0.02, fontsize="xx-large")
plt.xlabel('Overvoltage (V)')
plt.xlim(0, 10)
# plt.xscale('log')
plt.ylabel('Gain')
plt.ylim(0, 1e6)
# plt.yscale('linear')
plt.title('XA Calibration Data', fontsize="xx-large")
plt.legend()

title = "XA_GAIN"
if args.channel != [0, 1]:
    title += f"_CH{'-'.join(args.channel)}"
if args.exclusive:
    title += "_EXCLUSIVE"   

# Step 5: Save the plot
if not os.path.exists('images'):
    os.makedirs('images')
plt.savefig(f'images/{title}.png', dpi=300)
# Step 6: Show the plot
if args.debug:
    print(f"Plot saved as 'images/{title}.png'")
plt.show()