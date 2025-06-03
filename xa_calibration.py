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
parser.add_argument('-n', '--name', type=str, help='XA name', default="DF-XA")
parser.add_argument('-i', '--institution', type=list, help='Institution names', default=["ciemat", "naples"])
parser.add_argument('-c', '--channel', type=list, help='Channel numbers', default=[0,1])
parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
args = parser.parse_args()

# Step 1: Load data from YAML file
for institution, channel in product(args.institution, args.channel):
    with open(f'data/xa_ch{channel}_{institution}_calibration.txt', 'r') as file:
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
    subset = data[data['Name'] == args.name]
    x = subset['OV'].values
    y = subset['Gain'].values
    x_new = np.linspace(0, 10, 300)
    y_new = interpolate.interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
    plt.plot(x_new, y_new(x_new), label=f"{institution} ch{channel}", linewidth=2, ls=':' if channel == 0 else '--', color="C0" if institution == "ciemat" else "C1")
    plt.scatter(x, y, color="C0" if institution == "ciemat" else "C1", marker='o' if channel == 0 else 'x')

plt.xlabel('Overvoltage (V)')
plt.xlim(0, 10)
plt.ylabel('Gain')
plt.ylim(0, 1e6)
plt.title('XA Calibration Data', fontsize="xx-large")
plt.legend()

# Step 5: Save the plot
if not os.path.exists('images'):
    os.makedirs('images')
plt.savefig(f'images/XA_GAIN.png', dpi=300)
# Step 6: Show the plot
if args.debug:
    print("Plot saved as 'images/XA_GAIN.png'")
plt.show()