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
parser.add_argument('-i', '--institution', type=str, help='Institution names', default="ciemat")
parser.add_argument('-c', '--channel', type=str, nargs='+', help='Channel numbers', default=["4"])
parser.add_argument('-p', '--plot', action='store_true', help='Show plot')
parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
args = parser.parse_args()

# Step 1: Load data from YAML file
with open(f'data/sipm_ch{"-".join(args.channel)}_{args.institution}_data.txt', 'r') as file:
    data = file.read()
    # Ignore lines that start with '#'
    data = '\n'.join(line for line in data.splitlines() if not line.startswith('#'))
    if args.debug:
        print("Loaded data from text file:")
        print(data)
# Find teh file starting with 'sipm_ch' and ending with '_fit.txt'
file_list = os.listdir('data')
for file in file_list:
    if file.startswith(f'sipm_ch{"-".join(args.channel)}_{args.institution}_') and file.endswith('_fit.txt'):
        with open(f'data/{file}', 'r') as fit_file:
            fit_data = fit_file.read()
            # Ignore lines that start with '#'
            fit_data = '\n'.join(line for line in fit_data.splitlines() if not line.startswith('#'))
            if args.debug:
                print("Loaded fit data from text file:")
                print(fit_data)

        # Fit type is the remaining part of the file name after 'sipm_ch' and before '_fit.txt'
        fit_type = file[len(f'sipm_ch{"-".join(args.channel)}_{args.institution}_'):-len('_fit.txt')]
        if args.debug:
            print(f"Fit type: {fit_type}")
        break
    else:
        fit_type = 'unknown'
        if args.debug:
            print("No fit data found, using default fit type 'unknown'")    

# Step 2: Parse the data (assuming it's space-separated values with 1 header row)
header = data.split('\n')[0].strip()  # Get the header line
data_lines = data.strip().split('\n')[1:]  # Skip the first line if it's a header
data = np.array([list(map(float, line.split())) for line in data_lines])

# Convert data to df 
data = pd.DataFrame(data, columns=header.split())
# Convert all columns except 'Name' to numeric
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

if args.debug:
    print("Parsed data:")
    print(data)

# Step 3: Plot the data dividing by name
x = data['PE'].values
y = data['AMP'].values
y_fit = data['FIT'].values
dy_fit = data['DFIT'].values
x_new = np.linspace(np.min(x), np.max(x), 1000)
# plt.plot(x, y, color="C0" if institution == "ciemat" else "C1", marker='o')
plt.hist(x, weights=y, label=f"SiPM Data", color="C0" if args.institution == "ciemat" else "C1", edgecolor='black', bins=len(x), histtype='step', zorder=0)

y_new = interpolate.interp1d(x, y_fit, kind='cubic', bounds_error=False, fill_value='extrapolate')
y_new_upper = interpolate.interp1d(x, y_fit + dy_fit, kind='cubic', bounds_error=False, fill_value='extrapolate')
y_new_lower = interpolate.interp1d(x, y_fit - dy_fit, kind='cubic', bounds_error=False, fill_value='extrapolate')
plt.plot(x_new, y_new(x_new), label=f"SiPM {fit_type} Fit", linewidth=2, ls=':' if args.channel == 0 else '--', color="red", zorder=1)
# Add error band
plt.fill_between(x_new, y_new_lower(x_new), y_new_upper(x_new), color="red", alpha=0.2, edgecolor='none')

plt.xlabel('#Photo Electrons (PE)')
plt.ylabel('Norm.')
plt.title('Reference SiPM Fit', fontsize="xx-large")
plt.legend()

# Step 5: Save the plot
if not os.path.exists('images'):
    os.makedirs('images')
plt.savefig(f'images/SiPM_{fit_type.upper()}_FIT.png', dpi=300)
# Step 6: Show the plot
if args.debug:
    print(f"Plot saved as 'images/SiPM_{fit_type.upper()}_FIT.png'")

if args.plot:
    plt.show()