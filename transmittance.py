import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dunestyle.matplotlib as dunestyle

# Create -d flag to run this script with terminal output
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-p', '--plot', action='store_true', help='Show plot')
parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
args = parser.parse_args()

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
bins = bin_centers[:-1] + np.diff(bin_centers) / 2  # Calculate bin centers for histogram
for col in data.columns[1:]:
    # plt.scatter(data[data.columns[0]], data[col], label=f"{col}°", alpha=0.7)
    plt.hist(data[data.columns[0]], bins=bins, weights=data[col], label=f"{col}°", histtype='step')
# Add vline at 410 nm (reference cutoff value for 45º angle)
# plt.axvline(x=400, color='r', linestyle='--', label='Cutoff', zorder=3)

# Step 4: Customize the plot
# plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
plt.xlabel('Wavelength (nm)', fontsize='xx-large')
# plt.xlim(data[data.columns[0]].min(), data[data.columns[0]].max())  # Set x-axis limits
plt.ylabel('Transmittance (%)', fontsize='xx-large')
# plt.ylim(0, 120)  # Set y-axis limits to 0-100%
plt.title('Transmittance Spectrum', fontsize='xx-large')
plt.legend(title='Incidence Angle', ncol=1, fontsize='large', title_fontsize='large', loc='lower left')

# Step 5: Save the plot
if not os.path.exists('images'):
    os.makedirs('images')
plt.savefig(f'images/TRANSMITTANCE.png', dpi=300)
# Step 6: Show the plot
if args.debug:
    print("Plot saved as 'images/TRANSMITTANCE.png'")

if args.plot:
    plt.show()