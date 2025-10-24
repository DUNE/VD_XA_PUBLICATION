import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import dunestyle.matplotlib as dunestyle

# Create -d flag to run this script with terminal output
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-r', '--run', type=int, required=False, help='Run number', default=1)
parser.add_argument('-c', '--channel', type=int, required=False, help='Channel number', default=0)
parser.add_argument('-e', '--event', type=int, required=False, help='Event number', default=1)
parser.add_argument('-p', '--plot', action='store_true', help='Show plot')
parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
args = parser.parse_args()

# Step 1: Load data from YAML file
with open(f'data/run{args.run}_ch{args.channel}_event{args.event}.txt', 'r') as file:
    data = file.read()
    if args.debug:
        print("Loaded data from text file:")
        print(data)

# Step 2: Parse the data (assuming it's space-separated values with 1 header row)
data_lines = data.strip().split('\n')[1:]  # Skip header
data = np.array([list(map(float, line.split())) for line in data_lines])
# Step 3: Extract columns
x = data[:, 0]  # Assuming first column is x values
y = data[:, 1]  # Assuming second column is y values

# Step 4: Process the data
peak_risetime = 50  # Example risetime value in ns
max_y = np.max(y)
max_x = x[np.argmax(y)]
petrigger_lim = x[np.argmax(y)-peak_risetime]  # Assuming pretrigger limit is 20 tick before the peak
pretrigger_STD = np.std(y[:np.argmax(y)-peak_risetime])  # Assuming pretrigger goes up to the first 10 points before the peak
pretrigger_mean = np.mean(y[:np.argmax(y)-peak_risetime])  # Mean of pretrigger region
y = y - pretrigger_mean  # Subtract mean of pretrigger region

# Step 5: Plot the data
plt.hist(1e-3*x, bins=len(x), weights=y, label='Data', histtype='step', zorder=0)
plt.axvline(x=1e-3*petrigger_lim, color=f"C1", label='Integration Limits', zorder=3)
plt.axvline(x=2 + 1e-3*petrigger_lim, linestyle='--', color=f"C1", zorder=3)
plt.axhline(y=0, color='C2', label=r'Baseline $\pm$ STD', zorder=1)
plt.axhline(y=pretrigger_STD, linestyle='--', color='C2', zorder=1)
plt.axhline(y=-pretrigger_STD, linestyle='--', color='C2', zorder=1)
# plt.fill_between(x, -pretrigger_STD, pretrigger_STD, color='C2', alpha=0.5, label='Baseline STD', zorder=2, edgecolor='none')  # Draw std as shaded area with transparent external lines
# Draw std as shaded area

plt.title('XA Waveform Example', fontsize='xx-large')
plt.xlabel('Time (us)')
plt.xlim(0, 12)
plt.ylabel('Amplitude (ADC)')
plt.legend()

# Step 5: Save the plot
if not os.path.exists('images'):
    os.makedirs('images')
plt.savefig(f'images/WVF_RUN{args.run}_CH{args.channel}_EVENT{args.event}.png', dpi=300)
# Step 6: Show the plot
if args.debug:
    print("Plot saved as 'images/WVF_RUN{args.run}_CH{args.channel}_EVENT{args.event}.png'")

if args.plot:
    plt.show()