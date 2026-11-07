import argparse
import numpy as np
import matplotlib.pyplot as plt
import dunestyle.matplotlib as dunestyle
from _bootstrap import ensure_src_path
ensure_src_path()
from lib.plot_style import new_figure, apply_common_margins, get_output_path, suptitle, LEGEND_FONTSIZE

# Create -d flag to run this script with terminal output
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-r', '--run', type=int, required=False, help='Run number', default=1)
parser.add_argument('-c', '--channel', type=int, required=False, help='Channel number', default=0)
parser.add_argument('-e', '--event', type=int, required=False, help='Event number', default=1)
parser.add_argument('-p', '--plot', action='store_true', help='Show plot')
parser.add_argument('-sf', '--subfolder', type=str, default=None, help='Save the plot to a subfolder within the export path')
parser.add_argument('-o', '--output-dir', type=str, default='images', help='Base directory to save the plot into (default: images)')
parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
args = parser.parse_args()

new_figure()

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

suptitle('XA Waveform Example', fontsize='xx-large')
plt.xlabel('Time (us)')
plt.xlim(0, 12)
plt.ylabel('Amplitude (ADC)')
plt.legend(fontsize=LEGEND_FONTSIZE, frameon=False)

# Step 5: Save the plot
apply_common_margins()
out_path = get_output_path(f'WVF_RUN{args.run}_CH{args.channel}_EVENT{args.event}.png', args.subfolder, args.output_dir)
plt.savefig(out_path, dpi=300)
# Step 6: Show the plot
if args.debug:
    print(f"Plot saved as '{out_path}'")

if args.plot:
    plt.show()