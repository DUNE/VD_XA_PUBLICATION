import os
import yaml
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dunestyle.matplotlib as dunestyle
from cycler import cycler
import copy
from _bootstrap import ensure_src_path
ensure_src_path()
from lib.plot_style import new_figure, apply_common_margins, get_output_path, suptitle, TITLE_FONTSIZE, LEGEND_FONTSIZE

# Get colors from dunestyle
# colors = ['#000000', '#D55E00', '#56B4E9', '#E69F00', '#009E73', '#CC79A7', '#0072B2', '#F0E442']

def fit_logarithmic_regression(x, y, w=None):
    # Ignore values where y is NaN
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    log_x = np.log(x)
    coeffs = np.polyfit(log_x, y, 1, w=w[mask] if w is not None else None)
    return coeffs

def fit_logarithmic_regression_with_error(x, y, w=None):
    # Ignore values where y is NaN
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    log_x = np.log(x)
    coeffs, cov = np.polyfit(log_x, y, 1, cov=True, w=w[mask] if w is not None else None)
    return coeffs, np.sqrt(np.diag(cov))

# Create -d flag to run this script with terminal output
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-p', '--plot', action='store_true', help='Show plot')
parser.add_argument('-sf', '--subfolder', type=str, default=None, help='Save the plot to a subfolder within the export path')
parser.add_argument('-o', '--output-dir', type=str, default='images', help='Base directory to save the plot into (default: images)')
parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
parser.add_argument('-i', '--institutes', type=str, default='CIEMAT,INFN',
                    help='Comma-separated institute base names to plot (e.g. CIEMAT,INFN)')
parser.add_argument('--list-institutes', action='store_true', help='List available institutes from the data and exit')
args = parser.parse_args()

# Step 1: Load data from YAML file
with open('data/PDE.yml', 'r') as file:
    data = yaml.safe_load(file)
    if args.debug:
        print("\nLoaded data from YAML file:")
        print(data)

df = pd.DataFrame(data["data"])
if args.debug:
    print("\nInitial DataFrame:")
    print(df)

# Discover available institutes (columns ending with '_PDE')
available_institutes = [col[:-4] for col in df.columns if col.endswith('_PDE')]
if args.list_institutes:
    print('Available institutes:')
    for inst in available_institutes:
        print('-', inst)
    sys.exit(0)

# Parse requested institutes
selected_institutes = [s.strip() for s in args.institutes.split(',') if s.strip()]
if len(selected_institutes) != 2:
    print('Error: please specify exactly two institutes with -i, e.g. -i CIEMAT,INFN')
    print('Use --list-institutes to see available institute names')
    sys.exit(2)

# prettier label mapping
label_map = {
    'CIEMAT': 'CIEMAT (ref)',
    'CIEMAT_SIM': 'CIEMAT (sim)',
}
# Step 4: Plot the data for the selected institutes
new_figure()
plt.errorbar(df["OV"], df[f"{selected_institutes[0]}_PDE"], yerr=df.get(f"{selected_institutes[0]}_err"), fmt='o', label=label_map.get(selected_institutes[0], selected_institutes[0]))
plt.errorbar(df["OV"], df[f"{selected_institutes[1]}_PDE"], yerr=df.get(f"{selected_institutes[1]}_err"), fmt='o', label=label_map.get(selected_institutes[1], selected_institutes[1]))

# Step 3: Fit data with regression
coefficients_dict = {}
for idx, institute in enumerate(selected_institutes):
    institute_label = institute if institute != 'INFN' else 'INFN'
    x = df["OV"].values
    y = df[f"{institute}_PDE"].fillna(np.nan).values
    weights = 1 / df[f"{institute}_err"].fillna(np.nan).values**2
    coeffs, errors = fit_logarithmic_regression_with_error(x, y, w=weights)

    # Generate fitted line
    x_fit = np.linspace(1e-10, 8, 100)
    y_fit = coeffs[0] * np.log(x_fit) + coeffs[1]
    if args.debug:
        print(f"\nCoefficients for {institute_label} fit", coeffs, "with error", errors)
    # Use prettier mapped label for fit legend
    mapped_label = label_map.get(institute, institute if institute != 'INFN' else 'INFN Naples')
    plt.plot(x_fit, y_fit, label=f'Fit {mapped_label}', linestyle='--', color = f"C{idx}")
    # Add error band to the fitted line
    y_fit_upper = (coeffs[0] + errors[0]) * np.log(x_fit) + (coeffs[1] + errors[1])  # Upper bound
    y_fit_lower = (coeffs[0] - errors[0]) * np.log(x_fit) + (coeffs[1] - errors[1])  # Lower bound
    plt.fill_between(x_fit, y_fit_lower, y_fit_upper, alpha=0.2, edgecolor='none')
    # Add to data the extrapolated value at the OV where the original value is nan
    df.loc[df[f"{institute}_PDE"].isna(), f"{institute}_err"] = np.sqrt(errors[0]**2 + errors[1]**2)  # Set error to the fit error
    df.loc[df[f"{institute}_PDE"].isna(), f"{institute}_PDE"] = coeffs[0] * np.log(df["OV"][df[f"{institute}_PDE"].isna()]) + coeffs[1]
    coefficients_dict[institute] = coeffs.tolist()

if args.debug:
    print("\nDataFrame after fitting:")
    print(df)

# Export the data to a new YAML file
pair_key = f"{selected_institutes[0]}_{selected_institutes[1]}"
new_data = df.to_dict(orient='list')
# Load existing fitted file if present and add/update only the pair entry at top-level
fitted_path = 'data/PDE_fitted.yml'
root = {}
if os.path.exists(fitted_path):
    with open(fitted_path, 'r') as f:
        old = yaml.safe_load(f) or {}
    # Convert legacy formats into new top-level per-pair entries
    # If old has a 'pairs' mapping, lift its entries
    if isinstance(old, dict):
        if 'pairs' in old and isinstance(old['pairs'], dict):
            for k, v in old['pairs'].items():
                root[k] = copy.deepcopy(v)
        # Also lift any existing top-level pair-like entries (keys that contain '_')
        for k, v in old.items():
            if k not in ('data', 'coefficients', 'pairs'):
                root[k] = copy.deepcopy(v)

# Set/replace the current pair entry
root[pair_key] = {
    'selected_institutes': selected_institutes,
    'coefficients': coefficients_dict,
    'data': new_data,
}

with open(fitted_path, 'w') as file:
    yaml.dump(root, file, default_flow_style=False)

plt.xlabel('Overvoltage (V)')
plt.ylabel('PDE (%)')
# Set x-axis limits to match the OV values
plt.xlim(0, max(df["OV"]) + 1)
plt.ylim(0, 6)

suptitle('Comparison of PDE Values', fontsize=TITLE_FONTSIZE)
plt.legend(fontsize=LEGEND_FONTSIZE, frameon=False)

# dunestyle.Preliminary(x=0.02, fontsize="xx-large")
apply_common_margins()
out_path = get_output_path(f'RESULT_COMPARISON_{pair_key}.png', args.subfolder, args.output_dir)
plt.savefig(out_path)

if args.debug:
    print(f"\nPlot saved as '{out_path}'")
# Step 6: Show the plot
if args.plot:
    plt.show()
