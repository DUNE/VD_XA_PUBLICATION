import os
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dunestyle.matplotlib as dunestyle

# Get colors from dunestyle
# colors = ['#000000', '#D55E00', '#56B4E9', '#E69F00', '#009E73', '#CC79A7', '#0072B2', '#F0E442']

def weighted_average(row):
    values, weights = [], []
    for val, err in [(row.CIEMAT_PDE, row.CIEMAT_err), (row.INFN_PDE, row.INFN_err)]:
        if val is not None and err is not None:
            values.append(val)
            weights.append(1 / err**2)
    if values:
        avg = np.average(values, weights=weights)
        err = np.sqrt(1 / sum(weights))
        return avg, err
    else:
        return None, None

def fit_logarithmic_regression(x, y, w=None):
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
parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
args = parser.parse_args()

# Step 1: Load data from YAML file
with open('data/PDE_fitted.yml', 'r') as file:
    data = yaml.safe_load(file)
    if args.debug:
        print("Loaded data from YAML file:")
        print(data)

df = pd.DataFrame(data["data"])
if args.debug:
    print("Initial DataFrame:")
    print(df)

# Step 2: Compute combined values with inverse variance weighting
df[["Combined", "Combined_err"]] = df.apply(lambda row: pd.Series(weighted_average(row)), axis=1)

# Step 3: Where there is no data, set the combined values to the individual values
df["Combined"] = df["Combined"].fillna(df["CIEMAT_PDE"]).fillna(df["INFN_PDE"])
df["Combined_err"] = df["Combined_err"].fillna(df["CIEMAT_err"]).fillna(df["INFN_err"])

# Step 4: Plot the data
plt.errorbar(df["OV"], df["CIEMAT_PDE"], yerr=df["CIEMAT_err"], fmt='o', label='CIEMAT')
plt.errorbar(df["OV"], df["INFN_PDE"], yerr=df["INFN_err"], fmt='o', label='INFN Naples')
plt.errorbar(df["OV"], df["Combined"], yerr=df["Combined_err"], fmt='o', label='Combined')

# Fit once with a polynomial line that goes through the origin
x = df["OV"].values
y = df["Combined"].fillna(np.nan).values
weights = 1 / df["Combined_err"].fillna(np.nan).values**2
coeffs, errors = fit_logarithmic_regression(x, y, w=weights)

# Generate fitted line
x_fit = np.linspace(1e-10, 8, 100)
y_fit = coeffs[0] * np.log(x_fit) + coeffs[1]
if args.debug:
    print("Coefficients for fit", coeffs)
plt.plot(x_fit, y_fit, label=f'Fit Combined', linestyle='--', color=f"C2")
# Add error band to the fitted line
y_fit_upper = (coeffs[0] + errors[0]) * np.log(x_fit) + (coeffs[1] + errors[1])  # Upper bound
y_fit_lower = (coeffs[0] - errors[0]) * np.log(x_fit) + (coeffs[1] - errors[1])  # Lower bound
plt.fill_between(x_fit, y_fit_lower, y_fit_upper, alpha=0.2, color=f"C2", edgecolor='none')

# # Force the fit to go through the origin by adding a point at (0, 0) with zero error
# x = np.concatenate(([1e-1], x))
# y = np.concatenate(([1e-1], y))
# weights = np.concatenate(([1e1], weights))  # Add weight for the origin point
# coeffs, errors = fit_logarithmic_regression(x, y, w=weights)
# # Generate fitted line again
# y_fit = coeffs[0] * np.log(x_fit) + coeffs[1]
# plt.plot(x_fit, y_fit, label=f'Combined Origin Fit', linestyle=':', color=f"C2")
# # Add error band to the fitted line through origin
# y_fit_upper = (coeffs[0] + errors[0]) * np.log(x_fit) + (coeffs[1] + errors[1])  # Upper bound
# y_fit_lower = (coeffs[0] - errors[0]) * np.log(x_fit) + (coeffs[1] - errors[1])  # Lower bound
# plt.fill_between(x_fit, y_fit_lower, y_fit_upper, alpha=0.2, color = f"C2", edgecolor='none')

# Export the data to a new YAML file
output_data = {
    "data": df.to_dict(orient='list')
}
with open('data/PDE_combined.yml', 'w') as file:
    yaml.dump(output_data, file, default_flow_style=False)

plt.xlabel('Overvoltage (V)')
plt.ylabel('PDE (%)')
# Set x-axis limits to match the OV values
plt.xlim(0, max(df["OV"]) + 1)
plt.ylim(0, 6)

plt.title('Combination of PDE Values', fontsize='xx-large')
plt.legend()

# dunestyle.Preliminary(x=0.02, fontsize="xx-large")
# Create images directory if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')
plt.tight_layout()
plt.savefig('images/RESULT_COMBINED.png')

if args.plot:
    plt.show()