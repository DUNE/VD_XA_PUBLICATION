import os
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dunestyle.matplotlib as dunestyle
from cycler import cycler

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
parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
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

# Step 4: Plot the data
plt.errorbar(df["OV"], df["CIEMAT_PDE"], yerr=df["CIEMAT_err"], fmt='o', label='CIEMAT')
plt.errorbar(df["OV"], df["INFN_PDE"], yerr=df["INFN_err"], fmt='o', label='INFN Naples')

# Step 3: Fit data with regression
for idx, (institute, institute_label) in enumerate(zip(["CIEMAT", "INFN"], ["CIEMAT", "INFN Naples"])):
    # Fit once with a polynomial line that goes through the origin
    x = df["OV"].values
    y = df[f"{institute}_PDE"].fillna(np.nan).values
    weights = 1 / df[f"{institute}_err"].fillna(np.nan).values**2
    coeffs, errors = fit_logarithmic_regression_with_error(x, y, w=weights)

    # Generate fitted line
    x_fit = np.linspace(1e-10, 8, 100)
    y_fit = coeffs[0] * np.log(x_fit) + coeffs[1]
    if args.debug:
        print(f"\nCoefficients for {institute_label} fit", coeffs, "with error", errors)
    plt.plot(x_fit, y_fit, label=f'Fit {institute_label}', linestyle='--', color = f"C{idx}")
    # Add error band to the fitted line
    y_fit_upper = (coeffs[0] + errors[0]) * np.log(x_fit) + (coeffs[1] + errors[1])  # Upper bound
    y_fit_lower = (coeffs[0] - errors[0]) * np.log(x_fit) + (coeffs[1] - errors[1])  # Lower bound
    plt.fill_between(x_fit, y_fit_lower, y_fit_upper, alpha=0.2, edgecolor='none')
    # Add to data the extrapolated value at the OV where the original value is nan
    # Find where the original value is NaN
    df.loc[df[f"{institute}_PDE"].isna(), f"{institute}_err"] = np.sqrt(errors[0]**2 + errors[1]**2)  # Set error to the fit error
    df.loc[df[f"{institute}_PDE"].isna(), f"{institute}_PDE"] = coeffs[0] * np.log(df["OV"][df[f"{institute}_PDE"].isna()]) + coeffs[1]

if args.debug:
    print("\nDataFrame after fitting:")
    print(df)

# Export the data to a new YAML file
output_data = {
    "data": df.to_dict(orient='list'),
    "coefficients": {
        "CIEMAT": coeffs.tolist(),
        "INFN": coeffs.tolist()
    }
}
with open('data/PDE_fitted.yml', 'w') as file:
    yaml.dump(output_data, file, default_flow_style=False)

plt.xlabel('Overvoltage (V)')
plt.ylabel('PDE (%)')
# Set x-axis limits to match the OV values
plt.xlim(0, max(df["OV"]) + 1)
plt.ylim(0, 6)

plt.title('Comparison of PDE Values', fontsize='xx-large')
plt.legend()

dunestyle.Preliminary(x=0.02, fontsize="xx-large")
# Create images directory if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')
plt.savefig('images/RESULT_COMPARISON.png')
plt.show()
