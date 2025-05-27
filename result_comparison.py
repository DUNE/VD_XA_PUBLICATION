import os
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dunestyle.matplotlib as dunestyle

def fit_logarithmic_regression(x, y, w=None):
    # Ignore values where y is NaN
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    log_x = np.log(x)
    coeffs = np.polyfit(log_x, y, 1, w=w[mask] if w is not None else None)
    return coeffs

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
# plt.errorbar(df["OV"], df["Combined"], yerr=df["Combined_err"], fmt='o', label='Combined PDE', color='red')

# Step 3: Fit data with regression
for institute, institute_label in zip(["CIEMAT", "INFN"], ["CIEMAT", "INFN Naples"]):
    # Fit once with a polynomial line that goes through the origin
    x = df["OV"].values
    y = df[f"{institute}_PDE"].fillna(np.nan).values
    weights = 1 / df[f"{institute}_err"].fillna(np.nan).values**2
    coeffs = fit_logarithmic_regression(x, y, w=weights)

    # Generate fitted line
    x_fit = np.linspace(1e-10, 8, 100)
    y_fit = coeffs[0] * np.log(x_fit) + coeffs[1]
    if args.debug:
        print(f"\nCoefficients for {institute_label} fit", coeffs)
    plt.plot(x_fit, y_fit, label=f'Fit {institute_label}', linestyle='--')
    # Add error band to the fitted line
    y_fit_upper = y_fit + np.sqrt(df[f"{institute}_err"].mean())  # Upper bound 
    y_fit_lower = y_fit - np.sqrt(df[f"{institute}_err"].mean())  # Lower bound
    plt.fill_between(x_fit, y_fit_lower, y_fit_upper, alpha=0.2)


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
