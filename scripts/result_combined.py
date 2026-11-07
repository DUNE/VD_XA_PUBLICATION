import yaml
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dunestyle.matplotlib as dunestyle
from _bootstrap import ensure_src_path
ensure_src_path()
from lib.plot_style import new_figure, apply_common_margins, get_output_path, suptitle, TITLE_FONTSIZE, LEGEND_FONTSIZE

# Helper: format value ± error with sensible significant digits
def format_with_uncertainty(val, err, sig=2):
    try:
        if not np.isfinite(val) or not np.isfinite(err) or err == 0:
            return f"{val:.4f} \\pm {err:.4f}"
        # exponent of error
        exp = int(np.floor(np.log10(abs(err))))
        # decimals to show so that error has `sig` significant digits
        decimals = max(0, -exp + (sig - 1))
        val_r = round(val, decimals)
        err_r = round(err, decimals)
        # build format string with appropriate decimals and format the rounded values
        return ("{0:.%df} \\pm {1:.%df}" % (decimals, decimals)).format(val_r, err_r)
    except Exception:
        return f"{val:.4f} \\pm {err:.4f}"

# Get colors from dunestyle
# colors = ['#000000', '#D55E00', '#56B4E9', '#E69F00', '#009E73', '#CC79A7', '#0072B2', '#F0E442']

def weighted_average(row, institutes):
    values, weights = [], []
    for inst in institutes:
        val = getattr(row, f"{inst}_PDE", None)
        err = getattr(row, f"{inst}_err", None)
        if val is not None and err is not None and not pd.isna(val) and not pd.isna(err):
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
    return coeffs, cov

# Create -d flag to run this script with terminal output
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-p', '--plot', action='store_true', help='Show plot')
parser.add_argument('-sf', '--subfolder', type=str, default=None, help='Save the plot to a subfolder within the export path')
parser.add_argument('-o', '--output-dir', type=str, default='images', help='Base directory to save the plot into (default: images)')
parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
parser.add_argument('-i', '--institutes', type=str, default='CIEMAT,INFN',
                    help='Comma-separated institute base names to combine (e.g. CIEMAT,INFN)')
parser.add_argument('--list-institutes', action='store_true', help='List available institutes from the data and exit')
args = parser.parse_args()

# Parse requested institutes
selected_institutes = [s.strip() for s in args.institutes.split(',') if s.strip()]
if len(selected_institutes) < 1:
    print('Error: please specify at least one institute with -i, e.g. -i CIEMAT,INFN')
    print('Use --list-institutes to see available institute names')
    sys.exit(2)

pair_key = "_".join(selected_institutes)

# Step 1: Load data from YAML file
with open('data/PDE_fitted.yml', 'r') as file:
    data_all = yaml.safe_load(file) or {}
    if args.debug:
        print("Loaded data from YAML file:")
        print(data_all)

# Prefer per-pair top-level entries. If present, use that; else fall back to legacy 'data' key.
if pair_key in data_all:
    data = data_all[pair_key].get('data', {})
else:
    # legacy support
    data = data_all.get('data', data_all)

df = pd.DataFrame(data)
if args.debug:
    print("Initial DataFrame:")
    print(df)

# Discover available institutes (columns ending with '_PDE')
available_institutes = [col[:-4] for col in df.columns if col.endswith('_PDE')]
if args.list_institutes:
    print('Available institutes:')
    for inst in available_institutes:
        print('-', inst)
    sys.exit(0)

# Step 2: Compute combined values with inverse variance weighting
df[["Combined", "Combined_err"]] = df.apply(lambda row: pd.Series(weighted_average(row, selected_institutes)), axis=1)

# Step 3: Where there is no data, set the combined values to the individual values
for inst in selected_institutes:
    df["Combined"] = df["Combined"].fillna(df.get(f"{inst}_PDE"))
    df["Combined_err"] = df["Combined_err"].fillna(df.get(f"{inst}_err"))

# Step 3b: Drop combined points at OV values where none of the selected
# institutes has real (non-extrapolated) data. PDE_fitted.yml already has
# such gaps filled in with per-institute logarithmic-fit extrapolations, so
# without this check the combined point would be built purely out of fit
# values with no underlying measurement.
with open('data/PDE.yml', 'r') as file:
    raw_data = yaml.safe_load(file)
raw_df = pd.DataFrame(raw_data["data"])
raw_indexed = raw_df.set_index("OV")
has_real_data = raw_indexed[[f"{inst}_PDE" for inst in selected_institutes if f"{inst}_PDE" in raw_df.columns]].notna().any(axis=1)
has_real_data = df["OV"].map(has_real_data).fillna(False)
df.loc[~has_real_data, "Combined"] = np.nan
df.loc[~has_real_data, "Combined_err"] = np.nan

# Step 4: Plot the data for the selected institutes
new_figure()
for idx, inst in enumerate(selected_institutes):
    pde_col = f"{inst}_PDE"
    err_col = f"{inst}_err"
    if pde_col not in df.columns:
        if args.debug:
            print(f"Warning: {pde_col} not in data, skipping plot for {inst}")
        continue
    # prettier label mapping
    label_map = {
        'CIEMAT': 'CIEMAT (ref)',
        'CIEMAT_SIM': 'CIEMAT (sim)',
    }
    label = label_map.get(inst, inst if inst != 'INFN' else 'INFN Naples')
    # Only plot points backed by a real (non-extrapolated) measurement for
    # this institute; PDE_fitted.yml fills gaps with fit extrapolations,
    # which shouldn't be drawn as if they were data.
    if pde_col in raw_df.columns:
        inst_has_real_data = df["OV"].map(raw_indexed[pde_col].notna()).fillna(False)
    else:
        inst_has_real_data = pd.Series(False, index=df.index)
    plot_y = df[pde_col].where(inst_has_real_data)
    plot_err = df[err_col].where(inst_has_real_data)
    plt.errorbar(df["OV"], plot_y, yerr=plot_err, fmt='o', label=label, color=f"C{idx}")
# Plot combined
plt.errorbar(df["OV"], df["Combined"], yerr=df["Combined_err"], fmt='o', label='Combined', color=f"C{len(selected_institutes)}")

# Fit once with a polynomial line that goes through the origin
x = df["OV"].values
y = df["Combined"].fillna(np.nan).values
weights = 1 / np.square(df["Combined_err"].fillna(np.nan).values)

coeffs, cov = fit_logarithmic_regression(x, y, w=weights)

# Generate fitted line
x_fit = np.linspace(1e-10, 8, 100)
y_fit = coeffs[0] * np.log(x_fit) + coeffs[1]
if args.debug:
    print("Coefficients for fit", coeffs)
# Compute chi2 for the fit using points with finite values
mask = (~np.isnan(y)) & (np.isfinite(weights)) & (weights > 0)
if mask.sum() > 0:
    x_mask = x[mask]
    y_mask = y[mask]
    w_mask = weights[mask]
    sigma = 1.0 / np.sqrt(w_mask)
    y_model = coeffs[0] * np.log(x_mask) + coeffs[1]
    chi2 = np.sum(((y_mask - y_model) / sigma) ** 2)
    ndf = max(mask.sum() - len(coeffs), 1)
    red_chi2 = chi2 / ndf
else:
    chi2 = float('nan')
    ndf = 0
    red_chi2 = float('nan')
# If reduced chi2 > 1, scale covariance so parameter errors reflect observed scatter
cov_used = cov
if ndf > 0 and np.isfinite(red_chi2) and red_chi2 > 1.0:
    cov_used = cov * red_chi2

# Parameter errors from (possibly scaled) covariance
errors = np.sqrt(np.diag(cov_used))

# Build legend label with math notation, line splits and parameter values with errors and chi2
a_str = format_with_uncertainty(coeffs[0], errors[0], sig=2)
b_str = format_with_uncertainty(coeffs[1], errors[1], sig=2)

label_fit = (
    f'Fit: $y = a\\ln(x) + b$'
    f'\n$a = {a_str},\\; b = {b_str}$'
    f'\n$\\chi^2/ndf = {chi2:.2f}/{ndf} = {red_chi2:.2f}$'
)
plt.plot(x_fit, y_fit, label=label_fit, linestyle='--', color=f"C{len(selected_institutes)}")

# Add error band to the fitted line using full (possibly scaled) covariance (prediction variance)
log_x_fit = np.log(x_fit)
J = np.vstack([log_x_fit, np.ones_like(log_x_fit)])  # shape (2, N)
# prediction variance: var = J.T @ cov_used @ J  (vectorized)
var_pred = np.sum((cov_used @ J) * J, axis=0)
var_pred = np.where(var_pred > 0, var_pred, 0.0)
sigma_pred = np.sqrt(var_pred)
y_fit_upper = y_fit + sigma_pred
y_fit_lower = y_fit - sigma_pred
plt.fill_between(x_fit, y_fit_lower, y_fit_upper, alpha=0.2, color=f"C{len(selected_institutes)}", edgecolor='none')

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

# Export the data to a new YAML file as a single top-level pair entry
pair_output = {
    "selected_institutes": selected_institutes,
    "coefficients": {"Combined": coeffs.tolist()},
    "data": df.to_dict(orient='list'),
}
with open('data/PDE_combined.yml', 'w') as file:
    yaml.dump({pair_key: pair_output}, file, default_flow_style=False)

plt.xlabel('Overvoltage (V)')
plt.ylabel('PDE (%)')
# Set x-axis limits to match the OV values
plt.xlim(0, max(df["OV"]) + 1)
plt.ylim(0, 6)

suptitle('Combination of PDE Values', fontsize=TITLE_FONTSIZE)
plt.legend(fontsize=LEGEND_FONTSIZE, frameon=False)

# dunestyle.Preliminary(x=0.02, fontsize="xx-large")
apply_common_margins()
out_path = get_output_path(f'RESULT_COMBINED_{pair_key}.png', args.subfolder, args.output_dir)
plt.savefig(out_path)

if args.plot:
    plt.show()