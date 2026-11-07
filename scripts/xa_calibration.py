import os
import argparse
import glob
import numpy as np
import pandas as pd
from itertools import product
from scipy import interpolate
import matplotlib.pyplot as plt
import dunestyle.matplotlib as dunestyle
from _bootstrap import ensure_src_path
ensure_src_path()
from lib.plot_style import new_figure, apply_common_margins, get_output_path, suptitle, TITLE_FONTSIZE, LEGEND_FONTSIZE

# Create -d flag to run this script with terminal output
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-c', '--channel', type=int, help='Channel numbers', default=["ALL"], nargs='+', required=False)
parser.add_argument('-e', '--exclusive', action='store_true', help='Include or exclude name in the plot')
parser.add_argument('-i', '--institution', type=str, help='Institution names', default=["Ciemat", "INFN Naples"], nargs='+')
parser.add_argument('-n', '--name', type=str, help="XA name (omit to plot every name found in the data; 'ALL' is also a literal Name value meaning box positions combined)", default=None)
parser.add_argument('-ls', '--linestyle', type=str, default=None, help="Line style of every curve (default: ':' for CH 0 and '--' for CH 1 with two channels, '--' otherwise)")
parser.add_argument('-f', '--fit', action='store_true', help='Draw a weighted linear fit of gain vs OV instead of the point-to-point interpolation')
parser.add_argument('-p', '--plot', action='store_true', help='Show plot')
parser.add_argument('-sf', '--subfolder', type=str, default=None, help='Save the plot to a subfolder within the export path')
parser.add_argument('-o', '--output-dir', type=str, default='images', help='Base directory to save the plot into (default: images)')
parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
args = parser.parse_args()

new_figure()


def institution_key(institution_name):
    return institution_name.split(" ")[-1].lower()


def load_calibration_file(path):
    base = os.path.basename(path)
    if base.startswith("xa_ch"):
        parts = base.split("_")
        channel = int(parts[1][2:])
        institution = parts[2]
    else:
        channel = "ALL"
        institution = base.split("_")[1]

    with open(path, 'r') as file:
        raw = file.read()

    raw = '\n'.join(line for line in raw.splitlines() if not line.startswith('#') and line.strip() != '')
    if raw.strip() == '':
        return None

    header = raw.split('\n')[0].strip()
    data_lines = raw.strip().split('\n')[1:]
    names = [line.split()[0] for line in data_lines]
    values = [list(map(float, line.split()[1:])) for line in data_lines]
    arr = np.column_stack((names, values))
    df = pd.DataFrame(arr, columns=header.split())
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return (institution, channel, df)

# Step 1: Load all calibration files into a store so interpolation can use data
# from other institutes and channels when a target dataset is missing points.
data_store = {}
for path in sorted(glob.glob('data/*_calibration.txt')):
    loaded = load_calibration_file(path)
    if loaded is None:
        if args.debug:
            print(f"No data in {path}")
        continue

    institution, channel, df = loaded
    data_store[(institution, channel)] = df
    if args.debug:
        print(f"Loaded {len(df)} rows from {path}")

# Determine which names to plot
if args.name is None:
    all_names = set()
    for df in data_store.values():
        all_names.update(df['Name'].unique())
    names_to_plot = sorted(all_names)
else:
    names_to_plot = [args.name]

# Step 2: For each channel and name, ensure each institute has data at the union of OV points by interpolation
for channel in args.channel:
    for name in names_to_plot:
        # gather all OV from available institutes for this channel/name
        union_x = np.array([])
        selected_keys = {institution_key(inst) for inst in args.institution}
        for (inst, ch), df in data_store.items():
            if channel in [0, 1] and ch != channel:
                continue
            # Only institutes being plotted define the OV grid, so a subset
            # (e.g. -i Ciemat) is not interpolated at another institute's points.
            if inst not in selected_keys:
                continue
            subset = df[df['Name'] == name]
            if subset.empty:
                continue
            union_x = np.unique(np.concatenate((union_x, subset['OV'].values.astype(float))))
        if union_x.size == 0:
            if args.debug:
                print(f"No OV points found for name '{name}' channel {channel} across institutes")
            continue
        union_x = np.sort(union_x)

        # now for each institute, add missing points
        for inst in args.institution:
            key = (institution_key(inst), channel)
            if key not in data_store:
                continue
            df = data_store[key]
            subset = df[df['Name'] == name].copy()
            if subset.empty:
                # no data for this name at this institute/channel -> skip interpolation
                continue
            existing_x = subset['OV'].values.astype(float)
            missing_x = [x for x in union_x if not np.any(np.isclose(existing_x, x, atol=1e-8))]
            if len(missing_x) == 0:
                continue
            if len(existing_x) < 2:
                if args.debug:
                    print(f"Not enough points to interpolate for {inst} CH {channel} name {name}")
                continue
            # create interpolation functions
            y_interp = interpolate.interp1d(existing_x, subset['Gain'].values.astype(float), kind='linear', bounds_error=False, fill_value='extrapolate')
            try:
                dy_interp = interpolate.interp1d(existing_x, subset['DGain'].values.astype(float), kind='linear', bounds_error=False, fill_value='extrapolate')
            except Exception:
                dy_interp = None

            # build new rows
            new_rows = []
            for mx in missing_x:
                new_gain = float(y_interp(mx))
                new_dgain = float(dy_interp(mx)) if dy_interp is not None else 0.0
                new_rows.append({
                    'Name': name,
                    'OV': mx,
                    'Gain': new_gain,
                    'DGain': new_dgain
                })
            if new_rows:
                newdf = pd.DataFrame(new_rows)
                # append and sort by OV
                df = pd.concat([df, newdf], ignore_index=True, sort=False)
                df['OV'] = pd.to_numeric(df['OV'], errors='coerce')
                df = df.sort_values(by='OV').reset_index(drop=True)
                # ensure numeric columns
                for col in ['Gain', 'DGain']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                data_store[key] = df
                if args.debug:
                    print(f"Added {len(new_rows)} interpolated points for {inst} CH {channel} name {name}")

# Step 3: Plot the data dividing by name
for channel in args.channel:
    for inst_idx, institution in enumerate(args.institution):
        key = (institution_key(institution), channel)
        if key not in data_store:
            if args.debug:
                print(f"No data loaded for {institution} CH {channel}")
            continue
        df = data_store[key]
        if args.exclusive:
            selection = df[df['Name'] != args.name]
        else:
            if args.name is None:
                selection = df.copy()
            else:
                selection = df[df['Name'] == args.name]
        if selection.empty:
            if args.debug:
                print(f"No data found for plotting for {institution} CH {channel}")
            continue
        for idx, nm in enumerate(selection['Name'].unique()):
            label = f"{institution}"
            if channel in [0, 1]:
                label += f" CH {channel}"
            if len(selection['Name'].unique()) > 1:
                label += f" {nm}"
            if len(selection['Name'].unique()) == 1 and len(args.institution) == 1:
                label = f"CH {channel}" if len(args.channel) > 1 else "Data"

            subset = selection[selection['Name'] == nm]
            x = subset['OV'].values.astype(float)
            dx = 0.02 * x
            y = subset['Gain'].values.astype(float)
            dy = subset['DGain'].values.astype(float) if 'DGain' in subset.columns else np.zeros_like(y)
            x_new = np.linspace(0, 10, 100)
            # interpolation for plotting smooth curve
            if args.fit and len(x) >= 2:
                weights = 1 / dy if np.all(dy > 0) else None
                slope, intercept = np.polyfit(x, y, 1, w=weights)
                y_new = slope * x_new + intercept
            elif len(x) >= 2:
                y_new_func = interpolate.interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
                y_new = y_new_func(x_new)
            else:
                y_new = np.full_like(x_new, np.nan)

            color_idx = idx if institution == "Ciemat" else idx + 1
            if len(selection['Name'].unique()) == 1 and len(args.institution) == 1 and len(args.channel) > 1:
                # One curve per channel: distinguish the channels by colour.
                color_idx = args.channel.index(channel)
            if len(args.channel) > 1:
                plt.errorbar(x, y, xerr=dx, yerr=dy, ls='none', marker='o', mfc='w' if channel == 0 else None, color=f"C{color_idx}", label=label, zorder=10-idx)
                plt.plot(x_new, y_new, ls=args.linestyle or (':' if channel == 0 else '--'), color=f'C{color_idx}', zorder=9-idx)
            else:
                plt.errorbar(x, y, xerr=dx, yerr=dy, ls='none', color=f"C{color_idx}", marker='o', label=label, zorder=10-idx)
                plt.plot(x_new, y_new, ls=args.linestyle or '--', color=f'C{color_idx}', zorder=9-idx)

# dunestyle.Preliminary(x=0.02, fontsize="xx-large")
plt.xlabel('Overvoltage (V)')
plt.xlim(0, 10)
# plt.xscale('log')
plt.ylabel('Gain')
plt.ylim(0, 1e6)
# plt.yscale('linear')
if len(args.channel) == 1 and args.channel[0] in [0, 1]:
    suptitle(f'XA Calibration (CH {args.channel[0]})', fontsize=TITLE_FONTSIZE)
else:
    suptitle('XA Calibration', fontsize=TITLE_FONTSIZE)
# Place the legend in the lower right corner
plt.legend(loc='lower right', fontsize=LEGEND_FONTSIZE, frameon=False)

title = "XA_GAIN"
# If both channels are requested, match historical combined filename XA_GAIN_CHALL.png
if set(args.channel) == {0, 1}:
    title += "_CHALL"
elif len(args.channel) == 1:
    title += f"_CH{args.channel[0]}"

if args.exclusive:
    title += "_EXCLUSIVE"
elif args.name is not None and "ALL" not in args.channel:
    title += f"_{args.name}"

# Step 5: Save the plot
apply_common_margins()
out_path = get_output_path(f'{title}.png', args.subfolder, args.output_dir)
plt.savefig(out_path, dpi=300)
# Step 6: Show the plot
if args.debug:
    print(f"Plot saved as '{out_path}'")

if args.plot:
    plt.show()