# This is a script to run a named batch of the plotting scripts.
import os
import json
import shlex
import shutil
import argparse
# Add debug and show flags to run_all.py
parser = argparse.ArgumentParser(description='Run a named batch of plotting scripts.')
parser.add_argument('-s', '--scripts', type=str, default='all',
                     help="Name of the script set to run (default: all). "
                          "Must match a file at input/plots/<name>_scripts.txt")
parser.add_argument('-p', '--plot', action='store_true', help='Show plots after running scripts')
parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
args = parser.parse_args()

IMAGES_DIR = "images"
THESIS_FIGURES_DIR = "/home/smanthey/Code/THESIS/figures/xa"
SCRIPT_LIST_DIR = os.path.join("input", "plots")
OUTPUT_PATHS_CONFIG = os.path.join("config", "output_paths.json")

def load_script_list(name):
    list_path = os.path.join(SCRIPT_LIST_DIR, f"{name}_scripts.txt")
    if not os.path.isfile(list_path):
        print(f"Error: no script list found at {list_path}")
        if os.path.isdir(SCRIPT_LIST_DIR):
            available = sorted(
                f[: -len("_scripts.txt")]
                for f in os.listdir(SCRIPT_LIST_DIR) if f.endswith("_scripts.txt")
            )
            if available:
                print(f"       Available sets: {' '.join(available)}")
        raise SystemExit(1)
    with open(list_path) as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

def load_external_outputs(name):
    """External output directories for a script set, from config/output_paths.json.

    Mirrors LOWE_RECONSTRUCTION_PUBLICATION's run_plot_scripts.py: a script
    set can be mapped to one or more external directories that its plots
    should be saved into (via each script's -o/--output-dir flag) instead of
    the local images/ directory.
    """
    if not os.path.isfile(OUTPUT_PATHS_CONFIG):
        return []
    with open(OUTPUT_PATHS_CONFIG) as f:
        config = json.load(f)
    paths = config.get("plots", {}).get(name) or []
    return paths if isinstance(paths, list) else [paths]

def apply_external_outputs(script, external_outputs):
    # Each script's --output-dir takes a single directory (unlike LOWE's
    # nargs='+' --output, which can fan out to several paths at once), so
    # only the first configured path is used if more than one is listed.
    # Only the long flag is checked for an existing override: a few scripts
    # (e.g. xa_pde_variation.py) already use -o for --OV, so a bare "-o"
    # substring match would misfire on those.
    if external_outputs and " --output-dir " not in script:
        if len(external_outputs) > 1:
            print(f"Warning: multiple output paths configured; using the first ({external_outputs[0]})")
        script += f" --output-dir {shlex.quote(external_outputs[0])}"
    return script

def run_script(script_name):
    print(f"Running {script_name}...")
    result = os.system(f"python {os.path.join('scripts', script_name)}")
    if result != 0:
        print(f"Error: {script_name} failed to execute.")
    return result

def copy_plots_to_thesis():
    if not os.path.isdir(IMAGES_DIR):
        print(f"No '{IMAGES_DIR}' directory found; skipping copy to thesis.")
        return
    os.makedirs(THESIS_FIGURES_DIR, exist_ok=True)
    copied = 0
    for fname in os.listdir(IMAGES_DIR):
        if fname.lower().endswith(".png"):
            shutil.copy2(os.path.join(IMAGES_DIR, fname), os.path.join(THESIS_FIGURES_DIR, fname))
            copied += 1
    print(f"Copied {copied} plot(s) to {THESIS_FIGURES_DIR}")

if __name__ == "__main__":
    scripts = load_script_list(args.scripts)
    external_outputs = load_external_outputs(args.scripts)
    scripts = [apply_external_outputs(script, external_outputs) for script in scripts]

    all_results = []
    for script in scripts:
        if args.plot and args.debug:
            this_result = run_script(script + " -p -d")  # Adding -s flag to show plots
        elif args.plot:
            this_result = run_script(script + " -p")
        elif args.debug:
            this_result = run_script(script + " -d")
        else:
            this_result = run_script(script)
        
        if this_result != 0:
            print(f"Script {script} failed. Exiting.")
        all_results.append(this_result)

    if sum(all_results) == 0:
        print("\033[92m" + "All scripts executed successfully." + "\033[0m")
    else:
        for i, result in enumerate(all_results):
            if result != 0:
                print("\033[91m" + f"Script {scripts[i]} encountered an error." + "\033[0m")

    copy_plots_to_thesis()
