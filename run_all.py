# This is a script to run all the scripts
import os
import argparse
# Add debug and show flags to run_all.py
parser = argparse.ArgumentParser(description='Run all scripts with debug output.')
parser.add_argument('-p', '--plot', action='store_true', help='Show plots after running scripts')
parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
args = parser.parse_args()

def run_script(script_name):
    print(f"Running {script_name}...")
    result = os.system(f"python {script_name}")
    if result != 0:
        print(f"Error: {script_name} failed to execute.")
    return result

if __name__ == "__main__":
    scripts = [
        "transmittance.py",
        "wvf_example.py",
        "sipm_fit.py -c 4",
        "sipm_fit.py -c 4 5",
        "xa_calibration.py -c 0 1",
        "xa_calibration.py -e -c 0",
        "xa_calibration.py -e -c 1",
        "xa_calibration.py -i Ciemat",
        "xa_calibration_ratio.py",
        "xa_calibration_ratio.py -e -c 0",
        "xa_calibration_ratio.py -e -c 1",
        "xa_pde_variation.py -e",
        "xa_pde_variation.py -e -o 3.5",
        "xa_pde_variation.py -e -o 4.5",
        "xa_pde_variation.py -e -o 7.0",
        "xa_box_variation.py -e",
        "xa_box_variation.py -e -o 3.5",
        "xa_box_variation.py -e -o 4.5",
        "xa_box_variation.py -e -o 7.0",
        "result_comparison.py",
        "result_combined.py"
    ]

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
