"""Reference-method uncertainty budget for the PDE measurement.

Computes the combined relative uncertainty of the reference-method PDE,
correctly separating terms that are correlated (systematic, tied to fixed
hardware/geometry shared across repeated measurements of the same
configuration) from terms that are genuinely independent per data-taking
campaign (statistical photon-counting fits).

This is the calculation behind the DF-XA-SS row of Table 7 in the paper
(pde_xa_vd_paper/4.1.Configuration_Comparison.tex), the only configuration
measured in two independent CIEMAT campaigns (DF-XA_1, DF-XA_2). A previous,
undocumented hand calculation appears to have divided the *entire* combined
uncertainty (including the correlated terms) by sqrt(2), which is not valid
for terms that do not vary between the two runs.
"""

import numpy as np

# Six-term uncertainty budget (relative, percent). See
# 4.1.Configuration_Comparison.tex for the definition of each term.
CORRELATED_REL = {
    "Reference SiPM PDE": 8.7,   # fixed value from an external publication
    "Reference SiPM XT factor": 2.0,  # intrinsic hardware property of the reference SiPMs
    "Geometric factor": 1.5,     # single simulation of the fixed box geometry
    "XA XT factor": 1.0,         # intrinsic hardware property of the XA prototype
}

UNCORRELATED_REL = {
    "Reference SiPM PE count": 2.0,  # per-campaign counting-statistics fit
    "XA PE count": 1.0,              # per-campaign counting-statistics fit
}


def quadrature(values):
    return float(np.sqrt(np.sum(np.square(values))))


def main():
    correlated = quadrature(list(CORRELATED_REL.values()))
    uncorrelated = quadrature(list(UNCORRELATED_REL.values()))
    total_single = quadrature([correlated, uncorrelated])
    total_twice = quadrature([correlated, uncorrelated / np.sqrt(2)])

    print("Correlated (systematic) terms:")
    for name, value in CORRELATED_REL.items():
        print(f"  {name}: {value}%")
    print(f"  -> combined: {correlated:.2f}%\n")

    print("Uncorrelated (per-campaign statistical) terms:")
    for name, value in UNCORRELATED_REL.items():
        print(f"  {name}: {value}%")
    print(f"  -> combined: {uncorrelated:.2f}%\n")

    print(f"Single-measurement total (all six terms): {total_single:.2f}%")
    print(
        "Twice-measured (DF-XA-SS) total (uncorrelated part reduced by "
        f"sqrt(2), correlated part unchanged): {total_twice:.2f}%\n"
    )

    # DF-XA-SS central values from the box-weighted average, unchanged by
    # this fix (see scripts/xa_box_variation.py).
    ov = [3.5, 4.5, 7.0]
    central = [3.2, 3.7, 4.7]
    print("DF-XA-SS corrected absolute errors:")
    for ov_i, c in zip(ov, central):
        err = c * total_twice / 100
        print(f"  OV {ov_i}: {c} +/- {err:.3f} -> rounded {round(err, 1)}")


if __name__ == "__main__":
    main()
