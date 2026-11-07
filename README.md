# VD XA PUBLICATION

Plots for the DUNE VD X-ARAPUCA PDE paper, published as G. Botogoske et al., Eur. Phys. J. C 86 (2026) 971,
[doi:10.1140/epjc/s10052-026-16190-1](https://doi.org/10.1140/epjc/s10052-026-16190-1), arXiv:2511.12328.

## This Repository

To set up the environment, run:

```bash
source setup.sh
```

This will create a python environment called `.venv` with all the required dependencies and install the dune_plot_style module to be used in the scripts (checkout [dune_plot_style](https://github.com/DUNE/dune_plot_style) for more details).

## Producing the plots

```bash
python run_all.py -s <set>   # default: all
```

`<set>` selects the list `input/plots/<set>_scripts.txt` (`all`, `calibration`, `misc`, `paper`, `pde`, `results`); each
line is a script in `scripts/` with its arguments, reading the inputs in `data/`. Plots are written to `images/`
(gitignored). A set with a route in `config/output_paths.json` writes there instead: `paper` goes to the paper's `images/`
directory. After each run the PNGs in `images/` are copied to `THESIS/figures/xa`.

For further support please contact [sergio.manthey@ciemat.es](mailto:sergio.manthey@ciemat.es) or [gabrielbotogoske@gmail.com](mailto:gabrielbotogoske@gmail.com)
