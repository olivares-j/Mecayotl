# Mecayotl

Mecayotl is a pipeline to determine stellar group (open cluster) memberships
based on Gaussian-mixture modeling and Bayesian cluster inference (via
Kalkayotl). It coordinates data assembly from a Gaia-like catalogue,
synthetic cluster generation (via Amasijo), model inference and diagnostic
evaluation for probability thresholds.

This repository contains a high-level orchestrator (mecayotl.py) that:
- assembles and prepares data (catalogue + known members),
- fits GMMs to field and cluster samples,
- computes membership probabilities accounting for per-star uncertainties,
- runs synthetic experiments to calibrate classifier thresholds,
- interfaces with Kalkayotl for more involved Bayesian modeling of cluster parameters.

## Requirements and environment

This project uses several scientific Python packages and external repositories:
- Python 3.10 (the code was tested with Python 3.10)
- NumPy, SciPy, pandas, h5py
- Astropy, astroquery
- matplotlib, tqdm
- Amasijo (external repository)
- Kalkayotl (external repository)
- PyGaia (for astrometry utilities)
- The local repository contains a small GMM implementation (gmm.py) used by Mecayotl

A recommended environment creation (example using conda):

conda create -n mecayotl -c conda-forge pymc
conda activate mecayotl

You will also need to clone and place the following repositories in accessible paths:
- Amasijo (used for synthetic cluster generation and classifier quality helpers)
- Kalkayotl (Bayesian inference backend used by cleaning routines)
- Optionally, the directory paths can be passed to Mecayotl at initialization
  (see the example in the example block of mecayotl.py).

Install PyGaia and isochrones into the environment as required by Amasijo/Kalkayotl.

Note: mecayotl.py uses non-interactive matplotlib backend (Agg) and expects to be
run on a system where it can create directories and write HDF5/CSV/PDF outputs.

## Quick usage example

1. Edit paths and filenames in the __main__ example at the bottom of mecayotl.py or
   construct a Mecayotl object from another script, for example:

from mecayotl import Mecayotl

mcy = Mecayotl(
    dir_base="/path/to/output/base",
    file_gaia="/path/to/catalogue.fits",
    file_members="/path/to/members.csv",
    path_ayome="/path/to/Ayome/",
    path_kalkayotl="/path/to/Kalkayotl/",
    seed=12345)

mcy.run(
    iterations=1,
    synthetic_seeds=[0],
    n_cluster_real=int(1e3),
    n_field_real=int(1e3),
    n_samples_syn=int(1e3)
)

2. Inspect output directories created under dir_base/iter_*/ for:
- Real/Data/data.h5 (assembled arrays and computed probabilities)
- Real/Models/* (saved GMM models)
- Classification/members_mecayotl.csv (selected candidates)
- Classification/quality_*.pdf/.tex/.pkl (threshold calibration results)

## Structure of important outputs

- data.h5 contains arrays:
  - ids, mu, sd, cr, ex: observables and their uncertainties/correlations
  - mu_Cluster, sg_Cluster: synthetic cluster means and covariances
  - mu_Field, sg_Field: drawn field subset
  - prob_cls: computed membership probabilities (when produced)

- Models are saved as HDF5 files per (instance, case, n_components) with keys:
  - pros (weights), means, covs, dets, aic, bic, nmn

## Notes and best practices

- Ensure the paths to Amasijo and Kalkayotl are correct and those repositories are installed.
- The pipeline is I/O intensive and may create many files and directories; monitor disk usage.
- For reproducibility, always set seed when instantiating Mecayotl.
- Large runs may require increasing memory or splitting computations (the code supports chunking).
- If you plan to refactor or reuse parts of the code, consider extracting I/O, plotting, and inference
  into separate modules to simplify testing.

## License

Mecayotl is distributed under the GNU General Public License v3 (see the top of mecayotl.py for details).

### Citation
If you use Mecayotl, please cite the following articles.
```
@ARTICLE{2023arXiv230408618O,
       author = {{Olivares}, J. and {Lodieu}, N. and {B{\'e}jar}, V.~J.~S. and {Mart{\'\i}n}, E.~L. and {{\v{Z}}erjal}, M. and {Galli}, P.~A.~B.},
        title = "{The cosmic waltz of Coma Berenices and Latyshev 2 (Group X). Membership, phase-space structure, mass, and energy distributions}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Solar and Stellar Astrophysics, Astrophysics - Astrophysics of Galaxies},
         year = 2023,
        month = apr,
          eid = {arXiv:2304.08618},
        pages = {arXiv:2304.08618},
          doi = {10.48550/arXiv.2304.08618},
archivePrefix = {arXiv},
       eprint = {2304.08618},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv230408618O},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{2020A&A...644A...7O,
       author = {{Olivares}, J. and {Sarro}, L.~M. and {Bouy}, H. and {Miret-Roig}, N. and {Casamiquela}, L. and {Galli}, P.~A.~B. and {Berihuete}, A. and {Tarricq}, Y.},
        title = "{Kalkayotl: A cluster distance inference code}",
      journal = {\aap},
     keywords = {methods: statistical, parallaxes, open clusters and associations: general, stars: distances, virtual observatory tools, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Astrophysics of Galaxies, Astrophysics - Solar and Stellar Astrophysics},
         year = 2020,
        month = dec,
       volume = {644},
          eid = {A7},
        pages = {A7},
          doi = {10.1051/0004-6361/202037846},
archivePrefix = {arXiv},
       eprint = {2010.00272},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020A&A...644A...7O},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

