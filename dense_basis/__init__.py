from __future__ import print_function, division, absolute_import

"""
Modified dense_basis-like SED fitting toolkit with:
- GP / original dense-basis SFHs (logM, logSFR, tx...)
- Continuity SFHs (logM, nbin, log SFR ratios ...) + per-redshift age bins
"""

__version__ = "0.1.9.dev0.1"
__bibtex__ = """
@article{iyer2019non,
  title={Non-parametric Star Formation History Reconstruction with Gaussian Processes I: Counting Major Episodes of Star Formation},
  author={Iyer, Kartheik G and Gawiser, Eric and Faber, Sandra M and Ferguson, Henry C and Koekemoer, Anton M and Pacifici, Camilla and Somerville, Rachel},
  journal={arXiv preprint arXiv:1901.02877},
  year={2019}
}

@article{Leja_2019,
doi = {10.3847/1538-4357/ab133c},
url = {https://doi.org/10.3847/1538-4357/ab133c},
year = {2019},
month = {apr},
publisher = {The American Astronomical Society},
volume = {876},
number = {1},
pages = {3},
author = {Leja, Joel and Carnall, Adam C. and Johnson, Benjamin D. and Conroy, Charlie and Speagle, Joshua S.},
title = {How to Measure Galaxy Star Formation Histories. II. Nonparametric Models},
journal = {The Astrophysical Journal},
abstract = {Nonparametric star formation histories (SFHs) have long promised to be the “gold standard” for galaxy spectral energy distribution (SED) modeling as they are flexible enough to describe the full diversity of SFH shapes, whereas parametric models rule out a significant fraction of these shapes a priori. However, this flexibility is not fully constrained even with high-quality observations, making it critical to choose a well-motivated prior. Here, we use the SED-fitting code Prospector to explore the effect of different nonparametric priors by fitting SFHs to mock UV–IR photometry generated from a diverse set of input SFHs. First, we confirm that nonparametric SFHs recover input SFHs with less bias and return more accurate errors than do parametric SFHs. We further find that, while nonparametric SFHs robustly recover the overall shape of the input SFH, the primary determinant of the size and shape of the posterior star formation rate as a function of time (SFR(t)) is the choice of prior, rather than the photometric noise. As a practical demonstration, we fit the UV–IR photometry of ∼6000 galaxies from the Galaxy and Mass Assembly survey and measure scatters between priors to be 0.1 dex in mass, 0.8 dex in SFR100 Myr, and 0.2 dex in mass-weighted ages, with the bluest star-forming galaxies showing the most sensitivity. An important distinguishing characteristic for nonparametric models is the characteristic timescale for changes in SFR(t). This difference controls whether galaxies are assembled in bursts or in steady-state star formation, corresponding respectively to (feedback-dominated/accretion-dominated) models of galaxy formation and to (larger/smaller) confidence intervals derived from SED fitting. High-quality spectroscopy has the potential to further distinguish between these proposed models of SFR(t).}
}
"""  # NOQA

# core priors / SFH builders (priors.py imports gp_sfh helpers)
from .priors import *
from .gp_sfh import *

# model generation / grids (uses priors + gp_sfh)
from .pre_grid import *

# fitting + plotting
from .sed_fitter import *
from .plotter import *

# extras
from .mcmc import *
from .parallelization import *
from .basic_fesc import *
from .tests import *

