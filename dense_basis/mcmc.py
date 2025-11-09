# mcmc.py

import numpy as np
from tqdm import tqdm
import scipy.io as sio

try:
    import emcee
    from scipy.interpolate import NearestNDInterpolator
except Exception:
    print('running without emcee')

from .priors import *
from .gp_sfh import *
from .plotter import *
from .sed_fitter import evaluate_sed_likelihood


def get_mcmc_variables(atlas):
    """
    Build the parameter array and interpolation engine for MCMC.

    For classic / gp atlases (default):
        params = [logM, logSFR, t25, t50, t75, Z, Av, z]
    For continuity atlases:
        params = [logM, logSFR, Z, Av, z]

    We return (model_params, model_seds, interp_engine, param_limits, sfh_type)
    so the rest of the code knows which layout we used.
    """
    # detect SFH family
    sfh_type = atlas.get('sfh_type', 'gp')

    mstar = atlas['mstar']
    sfr = atlas['sfr']
    met = atlas['met'].ravel()
    dust = atlas['dust'].ravel()
    zval = atlas['zval'].ravel()
    model_seds = atlas['sed']

    if sfh_type == 'continuity':
        # simpler, fixed-length param vector
        model_params = np.vstack((mstar, sfr, met, dust, zval)).T
        interp_engine = NearestNDInterpolator(model_params, model_seds)
        param_min = np.amin(model_params, axis=0)
        param_max = np.amax(model_params, axis=0)
        param_limits = [param_min, param_max]
        return model_params, model_seds, interp_engine, param_limits, sfh_type

    else:
        # original dense_basis assumption: last 3 columns of sfh_tuple are t25, t50, t75
        # our pre_grid.py still produces fixed-length arrays for gp-like atlases
        sfh_tab = atlas['sfh_tuple']
        # grab the last 3 columns as times
        tx = sfh_tab[:, -3:].copy()
        # sometimes you may have tiny negative or NaN values; clip them
        tx = np.nan_to_num(tx, nan=0.0)
        model_params = np.vstack((mstar, sfr, tx.T, met, dust, zval)).T

        interp_engine = NearestNDInterpolator(model_params, model_seds)
        param_min = np.amin(model_params, axis=0)
        param_max = np.amax(model_params, axis=0)

        # keep some of your old floor logic
        # param order: [mass, sfr, t25, t50, t75, met, dust, z]
        param_min[1] = -3.0
        param_min[2] = 0.0

        param_limits = [param_min, param_max]

        return model_params, model_seds, interp_engine, param_limits, sfh_type


def log_prior(theta, param_limits, sfh_type):
    """
    Flat prior inside the atlas bounds, with extra ordering constraints
    for the gp-style t25<t50<t75 case.
    """
    theta = np.asarray(theta)
    if sfh_type == 'continuity':
        # theta = [mass, sfr, Z, Av, z]
        if (param_limits[0] < theta).all() and (theta < param_limits[1]).all():
            return 0.0
        return -np.inf
    else:
        # theta = [mass, sfr, t25, t50, t75, Z, Av, z]
        mass, sfr, t25, t50, t75, met, dust, zval = theta
        if (
            (param_limits[0] < theta).all()
            and (theta < param_limits[1]).all()
            and (0 < t25 < t50)
            and (t25 < t50 < t75)
        ):
            return 0.0
        return -np.inf


def log_likelihood(theta, sed, sed_err, interp_engine):
    """
    Gaussian likelihood using nearest-neighbor interpolation in model space.
    """
    pred_sed = interp_engine(theta)[0]
    fit_mask = (sed > 0)
    chi2 = (pred_sed[fit_mask] - sed[fit_mask])**2 / (sed_err[fit_mask])**2
    return -0.5 * np.sum(chi2)


def log_probability(theta, sed, sed_err, interp_engine, param_limits, sfh_type):
    lp = log_prior(theta, param_limits, sfh_type)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, sed, sed_err, interp_engine)


def run_emceesampler(
    gal_sed,
    gal_err,
    atlas,
    fit_mask=[],
    zbest=None,
    deltaz=None,
    nwalkers=100,
    epochs=1000,
    plot_posteriors=False
):
    """
    Run an emcee sampler on a single galaxy given an atlas.

    This auto-detects whether the atlas is 'gp' or 'continuity' and
    builds the right parameter vector.
    """
    model_params, model_seds, interp_engine, param_limits, sfh_type = get_mcmc_variables(atlas)

    # initial positions: pick first nwalkers models
    pos = model_params[0:nwalkers, :].copy()
    # simple fix for very low SFR initial points
    if pos.shape[1] > 1:
        pos[pos[:, 1] < -2, 1] = -2

    _, ndim = pos.shape

    # get normalization like your sed_fitter path
    _, norm_fac = evaluate_sed_likelihood(
        gal_sed,
        gal_err,
        atlas,
        fit_mask=[],
        zbest=None,
        deltaz=None,
        dynamic_norm=True
    )

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(gal_sed / norm_fac, gal_err / norm_fac, interp_engine, param_limits, sfh_type)
    )
    sampler.run_mcmc(pos, epochs, progress=True)

    if plot_posteriors:
        plot_emcee_posterior(sampler, norm_fac)

    return sampler, norm_fac


def plot_emcee_posterior(sampler, norm_fac, discard=100, thin=1):
    """
    Corner plot of the MCMC chain. We add back the normalization to logM, logSFR.
    """
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)

    # always: first two entries are logM and logSFR
    flat_samples[:, 0] = flat_samples[:, 0] + np.log10(norm_fac)
    flat_samples[:, 1] = flat_samples[:, 1] + np.log10(norm_fac)

    ndim = flat_samples.shape[1]

    if ndim == 5:
        # continuity case: [log M*, log SFR, Z, Av, z]
        atlas_labels = ['log M*', 'log SFR', 'Z', 'Av', 'z']
    else:
        # gp case: [log M*, log SFR, t25, t50, t75, Z, Av, z]
        atlas_labels = ['log M*', 'log SFR', 't25', 't50', 't75', 'Z', 'Av', 'z']

    fig = corner.corner(
        flat_samples,
        labels=atlas_labels,
        plot_datapoints=False,
        fill_contours=True,
        bins=20,
        smooth=1.0,
        quantiles=(0.16, 0.84),
        levels=[1 - np.exp(-(1/1)**2/2), 1 - np.exp(-(2/1)**2/2)],
        label_kwargs={"fontsize": 30},
        show_titles=True
    )
    fig.subplots_adjust(right=1.5, top=1.5)
    plt.show()

