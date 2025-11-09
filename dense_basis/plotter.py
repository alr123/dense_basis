import numpy as np
import matplotlib.pyplot as plt
import corner

from pylab import *

from .pre_grid import make_filvalkit_simple, load_atlas
from .gp_sfh import (
    tuple_to_sfh,
    correct_for_mass_loss,
    fsps_time,
    fsps_massloss,
    calctimes,
    continuity_to_sfh,
    make_continuity_agebins,
)

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _sfh_from_atlas_row(atlas, idx):
    """
    Return (sfh, timeax) for one model in the atlas, no matter which SFH family
    the atlas was generated with.
    """
    sfh_type = atlas.get('sfh_type', 'gp')
    sfh_tuple = atlas['sfh_tuple'][idx, :]
    zval = atlas['zval'][idx]

    # drop trailing NaNs from padded array
    if np.isnan(sfh_tuple).any():
        sfh_tuple = sfh_tuple[~np.isnan(sfh_tuple)]

    if sfh_type == 'continuity':
        nbin = int(sfh_tuple[1])
        agebins = make_continuity_agebins(zval, nbin)
        log_sfr_ratios = sfh_tuple[2:2 + nbin - 1]
        timeax, sfh, _ = continuity_to_sfh(
            zred=zval,
            logmass=sfh_tuple[0],
            log_sfr_ratios=log_sfr_ratios,
            agebins=agebins
        )
    else:
        sfh, timeax = tuple_to_sfh(sfh_tuple, zval)

    return sfh, timeax


# ---------------------------------------------------------------------
# style
# ---------------------------------------------------------------------
def set_plot_style():
    rc('axes', linewidth=3)
    rcParams['xtick.major.size'] = 12
    rcParams['ytick.major.size'] = 12
    rcParams['xtick.minor.size'] = 9
    rcParams['ytick.minor.size'] = 9
    rcParams['xtick.major.width'] = 3
    rcParams['ytick.major.width'] = 3


# ---------------------------------------------------------------------
# basic plots
# ---------------------------------------------------------------------
def plot_sfh(timeax, sfh, lookback = False, logx = False, logy = False, fig = None, label=None, **kwargs):
    set_plot_style()

    if fig is None:
        fig = plt.figure(figsize=(12,4))
    if lookback == True:
        plt.plot(np.amax(timeax) - timeax, sfh, label=label, **kwargs)
        plt.xlabel('lookback time [Gyr]');
    else:
        plt.plot(timeax, sfh, label=label, **kwargs)
        plt.xlabel('cosmic time [Gyr]');
    if label != None:
        plt.legend(edgecolor='w')
    plt.ylabel(r'SFR(t) [$M_\odot yr^{-1}$]')
    if logx == True:
        plt.xscale('log')
    if logy == True:
        plt.yscale('log')
    plt.xlim(0,np.amax(timeax))
    tempy = plt.ylim()
    plt.ylim(0,tempy[1])
    return fig


def plot_spec(lam, spec, logx = True, logy = True,
              xlim = (1e2,1e8),
              clip_bottom = True):
    set_plot_style()

    plt.figure(figsize=(12,4))
    plt.plot(lam, spec)
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$F_\nu$ [$\mu$Jy]')
    if logx == True:
        plt.xscale('log')
    if logy == True:
        plt.yscale('log')
    plt.xlim(xlim)
    if clip_bottom == True:
        plt.ylim(1e-3,np.amax(spec)*3)
    plt.show()


def plot_filterset(filter_list = 'filter_list_goodss.dat', filt_dir = 'filters/', zval = 1.0, lam_arr = 10**np.linspace(2,8,10000), rest_frame = True):
    set_plot_style()

    filcurves, lam_z, lam_z_lores = make_filvalkit_simple(lam_arr, zval, fkit_name = filter_list, filt_dir = filt_dir)

    plt.figure(figsize=(12,4))
    if rest_frame == True:
        plt.plot(lam_arr*(1+zval), filcurves,'k:');
        plt.xlabel(r'$\lambda$ [$\AA$]');
    else:
        plt.plot(lam_arr, filcurves,'k:');
        plt.xlabel(r'$\lambda$ [$\AA$; obs. frame]');
    plt.xscale('log'); plt.xlim(1e3,2e5)
    plt.ylabel('Filter transmission')
    plt.show()


def quantile_names(N_params):
    return (np.round(np.linspace(0,100,N_params+2)))[1:-1]


# ---------------------------------------------------------------------
# new-aware: atlas priors
# ---------------------------------------------------------------------
def plot_atlas_priors(atlas):
    """
    Corner-plot the *atlas* distribution itself.
    Now supports both gp-style and continuity-style SFHs.
    """
    sfh_type = atlas.get('sfh_type', 'gp')

    mass_unnormed = np.log10(10**atlas['mstar'] / atlas['norm'])
    sfr_unnormed  = np.log10(10**atlas['sfr']  / atlas['norm'])
    ssfr = sfr_unnormed - mass_unnormed

    dust = atlas['dust'].ravel()
    met  = atlas['met'].ravel()
    zval = atlas['zval'].ravel()

    if sfh_type == 'continuity':
        # sfh_tuple_rec: [logM_normed, nbin, log_ratio_0, ...] padded with NaNs
        sfh_tab = atlas['sfh_tuple_rec']
        # keep only columns that have at least one non-NaN
        valid_cols = ~np.all(np.isnan(sfh_tab), axis=0)
        sfh_tab = sfh_tab[:, valid_cols]

        # re-expand first column to unnormalized mass
        sfh_tab[:, 0] = sfh_tab[:, 0] + np.log10(atlas['norm'])

        # build matrix: [logM, logSFR, log sSFR, nbin, ratios..., Z, Av, z]
        quants = [mass_unnormed, sfr_unnormed, ssfr]
        for j in range(sfh_tab.shape[1]):
            quants.append(sfh_tab[:, j])
        quants.extend([met, dust, zval])
        quants = np.vstack(quants).T

        labels = ['log M*', 'log SFR', 'log sSFR', 'nbin']
        # ratios start at col 2 in tuple
        n_ratios = sfh_tab.shape[1] - 2
        for k in range(n_ratios):
            labels.append(f'log SFR ratio {k+1}')
        labels.extend(['Z', 'Av', 'z'])

    else:
        # original behavior
        txs = atlas['sfh_tuple_rec'][:, 3:]
        quants = np.vstack((mass_unnormed, sfr_unnormed, ssfr, txs.T, met, dust, zval)).T

        tx_names = ['t'+'%.0f' %i for i in quantile_names(txs.shape[1])]
        labels = ['log M*', 'log SFR', 'log sSFR', 'Z', 'Av', 'z']
        labels[3:3] = tx_names

    figure = corner.corner(
        quants,
        plot_datapoints=False,
        fill_contours=True,
        labels=labels,
        bins=20,
        smooth=1.0,
        quantiles=(0.16, 0.84),
        levels=[1 - np.exp(-(1/1)**2/2),1 - np.exp(-(2/1)**2/2)],
        label_kwargs={"fontsize": 30}
    )
    figure.subplots_adjust(right=1.5, top=1.5)
    plt.show()
    return


# ---------------------------------------------------------------------
# new-aware: posterior corner
# ---------------------------------------------------------------------
def plot_posteriors(chi2_array, norm_fac, sed, atlas, truths = [], **kwargs):
    """
    Corner-plot the *posterior* given chi^2 + atlas.
    """
    set_plot_style()

    sfh_type = atlas.get('sfh_type', 'gp')

    # base arrays
    mstar = atlas['mstar'] + np.log10(norm_fac)
    sfr   = atlas['sfr']   + np.log10(norm_fac)
    met   = atlas['met'].ravel()
    dust  = atlas['dust'].ravel()
    zval  = atlas['zval'].ravel()

    if sfh_type == 'continuity':
        sfh_tab = atlas['sfh_tuple']
        valid_cols = ~np.all(np.isnan(sfh_tab), axis=0)
        sfh_tab = sfh_tab[:, valid_cols]

        # col0 is tuple mass → also add norm_fac to show actual mass prior
        sfh_tab[:, 0] = sfh_tab[:, 0] + np.log10(norm_fac)

        # stack: mstar, sfr, (continuity columns...), Z, Av, z
        cols = [mstar, sfr]
        for j in range(sfh_tab.shape[1]):
            cols.append(sfh_tab[:, j])
        cols.extend([met, dust, zval])
        corner_params = np.vstack(cols)

        labels = ['log M*', 'log SFR', 'nbin']
        n_extra = sfh_tab.shape[1] - 2
        for k in range(n_extra):
            labels.append(f'log SFR ratio {k+1}')
        labels.extend(['Z', 'Av', 'z'])

    else:
        # gp case, your original
        txs = atlas['sfh_tuple'][:, 3:].T
        sfrvals = atlas['sfr'].copy()
        sfrvals[sfrvals < -3] = -3
        corner_params = np.vstack([
            mstar,
            sfrvals + np.log10(norm_fac),
            txs,
            met,
            dust,
            zval
        ])

        tx_names = ['t'+'%.0f' %i for i in quantile_names(corner_params.shape[0]-5)]
        labels = ['log M*', 'log SFR', 'Z', 'Av', 'z']
        labels[2:2] = tx_names

    weights = np.exp(-chi2_array/2)

    if len(truths) > 0:
        figure = corner.corner(
            corner_params.T,
            weights=weights,
            labels=labels,
            truths=truths,
            plot_datapoints=False,
            fill_contours=True,
            bins=20,
            smooth=1.0,
            quantiles=(0.16, 0.84),
            levels=[1 - np.exp(-(1/1)**2/2),1 - np.exp(-(2/1)**2/2)],
            label_kwargs={"fontsize": 30},
            show_titles=True,
            **kwargs
        )
    else:
        figure = corner.corner(
            corner_params.T,
            weights=weights,
            labels=labels,
            plot_datapoints=False,
            fill_contours=True,
            bins=20,
            smooth=1.0,
            quantiles=(0.16, 0.84),
            levels=[1 - np.exp(-(1/1)**2/2),1 - np.exp(-(2/1)**2/2)],
            label_kwargs={"fontsize": 30},
            show_titles=True,
            **kwargs
        )

    figure.subplots_adjust(right=1.5, top=1.5)
    return figure


# ---------------------------------------------------------------------
# prior plotting from file (kept, but note filename pattern)
# ---------------------------------------------------------------------
def plot_priors(fname, N_pregrid, N_param, dir = 'pregrids/'):
    """
    This is the older helper that expects the pregrid filename pattern.
    You can make a version that just passes the full path if you’re now saving
    atlases with continuity-based suffixes.
    """
    set_plot_style()

    cat = load_atlas(fname, N_pregrid, N_param, path = dir)
    sfh_tuples = cat['sfh_tuple']
    Av = cat['dust'].ravel()
    Z = cat['met'].ravel()
    z = cat['zval'].ravel()
    seds = cat['sed']
    norm_method = cat['norm_method']
    norm_facs = cat['norm'].ravel()

    pg_theta = [sfh_tuples, Z, Av, z, seds]
    pg_params = np.vstack([pg_theta[0][0,0:], pg_theta[0][1,0:], pg_theta[0][3:,0:], pg_theta[1], pg_theta[2], pg_theta[3]])

    txs = ['t'+'%.0f' %i for i in quantile_names(pg_params.shape[0]-5)]
    pg_labels = ['log M*', 'log SFR', 'Z', 'Av', 'z']
    pg_labels[2:2] = txs

    pg_priors = pg_params.copy()
    pg_priors[0,0:] += np.log10(norm_facs)
    pg_priors[1,0:] += np.log10(norm_facs)
    figure = corner.corner(pg_priors.T,labels=pg_labels, plot_datapoints=False, fill_contours=True,
            bins=50, smooth=1.0, levels=[1 - np.exp(-(1/1)**2/2),1 - np.exp(-(2/1)**2/2)], label_kwargs={"fontsize": 30})
    figure.subplots_adjust(right=1.5,top=1.5)
    plt.show()


# ---------------------------------------------------------------------
# SFH posterior (this is the one you’ll most likely want to call)
# ---------------------------------------------------------------------
def plot_SFH_posterior(chi2_array, norm_fac, sed, atlas, truths = [], plot_ci = True, sfh_threshold = 0.9, **kwargs):
    """
    Same idea as your old function, but now supports continuity atlases.
    """
    set_plot_style()

    weights = np.exp(-chi2_array/2)
    order = np.argsort(weights)

    # choose a common time axis
    best_idx = order[-1]
    best_sfh, best_time = _sfh_from_atlas_row(atlas, best_idx)
    common_time = best_time  # Gyr, forward in time

    # pick models above a likelihood threshold
    max_like = weights[best_idx]
    keep = weights >= sfh_threshold * max_like
    kept_indices = np.where(keep)[0]
    if len(kept_indices) == 0:
        kept_indices = order[-50:]  # just take top 50

    gathered = []
    gathered_w = []
    for idx in kept_indices:
        sfh, timeax = _sfh_from_atlas_row(atlas, idx)
        sfh = sfh * norm_fac
        sfh_interp = np.interp(common_time, timeax, sfh)
        gathered.append(sfh_interp)
        gathered_w.append(weights[idx])

    gathered = np.array(gathered)
    gathered_w = np.array(gathered_w)

    if plot_ci:
        # compute weighted percentiles in log-space
        sfh_50 = np.zeros_like(common_time)
        sfh_16 = np.zeros_like(common_time)
        sfh_84 = np.zeros_like(common_time)

        for ti in range(len(common_time)):
            this = gathered[:, ti]
            mask = (this > 0) & np.isfinite(this)
            if np.sum(mask) == 0:
                continue
            vals = np.log10(this[mask])
            wts = gathered_w[mask]

            # simple weighted CDF
            bins = 50
            h, edges = np.histogram(vals, weights=wts, bins=bins)
            cdf = np.cumsum(h)
            cdf = cdf / cdf[-1]
            centers = edges[:-1] + np.diff(edges)/2

            def wp(p):
                return 10**centers[np.argmin(np.abs(cdf - p))]
            sfh_50[ti] = wp(0.5)
            sfh_16[ti] = wp(0.16)
            sfh_84[ti] = wp(0.84)

        fig = plt.figure(figsize=(12,4))
        plt.plot(np.amax(common_time) - common_time, sfh_50, lw=3, color='k', label='median SFH')
        plt.fill_between(np.amax(common_time) - common_time, sfh_16, sfh_84, color='k', alpha=0.2)

    else:
        fig = plt.figure(figsize=(12,4))
        for sfh_interp, wt in zip(gathered, gathered_w):
            plt.plot(np.amax(common_time) - common_time, sfh_interp, color='k', alpha=0.1 + 0.8*(wt/np.max(gathered_w)))

    if len(truths) == 2:
        plot_sfh(truths[0], truths[1], lookback=True, fig=fig, lw=3, label='true SFH')
        plt.ylim(0, np.amax(truths[1])*1.5)

    plt.xlabel('lookback time [Gyr]')
    plt.ylabel(r'$SFR(t)$ [M$_\odot$/yr]')
    plt.legend(edgecolor='w')
    plt.show()
    return

