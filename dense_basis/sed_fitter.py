import numpy as np
from tqdm import tqdm
import scipy.io as sio

import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import gaussian_kde
from scipy.optimize import minimize

# cosmology assumption
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

from .priors import *
from .pre_grid import *
from .gp_sfh import *
from .plotter import *


def _sfh_from_atlas_row(atlas, idx):
    """
    Return (sfh, timeax) for a single atlas entry, regardless of whether the atlas
    was generated with the GP SFH or the continuity SFH.
    """
    sfh_type = atlas.get('sfh_type', 'gp')
    sfh_tuple = atlas['sfh_tuple'][idx, :]
    zval = atlas['zval'][idx]

    # strip trailing NaNs
    if np.isnan(sfh_tuple).any():
        sfh_tuple = sfh_tuple[~np.isnan(sfh_tuple)]

    if sfh_type == 'continuity':
        # continuity tuple: [logM, nbin, log_sfr_ratios...]
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
        # original DB / GP SFH
        sfh, timeax = tuple_to_sfh(sfh_tuple, zval)

    return sfh, timeax


class SedFit(object):
    """
    Class to incorporate SED likelihood evaluation and resulting posteriors
    """

    def __init__(self, sed, sed_err, atlas, fit_mask = [], zbest = None, deltaz = None):

        self.sed = sed
        self.sed_err = sed_err
        self.atlas = atlas
        self.fit_mask = fit_mask
        self.zbest = zbest
        self.deltaz = deltaz
        self.dynamic_norm = True

    def evaluate_likelihood(self):

        chi2_array, norm_fac = evaluate_sed_likelihood(
            self.sed, self.sed_err, self.atlas,
            self.fit_mask, self.zbest, self.deltaz, self.dynamic_norm
        )
        self.chi2_array = chi2_array
        self.norm_fac = norm_fac
        self.likelihood = np.exp(-(chi2_array)/2)

        return

    def evaluate_posterior_percentiles(self, bw_dex = 0.001, percentile_values = [50.,16.,84.], vb = False):
        """
        by default, the percentile values are median, lower68, upper68.
        change this to whatever the desired sampling of the posterior is.
        """

        quants = get_quants(
            self.chi2_array,
            self.atlas,
            self.norm_fac,
            bw_dex = bw_dex,
            percentile_values = percentile_values,
            vb = vb
        )

        # add in colours here
        nuvu = get_quants_key('nuvu', np.arange(-0.5,6,0.001), self.chi2_array,self.atlas,self.norm_fac)
        uv = get_quants_key('uv', np.arange(-0.5,6,0.001), self.chi2_array,self.atlas,self.norm_fac)
        vj = get_quants_key('vj', np.arange(-0.5,6,0.001), self.chi2_array,self.atlas,self.norm_fac)
        nuvr = get_quants_key('nuvr', np.arange(-0.5,6,0.001), self.chi2_array,self.atlas,self.norm_fac)
        rj = get_quants_key('rj', np.arange(-0.5,6,0.001), self.chi2_array,self.atlas,self.norm_fac)

        self.nuvu = nuvu
        self.uv = uv
        self.vj = vj
        self.nuvr = nuvr
        self.rj = rj
        self.mstar = quants[0]
        self.sfr = quants[1]
        self.Av = quants[2]
        self.Z = quants[3]
        self.z = quants[4]
        self.sfh_tuple = quants[5]
        self.percentile_values = percentile_values

        return

    def evaluate_MAP_mstar(self, bw_dex = 0.001, smooth = 'kde', lowess_frac = 0.3, bw_method = 'scott', vb = False):

        qty = self.atlas['mstar'] + np.log10(self.norm_fac)
        weights = self.likelihood
        bins = np.arange(4,14,bw_dex)
        self.mstar_MAP = evaluate_MAP(qty, weights, bins, smooth = smooth, lowess_frac=lowess_frac, bw_method=bw_method, vb=vb)
        return self.mstar_MAP

    def evaluate_MAP_sfr(self, bw_dex = 0.001, smooth = 'kde', lowess_frac = 0.3, bw_method = 'scott', vb = False):

        qty = self.atlas['sfr'] + np.log10(self.norm_fac)
        weights = self.likelihood
        bins = np.arange(-6,4,bw_dex)
        self.sfr_MAP = evaluate_MAP(qty, weights, bins, smooth = smooth, lowess_frac=lowess_frac, bw_method=bw_method, vb=vb)
        return self.sfr_MAP

    def plot_posteriors(self,truths = []):

        figure = plot_posteriors(self.chi2_array, self.norm_fac, self.sed, self.atlas, truths = truths)
        return figure

    def plot_posterior_spec(self, filt_centers, priors, ngals = 100, alpha=0.1, fnu=True, yscale='log', speccolor = 'k', sedcolor='b', titlestr = [],figsize=(12,7)):

        set_plot_style()

        lam_all = []
        spec_all = []
        z_all = []

        bestn_gals = np.argsort(self.likelihood)

        for i in range(ngals):

            lam_gen, spec_gen =  makespec_atlas(self.atlas, bestn_gals[-(i+1)], priors, mocksp, cosmo, filter_list = [], filt_dir = [], return_spec = True)

            lam_all.append(lam_gen)
            spec_all.append(spec_gen)
            z_all.append(self.atlas['zval'][bestn_gals[-(i+1)]])

        fig = plt.subplots(1,1,figsize=figsize)

        if fnu == True:
            for i in range(ngals):
                plt.plot(lam_all[i]*(1+z_all[i]), spec_all[i]*self.norm_fac, color = speccolor, alpha=alpha)
            plt.errorbar(filt_centers[self.sed>0], self.sed[self.sed>0], yerr=self.sed_err[self.sed>0]*2, color=sedcolor,lw=0, elinewidth=2, marker='o', markersize=12, capsize=5)
            plt.ylabel(r'$F_\nu$ [$\mu$Jy]')

        elif fnu == False:
            for i in range(ngals):
                spec_flam = ujy_to_flam(spec_all[i]*self.norm_fac, lam_all[i]*(1+z_all[i]))
                plt.plot(lam_all[i]*(1+z_all[i]), spec_flam, color = speccolor, alpha=alpha)
            sed_flam = ujy_to_flam(self.sed,filt_centers)
            sed_flam_err_up = ujy_to_flam(self.sed+self.sed_err,filt_centers) - sed_flam
            sed_flam_err_dn = sed_flam - ujy_to_flam(self.sed-self.sed_err,filt_centers)
            plt.errorbar(
                filt_centers[self.sed>0],
                sed_flam[self.sed>0],
                yerr=(sed_flam_err_up[self.sed>0], sed_flam_err_dn[self.sed>0]),
                color=sedcolor,lw=0, elinewidth=2, marker='o', markersize=12, capsize=5
            )
            plt.ylabel(r'$F_\lambda$')

        plt.xlabel(r'$\lambda$ [$\AA$]')
        plt.xlim(np.amin(filt_centers)*0.81, np.amax(filt_centers)*1.2)
        plt.ylim(np.amin(self.sed[self.sed>0])*0.8,np.amax(self.sed[self.sed>0]+self.sed_err[self.sed>0])*1.5)
        plt.xscale('log');plt.yscale(yscale);

        return fig

    def evaluate_posterior_SFH(self, zval, ngals=100):

        bestn_gals = np.argsort(self.likelihood)
        common_time = np.linspace(0,cosmo.age(zval).value,100)

        all_sfhs = []
        all_weights = []
        for i in range(ngals):
            sfh, timeax = _sfh_from_atlas_row(self.atlas, bestn_gals[-(i+1)])
            sfh = sfh * self.norm_fac
            sfh_interp = np.interp(common_time, timeax, sfh)

            all_sfhs.append(sfh_interp)
            all_weights.append(self.likelihood[bestn_gals[-(i+1)]])

        all_sfhs = np.array(all_sfhs)
        all_weights = np.array(all_weights)

        sfh_50 = np.zeros_like(common_time)
        sfh_16 = np.zeros_like(common_time)
        sfh_84 = np.zeros_like(common_time)
        for ti in range(len(common_time)):
            qty = np.log10(all_sfhs[0:,ti])
            qtymask = (qty > -np.inf) & (~np.isnan(qty))
            if np.sum(qtymask) > 0:
                smallwts = all_weights.copy()[qtymask.ravel()]
                qty = qty[qtymask]
                if len(qty) > 0:
                    sfh_50[ti], sfh_16[ti], sfh_84[ti] = 10**calc_percentiles(
                        qty, smallwts, bins=50, percentile_values=[50., 16., 84.]
                    )

        return sfh_50, sfh_16, sfh_84, common_time

    def plot_posterior_SFH(self, zval,ngals=100,alpha=0.1, speccolor = 'k', sedcolor='b',figsize=(12,7)):

        set_plot_style()
        bestn_gals = np.argsort(self.likelihood)

        fig = plt.subplots(1,1,figsize=figsize)

        common_time = np.linspace(0,cosmo.age(zval).value,100)

        all_sfhs = []
        all_weights = []
        for i in range(ngals):
            sfh, timeax = _sfh_from_atlas_row(self.atlas, bestn_gals[-(i+1)])
            sfh = sfh * self.norm_fac
            sfh_interp = np.interp(common_time, timeax, sfh)
            all_sfhs.append(sfh_interp)
            all_weights.append(self.likelihood[bestn_gals[-(i+1)]])
            alphawt = 1.0*alpha*self.likelihood[bestn_gals[-(i+1)]]/self.likelihood[bestn_gals[-1]]
            plt.plot(np.amax(common_time)-common_time, sfh_interp, color = sedcolor, alpha=alphawt)

        all_sfhs = np.array(all_sfhs)
        all_weights = np.array(all_weights)

        sfh_50 = np.zeros_like(common_time)
        sfh_16 = np.zeros_like(common_time)
        sfh_84 = np.zeros_like(common_time)
        for ti in range(len(common_time)):
            qty = np.log10(all_sfhs[0:,ti])
            qtymask = (qty > -np.inf) & (~np.isnan(qty))
            if np.sum(qtymask) > 0:
                smallwts = all_weights.copy()[qtymask.ravel()]
                qty = qty[qtymask]
                if len(qty) > 0:
                    sfh_50[ti], sfh_16[ti], sfh_84[ti] = 10**calc_percentiles(
                        qty, smallwts, bins=50, percentile_values=[50., 16., 84.]
                    )

        plt.plot(np.amax(common_time)-common_time, sfh_50,lw=3,color=speccolor)
        plt.fill_between(
            np.amax(common_time)-common_time.ravel(),
            sfh_16.ravel(),
            sfh_84.ravel(),
            alpha=0.3,color=speccolor
        )

        plt.xlabel(r'lookback time [$Gyr$]')
        plt.ylabel(r'$SFR(t)$ [M$_\odot /$yr]')
        try:
            plt.ylim(0,np.amax(sfh_84)*1.2)
        except:
            print('couldnt set axis limits')

        return fig


#-------------------------------------------------------------


def normerr(nf, pg_seds, sed, sed_err, fit_mask):
    c2v = np.amin(np.mean((pg_seds[fit_mask,0:] - sed.reshape(-1,1)/nf)**2 / (sed_err.reshape(-1,1)/nf)**2, 0))
    return c2v

def evaluate_sed_likelihood(sed, sed_err, atlas, fit_mask = [], zbest = None, deltaz = None, dynamic_norm = True):

    """
    Evaluate the likeihood of model SEDs in an atlas given
    an observed SED with uncertainties.
    """

    # preprocessing:
    if len(fit_mask) == len(sed):
        fit_mask = fit_mask & (sed > 0)
    else:
        fit_mask = (sed > 0)

    if len(sed) != len(sed_err):
        raise ValueError('SED uncertainty array does not match SED')

    sed = sed[fit_mask]
    sed_err = sed_err[fit_mask]
    pg_seds = atlas['sed'].copy().T

    if zbest is not None:
        pg_z = atlas['zval'].ravel()
        redshift_mask = (np.abs(pg_z - zbest) < deltaz)
        chi2 = np.zeros((len(pg_z),))
        chi2[~redshift_mask] = 1e10

        if np.sum(redshift_mask) == 0:
            print('atlas does not contain any models in redshift range')

    if dynamic_norm == True:
        if zbest is not None:
            nfmin = minimize(
                normerr,
                np.nanmedian(sed),
                args = (pg_seds[0:,redshift_mask], sed, sed_err, fit_mask)
            )
        else:
            nfmin = minimize(
                normerr,
                np.nanmedian(sed),
                args = (pg_seds, sed, sed_err, fit_mask)
            )
        norm_fac = nfmin['x'][0]
    elif dynamic_norm == False:
        norm_fac = np.nanmedian(sed)
    else:
        norm_fac = 1.0
        print('undefined norm method. using norm_fac = 1')

    sed_normed = sed.reshape(-1,1)/norm_fac
    sed_err_normed = sed_err.reshape(-1,1)/norm_fac

    if zbest is not None:
        chi2[redshift_mask] = np.mean(
            (pg_seds[fit_mask,0:][0:,redshift_mask] - sed_normed)**2 / (sed_err_normed)**2,
            0
        )
    else:
        chi2 = np.mean((pg_seds[fit_mask,0:] - sed_normed)**2 / (sed_err_normed)**2, 0)

    return chi2, norm_fac

def get_quants_key(key, bins, chi2_array, cat, norm_fac, percentile_values = [50.,16.,84.], return_uncert = True, vb = False):
    """
    Get posterior percentiles for an input key
    """
    relprob = np.exp(-(chi2_array)/2)
    key_vals = calc_percentiles(cat[key], weights = relprob, bins = bins, percentile_values = percentile_values, vb = vb)

    return key_vals


def get_quants(chi2_array, cat, norm_fac, bw_dex = 0.001, percentile_values = [50.,16.,84.], vb = False):

    """
    remember to check bin limits and widths before using quantities if you're fitting a new sample
    """

    relprob = np.exp(-(chi2_array)/2)
    if vb == True:
        plt.hist(relprob,100)
        plt.yscale('log')
        plt.show()

    # ---------------- stellar mass and SFR -----------------------------------

    mstar_vals = calc_percentiles(
        cat['mstar'] + np.log10(norm_fac),
        weights = relprob,
        bins = np.arange(4,14,bw_dex),
        percentile_values = percentile_values, vb=vb
    )

    sfr_vals = calc_percentiles(
        cat['sfr'] + np.log10(norm_fac),
        weights = relprob,
        bins = np.arange(-6,4,bw_dex),
        percentile_values = percentile_values, vb=vb
    )

    # ---------------- SFH tuple -----------------------------------
    sfh_type = cat.get('sfh_type', 'gp')
    sfh_tuple_rec = cat['sfh_tuple_rec']
    ncols = sfh_tuple_rec.shape[1]
    sfh_tuple_vals = np.zeros((len(percentile_values), ncols))

    if sfh_type == 'continuity':
        for i in range(ncols):
            col = sfh_tuple_rec[:, i]
            mask = ~np.isnan(col)
            col = col[mask]
            wts = relprob[mask]

            if i == 0:
                # logM in the tuple (already normalized in pregrid, so add norm_fac)
                bins = np.arange(4,14,bw_dex)
                vals = calc_percentiles(col + np.log10(norm_fac), wts, bins, percentile_values, vb=vb)
            elif i == 1:
                # nbin: treat as small integer
                bins = np.arange(1, 21, 1)
                vals = calc_percentiles(col, wts, bins, percentile_values, vb=vb)
            else:
                # log SFR ratios ~ around 0
                bins = np.arange(-5, 5, bw_dex)
                vals = calc_percentiles(col, wts, bins, percentile_values, vb=vb)

            sfh_tuple_vals[:, i] = vals
    else:
        # original behavior for GP tuples
        for i in range(ncols):
            col = sfh_tuple_rec[:, i]
            mask = ~np.isnan(col)
            col = col[mask]
            wts = relprob[mask]

            if i == 0:
                vals = calc_percentiles(col + np.log10(norm_fac),
                                        weights = wts,
                                        bins = np.arange(4,14,bw_dex),
                                        percentile_values = percentile_values, vb=vb)
            elif i == 1:
                vals = calc_percentiles(col + np.log10(norm_fac),
                                        weights = wts,
                                        bins = np.arange(-6,4,bw_dex),
                                        percentile_values = percentile_values, vb=vb)
            elif i == 2:
                # number of tx — keep as mean as in your original
                vals = sfh_tuple_vals[:, i] + np.nanmean(col)
            else:
                vals = calc_percentiles(col,
                                        weights = wts,
                                        bins = np.arange(0,1,bw_dex),
                                        percentile_values = percentile_values, vb=vb)
            sfh_tuple_vals[:, i] = vals

    # ------------------------ dust, metallicity, redshift ----------------------

    Av_vals = calc_percentiles(
        cat['dust'].ravel(),
        weights = relprob,
        bins = np.arange(0,np.amax(cat['dust']),bw_dex),
        percentile_values = percentile_values, vb=vb
    )

    Z_vals = calc_percentiles(
        cat['met'].ravel(),
        weights = relprob,
        bins = np.arange(-1.5, 0.5,bw_dex),
        percentile_values = percentile_values, vb=vb
    )

    z_vals = calc_percentiles(
        cat['zval'].ravel(),
        weights = relprob,
        bins = np.arange(np.amin(cat['zval']), np.amax(cat['zval']),bw_dex),
        percentile_values = percentile_values, vb=vb
    )

    return [mstar_vals, sfr_vals, Av_vals, Z_vals, z_vals, sfh_tuple_vals]


def get_flat_posterior(qty, weights, bins):

    post, xaxis = np.histogram(qty, weights=weights, bins=bins)
    prior, xaxis = np.histogram(qty, bins=bins)

    post_weighted = post
    post_weighted[np.isnan(post_weighted)] = 0

    return post_weighted, xaxis

def calc_percentiles(qty, weights, bins, percentile_values, vb = False):

    qty_percentile_values = np.zeros((len(percentile_values),))

    post_weighted, xaxis = get_flat_posterior(qty, weights, bins)
    bw = np.nanmean(np.diff(xaxis))

    normed_cdf = np.cumsum(post_weighted)/np.amax(np.cumsum(post_weighted))

    for i in range(len(percentile_values)):
        qty_percentile_values[i] = xaxis[0:-1][np.argmin(np.abs(normed_cdf - percentile_values[i]/100))] + bw/2
        if (qty_percentile_values[i] == xaxis[0]+bw/2) or (qty_percentile_values[i] == xaxis[-1]+bw/2):
            qty_percentile_values[i] = np.nan

    if vb == True:

        qty_50 = xaxis[0:-1][np.argmin(np.abs(normed_cdf - 0.5))] + bw/2
        qty_16 = xaxis[0:-1][np.argmin(np.abs(normed_cdf - 0.16))] + bw/2
        qty_84 = xaxis[0:-1][np.argmin(np.abs(normed_cdf - 0.84))] + bw/2

        plt.plot(xaxis[0:-1]+bw/2, post_weighted)
        bf_value = qty[np.argmax(weights)]
        tempy = plt.ylim()
        plt.plot([bf_value, bf_value],[0,tempy[1]],'k-',label='best-fit')
        plt.plot([qty_50, qty_50],[0,tempy[1]],'-',label='50th percentile')
        plt.plot([qty_16, qty_16],[0,tempy[1]],'-',label='16th percentile')
        plt.plot([qty_84, qty_84],[0,tempy[1]],'-',label='84th percentile')
        print(bf_value, qty_50, np.argmax(weights), np.amax(weights))
        plt.legend(edgecolor='w',fontsize=14)
        plt.show()

    return qty_percentile_values


def evaluate_MAP(qty, weights, bins, smooth = 'kde', lowess_frac = 0.3, bw_method = 'scott', vb = False):

    post, xaxis = np.histogram(qty, weights=weights, bins=bins)
    xaxis_centers = xaxis[0:-1] + np.mean(np.diff(xaxis))

    if smooth == 'lowess':
        a = lowess(post, xaxis_centers,frac=lowess_frac)
        MAP = a[np.argmax(a[0:,1]),0]
    elif smooth == 'kde':
        a = gaussian_kde(qty,bw_method=bw_method, weights=weights)
        MAP = xaxis[np.argmax(a.evaluate(xaxis))]
    else:
        MAP = xaxis[np.argmax(post)+1]

    if vb == True:
        areapost = np.trapz(x=xaxis_centers, y=post)
        plt.plot(xaxis_centers, post/areapost)
        if smooth == 'lowess':
            plt.plot(a[0:,0],a[0:,1]/areapost)
        elif smooth == 'kde':
            plt.plot(xaxis, a.pdf(xaxis))
        plt.plot([MAP,MAP],plt.ylim())
        plt.show()

    return MAP

