import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

import george
from george import kernels

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import DotProduct, RationalQuadratic
except ImportError:
    print("sklearn not available. The `gp_sklearn` interpolator will fail if chosen.")

from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator, interp1d
import scipy.io as sio

from astropy.cosmology import FlatLambdaCDM #bug fix
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

import pkg_resources

def get_file(folder, filename):
    resource_package = __name__
    resource_path = '/'.join((folder, filename))  # Do not use os.path.join()
    template = pkg_resources.resource_stream(resource_package, resource_path)
    return template

fsps_mlc = sio.loadmat(get_file('train_data','fsps_mass_loss_curve.mat'))
fsps_time = fsps_mlc['timeax_fsps'].ravel()
fsps_massloss = fsps_mlc['mass_loss_fsps'].ravel()

# basic SFH tuples
rising_sfh = np.array([10.0,1.0,3,0.5,0.7,0.9])
regular_sfg_sfh = np.array([10.0,0.3,3,0.25,0.5,0.75])
young_quenched_sfh = np.array([10.0,-1.0,3,0.3,0.6,0.8])
old_quenched_sfh = np.array([10.0,-1.0,3,0.1,0.2,0.4])
old_very_quenched_sfh = np.array([10.0,-10.0,3,0.1,0.2,0.4])
double_peaked_SF_sfh = np.array([10.0,0.5,3,0.25,0.4,0.7])
double_peaked_Q_sfh = np.array([10.0,-1.0,3,0.2,0.4,0.8])


def correct_for_mass_loss(sfh, time, mass_loss_curve_time, mass_loss_curve):
    correction_factors = np.interp(time, mass_loss_curve_time, mass_loss_curve)
    return sfh * correction_factors

# functions:
def nll(p, gp, y):
    gp.set_parameter_vector(p)
    ll = gp.log_likelihood(y, quiet=True)
    return -ll if np.isfinite(ll) else 1e25

def grad_nll(p, gp, y):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y, quiet=True)

#--------------------------------------------


def gp_interpolator(x,y,res = 1000, Nparam = 3, decouple_sfr = False):

    yerr = np.zeros_like(y)
    yerr[2:(2+Nparam)] = 0.001/np.sqrt(Nparam)
    if len(yerr) > 26:
        yerr[2:(2+Nparam)] = 0.1/np.sqrt(Nparam)
    if decouple_sfr == True:
        yerr[(2+Nparam):] = 0.1

    kernel = np.var(y) * (kernels.Matern32Kernel(np.median(y)) + kernels.LinearKernel(np.median(y), order=2))
    gp = george.GP(kernel, solver=george.HODLRSolver)

    gp.compute(x.ravel(), yerr.ravel())

    x_pred = np.linspace(np.amin(x), np.amax(x), res)
    y_pred, pred_var = gp.predict(y.ravel(), x_pred, return_var=True)

    return x_pred, y_pred

def gp_sklearn_interpolator(x,y,res = 1000):

    kernel = DotProduct(10.0, (1e-2,1e2)) * RationalQuadratic(0.1)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp.fit(x.reshape(-1,1),(y-x).reshape(-1,1))

    x_pred = np.linspace(0,1,1000)
    y_pred, sigma = gp.predict(x_pred[:,np.newaxis], return_std=True)
    y_pred = y_pred.ravel() + x_pred

    return x_pred, y_pred

def linear_interpolator(x,y,res = 1000):

    interpolator = interp1d(x,y)
    x_pred = np.linspace(np.amin(x), np.amax(x), res)
    y_pred = interpolator(x_pred)

    return x_pred, y_pred

def Pchip_interpolator(x,y,res = 1000):

    interpolator = PchipInterpolator(x,y)
    x_pred = np.linspace(np.amin(x), np.amax(x), res)
    y_pred = interpolator(x_pred)

    return x_pred, y_pred


def tuple_to_sfh(sfh_tuple, zval, interpolator = 'gp_george', decouple_sfr = False, decouple_sfr_time = 10, sfr_tolerance = 0.05, vb = False,cosmo = cosmo, res = 1000, print_warnings = False):
    # generate an SFH from an input tuple (Mass, SFR, {tx}) at a specified redshift

    Nparam = int(sfh_tuple[2])
    mass_quantiles = np.linspace(0,1,Nparam+2)
    time_quantiles = np.zeros_like(mass_quantiles)
    time_quantiles[-1] = 1
    time_quantiles[1:-1] = sfh_tuple[3:]

    # now add SFR constraints

    # SFR smoothly increasing from 0 at the big bang
    mass_quantiles = np.insert(mass_quantiles,1,[0.00])
    time_quantiles = np.insert(time_quantiles,1,[0.01])

    # SFR constrained to SFR_inst at the time of observation
    SFH_constraint_percentiles = np.array([0.97, 0.98, 0.99])
    for const_vals in SFH_constraint_percentiles:

        delta_mstar = 10**(sfh_tuple[0]) *(1-const_vals)
        delta_t = 1 - delta_mstar/(10**sfh_tuple[1])/(cosmo.age(zval).value*1e9)

        if (delta_t > time_quantiles[-2]) & (delta_t > 0.9):
            mass_quantiles = np.insert(mass_quantiles, -1, [const_vals], )
            time_quantiles = np.insert(time_quantiles, -1, [delta_t],)
        else:
            delta_m = 1 - ((cosmo.age(zval).value*1e9)*(1-const_vals)*(10**sfh_tuple[1]))/(10**sfh_tuple[0])
            time_quantiles = np.insert(time_quantiles, -1, [const_vals])
            mass_quantiles=  np.insert(mass_quantiles, -1, [delta_m])

    if interpolator == 'gp_george':
        time_arr_interp, mass_arr_interp = gp_interpolator(time_quantiles, mass_quantiles, Nparam = int(Nparam), decouple_sfr = decouple_sfr, res=res)
    elif interpolator == 'gp_sklearn':
        time_arr_interp, mass_arr_interp = gp_sklearn_interpolator(time_quantiles, mass_quantiles)
    elif interpolator == 'linear':
        time_arr_interp, mass_arr_interp = linear_interpolator(time_quantiles, mass_quantiles)
    elif interpolator == 'pchip':
        time_arr_interp, mass_arr_interp = Pchip_interpolator(time_quantiles, mass_quantiles)
    else:
        raise Exception('specified interpolator does not exist: {}. \n use one of the following: gp_george, gp_sklearn, linear, and pchip '.format(interpolator))

    sfh_scale = 10**(sfh_tuple[0])/(cosmo.age(zval).value*1e9/1000)
    sfh = np.diff(mass_arr_interp)*sfh_scale
    sfh[sfh<0] = 0
    sfh = np.insert(sfh,0,[0])

    sfr_decouple_time_index = np.argmin(np.abs(time_arr_interp*cosmo.age(zval).value - decouple_sfr_time/1e3))
    if sfr_decouple_time_index == 0:
        sfr_decouple_time_index = 2
    mass_lastbins = np.trapz(x=time_arr_interp[-sfr_decouple_time_index:]*1e9*(cosmo.age(zval).value), y=sfh[-sfr_decouple_time_index:])
    mass_remaining = 10**(sfh_tuple[0]) - mass_lastbins
    if mass_remaining < 0:
        mass_remaining = 0
        if print_warnings == True:
            print('input SFR, M* combination is not physically consistent (log M*: %.2f, log SFR: %.2f.)' %(sfh_tuple[0],sfh_tuple[1]))
    mass_initbins = np.trapz(x=time_arr_interp[0:(1000-sfr_decouple_time_index)]*1e9*(cosmo.age(zval).value), y=sfh[0:(1000-sfr_decouple_time_index)])
    sfh[0:(1000-sfr_decouple_time_index)] = sfh[0:(1000-sfr_decouple_time_index)] * mass_remaining / mass_initbins

    if (np.abs(np.log10(sfh[-1]) - sfh_tuple[1]) > sfr_tolerance) or (decouple_sfr == True):
        sfh[-sfr_decouple_time_index:] = 10**sfh_tuple[1]

    timeax = time_arr_interp * cosmo.age(zval).value

    if vb == True:
        print('time and mass quantiles:')
        print(time_quantiles, mass_quantiles)
        import matplotlib.pyplot as plt
        plt.plot(time_quantiles, mass_quantiles,'--o')
        plt.plot(time_arr_interp, mass_arr_interp)
        plt.axis([0,1,0,1])
        plt.show()

        print('instantaneous SFR: %.1f' %sfh[-1])
        plt.plot(np.amax(time_arr_interp) - time_arr_interp, sfh)
        plt.show()

    return sfh, timeax

def calctimes(timeax,sfh,nparams):

    massint = np.cumsum(sfh)
    massint_normed = massint/np.amax(massint)
    tx = np.zeros((nparams,))
    for i in range(nparams):
        tx[i] = timeax[np.argmin(np.abs(massint_normed - 1*(i+1)/(nparams+1)))]

    mass = np.log10(np.trapz(sfh,timeax*1e9))
    sfr = np.log10(sfh[-1])

    return mass, sfr, tx/np.amax(timeax)

def calctimes_to_tuple(calctimelist):
    nparam = len(calctimelist[2])

    sfhtuple = []
    sfhtuple.append(calctimelist[0])
    sfhtuple.append(calctimelist[1])
    sfhtuple.append(nparam)
    for i in range(nparam):
        sfhtuple.append(calctimelist[2][i])
    return sfhtuple

def scale_t50(t50_val = 1.0, zval = 1.0):
    """
    Change a t50 value from lookback time in Gyr at a given redshift
    to fraction of the age of the universe.
    """
    return (1 - t50_val/cosmo.age(zval).value)

def scale_t50_inv(t50_val_frac = 1.0, zval = 1.0):
    """
    Change a t50 value from a fraction of the age of the universe to
    lookback time in Gyr at a given redshift
    """
    return (1- t50_val_frac)*cosmo.age(zval).value


# ---------------------------------------------------------------------
# NEW: continuity SFH helpers
# ---------------------------------------------------------------------
import numpy as np
from astropy.cosmology import FlatLambdaCDM

# if you already have cosmo defined elsewhere, reuse that
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def make_continuity_agebins(zval, nbin=None, bin_edges_fraction=None, cosmo=cosmo):
    """
    Build *cosmic-time* bins between 0 and age(z).

    Parameters
    ----------
    zval : float
        redshift
    nbin : int, optional
        number of bins (used only if bin_edges_fraction is None)
    bin_edges_fraction : array-like, optional
        fractional edges in [0,1], e.g. [0, 0.01, 0.05, 0.1, 0.9, 1.0]
        will be multiplied by age(z)

    Returns
    -------
    agebins : (nbin, 2) array
        [[t0_start, t0_end],
         [t1_start, t1_end],
         ...
        ] in Gyr, cosmic time, increasing
    """
    t_univ = cosmo.age(zval).value  # Gyr

    if bin_edges_fraction is not None:
        edges = np.asarray(bin_edges_fraction, float) * t_univ
        if not np.all(np.diff(edges) >= 0):
            raise ValueError("bin_edges_fraction must be non-decreasing")
    else:
        if nbin is None:
            raise ValueError("either nbin or bin_edges_fraction must be provided")
        edges = np.linspace(0.0, t_univ, nbin + 1)

    # make (nbin, 2)
    return np.vstack([edges[:-1], edges[1:]]).T


def continuity_to_sfh(logmass,
                      log_sfr_ratios,
                      zval,
                      agebins=None,
                      bin_edges_fraction=None,
                      cosmo=cosmo,
                      n_time_fine=300):
    """
    Build a piecewise-constant SFH in *cosmic time* for FSPS.

    Parameters
    ----------
    logmass : float
        log10 of total formed mass to reproduce
    log_sfr_ratios : array, shape (nbin-1,)
        log10(SFR_i / SFR_{i-1}) for i = 1..nbin-1
    zval : float
        redshift; used to get age(z)
    agebins : (nbin, 2) array, optional
        cosmic-time bins in Gyr, increasing (0 -> age(z))
    bin_edges_fraction : array-like, optional
        alternative to agebins: fractional edges in [0,1] to be stretched to age(z)
    cosmo : astropy cosmology
    n_time_fine : int
        number of points in the returned fine time axis

    Returns
    -------
    sfh_fine : (n_time_fine,) array
        SFR(t) in Msun/yr, cosmic time, increasing
    time_fine : (n_time_fine,) array
        cosmic time in Gyr, 0 -> age(z)
    """
    # 1. build bins in *cosmic* time
    if agebins is None:
        # will raise if neither nbin nor fraction is given
        # we infer nbin from len(log_sfr_ratios)+1
        nbin = len(log_sfr_ratios) + 1
        agebins = make_continuity_agebins(zval,
                                          nbin=nbin,
                                          bin_edges_fraction=bin_edges_fraction,
                                          cosmo=cosmo)
    else:
        agebins = np.asarray(agebins, float)
        nbin = agebins.shape[0]

    # 2. sanity: ratios should be nbin-1
    if len(log_sfr_ratios) != nbin - 1:
        raise ValueError(f"len(log_sfr_ratios)={len(log_sfr_ratios)} but bins={nbin} (need nbin-1)")

    # 3. build SFR per bin from ratios (log10)
    sfr_bins = np.zeros(nbin)
    sfr_bins[0] = 1.0  # arbitrary starter
    for i in range(1, nbin):
        sfr_bins[i] = sfr_bins[i - 1] * 10 ** (log_sfr_ratios[i - 1])

    # 4. convert bin widths -> mass per bin
    widths_yr = (agebins[:, 1] - agebins[:, 0]) * 1e9  # Gyr -> yr
    m_bin = sfr_bins * widths_yr

    # 5. rescale to total formed mass
    target_mass = 10 ** logmass
    m_bin *= target_mass / m_bin.sum()
    sfr_bins = m_bin / widths_yr  # back to Msun/yr

    # 6. make a fine *cosmic-time* axis and fill
    t0 = agebins[0, 0]
    t1 = agebins[-1, 1]
    time_fine = np.linspace(t0, t1, n_time_fine)
    sfh_fine = np.zeros_like(time_fine)
    for i, (ts, te) in enumerate(agebins):
        mask = (time_fine >= ts) & (time_fine < te)
        sfh_fine[mask] = sfr_bins[i]
    # include the last edge
    sfh_fine[time_fine >= agebins[-1, 1]] = sfr_bins[-1]

    return sfh_fine, time_fine




