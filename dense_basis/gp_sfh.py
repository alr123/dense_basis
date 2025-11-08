# priors.py
# SED fitting priors and modeling assumptions, all in one place

import numpy as np
import matplotlib.pyplot as plt
import corner

# cosmology assumption
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# this is the DB helper file (we’ll add continuity hooks there)
from .gp_sfh import (
    tuple_to_sfh,
    make_continuity_agebins,
    continuity_to_sfh,
    calctimes,
    calctimes_to_tuple,
)

def quantile_names(N_params):
    return (np.round(np.linspace(0, 100, N_params + 2)))[1:-1]


class Priors(object):
    """ A class that holds prior information for SED fitting.

    NOTE: now supports TWO SFH families:
        - sfh_type = 'gp'          (original DB behavior)
        - sfh_type = 'continuity'  (ratios between time bins)

    For continuity, the user can provide:
        - a fixed (N,2) array of bins: priors.set_continuity_agebins(...)
        - OR a callable: priors.set_continuity_agebin_fn(lambda z: bins)
    """

    def __init__(self):
        # ----- mass -----
        self.mass_max = 12.0
        self.mass_min = 9.0

        # ----- SFR / sSFR priors -----
        self.sfr_prior_type = 'sSFRflat'  # SFRflat, sSFRflat, sSFRlognormal
        self.sfr_max = -1.0
        self.sfr_min = 2.0
        self.ssfr_min = -12.0
        self.ssfr_max = -7.5
        self.ssfr_mean = 0.6
        self.ssfr_sigma = 0.4
        self.ssfr_shift = -0.3

        # ----- redshift -----
        self.z_min = 0.9
        self.z_max = 1.1

        # ----- metallicity -----
        self.met_treatment = 'flat'  # or 'massmet'
        self.Z_min = -1.5
        self.Z_max = 0.25
        self.massmet_width = 0.3

        # ----- dust -----
        self.dust_model = 'Calzetti'
        self.dust_prior = 'exp'
        self.Av_min = 0.0
        self.Av_max = 1.0
        self.Av_exp_scale = 1.0 / 3.0

        # ----- original DB SFH controls -----
        self.sfh_treatment = 'custom'  # or 'TNGlike'
        self.tx_alpha = 5.0
        self.Nparam = 3
        self.decouple_sfr = False
        self.decouple_sfr_time = 100  # in Myr
        self.dynamic_decouple = True  # set decouple time according to z

        # ----- NEW: choose SFH family -----
        self.sfh_type = 'gp'  # default = original DB

        # ----- continuity-specific -----
        self.continuity_nbin = 7            # default number of time bins
        self.continuity_df = 2.0            # student-t df for log ratios
        self.continuity_scale = 0.3         # scale for log ratios
        self.custom_agebins = None          # fixed (N,2) in Gyr
        self.custom_agebin_fn = None        # callable: z -> (N,2)

    # ------------------------------------------------------------------
    # basic priors
    # ------------------------------------------------------------------
    def sample_mass_prior(self, size=1):
        massval = np.random.uniform(size=size) * (self.mass_max - self.mass_min) + self.mass_min
        self.massval = massval
        return massval

    def sample_sfr_prior(self, zval=1.0, size=1):
        if self.sfr_prior_type == 'SFRflat':
            return np.random.uniform(size=size) * (self.sfr_max - self.sfr_min) + self.sfr_min

        elif self.sfr_prior_type == 'sSFRflat':
            return np.random.uniform(size=size) * (self.ssfr_max - self.ssfr_min) + self.ssfr_min + self.massval

        elif self.sfr_prior_type == 'sSFRlognormal':
            temp = np.random.lognormal(mean=self.ssfr_mean, sigma=self.ssfr_sigma, size=size)
            temp = temp - np.exp(self.ssfr_mean)
            temp = np.log10(10.0 / (cosmo.age(zval).value * 1e9)) - temp + self.ssfr_shift
            sfrval = temp + self.massval
            return sfrval

        else:
            print('unknown SFR prior type. choose from SFRflat, sSFRflat, or sSFRlognormal.')
            return np.nan

    def sample_z_prior(self, size=1):
        zval = np.random.uniform(size=size) * (self.z_max - self.z_min) + self.z_min
        self.zval = zval
        return zval

    def sample_Z_prior(self, size=1):
        if self.met_treatment == 'flat':
            return np.random.uniform(size=size) * (self.Z_max - self.Z_min) + self.Z_min
        elif self.met_treatment == 'massmet':
            met = (
                np.random.normal(scale=self.massmet_width, size=size)
                + (self.massval - 7) / (10.8 - 7)
                - 1.0
            )
            return met

    def sample_Av_prior(self, size=1):
        if self.dust_model == 'Calzetti':
            if self.dust_prior == 'flat':
                return np.random.uniform(size=size) * (self.Av_max - self.Av_min) + self.Av_min
            elif self.dust_prior == 'exp':
                return np.random.exponential(size=size) * (self.Av_exp_scale)
        # other dust models unchanged from original code
        print('not currently coded up, please email me regarding this functionality.')
        return np.zeros(size=size) * np.nan

    # ------------------------------------------------------------------
    # original DB / tx SFH
    # ------------------------------------------------------------------
    def sample_tx_prior(self, size=1):
        if self.sfh_treatment == 'TNGlike':
            # these are as in the original repo
            tng_zvals = np.load(get_file('train_data/alpha_lookup_tables', 'tng_alpha_zvals.npy'))
            tng_alphas = np.load(get_file('train_data/alpha_lookup_tables', 'tng_alpha_Nparam_%.0f.npy' % self.Nparam))
            tng_best_z_index = np.argmin(np.abs(tng_zvals - self.zval))
            self.tx_alpha = tng_alphas[0:, tng_best_z_index]

        if size == 1:
            temp_tx = np.cumsum(
                np.random.dirichlet(np.ones((self.Nparam + 1,)) * self.tx_alpha, size=size)
            )[0:-1]
            return temp_tx
        else:
            temp_tx = np.cumsum(
                np.random.dirichlet(np.ones((self.Nparam + 1,)) * self.tx_alpha, size=size),
                axis=1
            )[0:, 0:-1]
            return temp_tx

    def sample_sfh_tuple_gp(self):
        sfh_tuple = np.zeros((self.Nparam + 3,))
        sfh_tuple[0] = self.sample_mass_prior()
        sfh_tuple[1] = self.sample_sfr_prior()
        sfh_tuple[2] = self.Nparam
        sfh_tuple[3:] = self.sample_tx_prior()
        return sfh_tuple

    # ------------------------------------------------------------------
    # continuity SFH
    # ------------------------------------------------------------------
    def set_continuity_agebins(self, agebins):
        """
        Set fixed time bins for continuity SFH.
        agebins: (N, 2) array, lookback time in Gyr
        """
        agebins = np.asarray(agebins, float)
        assert agebins.ndim == 2 and agebins.shape[1] == 2
        self.custom_agebins = agebins
        self.custom_agebin_fn = None
        self.continuity_nbin = agebins.shape[0]

    def set_continuity_agebin_fn(self, fn):
        """
        Set a redshift-dependent binning:
        fn(z) -> (N, 2) array in Gyr
        """
        self.custom_agebin_fn = fn
        self.custom_agebins = None

    def sample_continuity_ratios(self, nbin=None):
        if nbin is None:
            nbin = self.continuity_nbin
        raw = np.random.standard_t(df=self.continuity_df, size=nbin - 1)
        return raw * self.continuity_scale

    def sample_sfh_tuple_continuity(self):
        # we may not know actual nbin if user gave a callable, so fall back
        if self.custom_agebins is not None:
            nbin = self.custom_agebins.shape[0]
        else:
            nbin = self.continuity_nbin

        logM = self.sample_mass_prior()
        log_sfr_ratios = self.sample_continuity_ratios(nbin=nbin)

        sfh_tuple = np.zeros((2 + (nbin - 1),))
        sfh_tuple[0] = logM
        sfh_tuple[1] = nbin
        sfh_tuple[2:] = log_sfr_ratios
        return sfh_tuple

    # ------------------------------------------------------------------
    # unified entry point
    # ------------------------------------------------------------------
    def sample_sfh_tuple(self):
        if self.sfh_type == 'continuity':
            return self.sample_sfh_tuple_continuity()
        else:
            return self.sample_sfh_tuple_gp()

    def sample_all_params(self, random_seed=np.random.randint(1)):
        np.random.seed(random_seed)
        temp_z = self.sample_z_prior()
        sfh_tuple = self.sample_sfh_tuple()
        temp_Z = self.sample_Z_prior()
        temp_Av = self.sample_Av_prior()
        return sfh_tuple, temp_Z, temp_Av, temp_z

    # rest of plotting helpers from your original file can stay unchanged

