# priors.py
# SED fitting priors and modeling assumptions, all in one place

import numpy as np
import matplotlib.pyplot as plt
import corner

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# this should provide:
# - tuple_to_sfh (existing DB)
# - calctimes, calctimes_to_tuple (existing DB)
# - and in modified version: continuity_to_sfh, make_continuity_agebins
from .gp_sfh import *


def quantile_names(N_params):
    """helper for labeling tx quantiles"""
    return (np.round(np.linspace(0, 100, N_params + 2)))[1:-1]


class Priors(object):
    """
    A class that holds prior info for SED fitting and gets passed into
    the atlas/pregrid builder.

    Supports two SFH families:
    - 'gp'          : original Dense Basis / tx-based SFH
    - 'continuity'  : continuity SFH with ratios between adjacent bins
    """

    def __init__(self):
        # ---------------------------------------------------------------
        # STELLAR MASS
        # ---------------------------------------------------------------
        self.mass_max = 12.0
        self.mass_min = 9.0

        # ---------------------------------------------------------------
        # SFR / sSFR PRIOR
        # options: 'SFRflat', 'sSFRflat', 'sSFRlognormal'
        # ---------------------------------------------------------------
        self.sfr_prior_type = 'sSFRflat'
        self.sfr_max = -1.0
        self.sfr_min = 2.0
        self.ssfr_min = -12.0
        self.ssfr_max = -7.5
        self.ssfr_mean = 0.6
        self.ssfr_sigma = 0.4
        self.ssfr_shift = -0.3

        # ---------------------------------------------------------------
        # REDSHIFT PRIOR
        # ---------------------------------------------------------------
        self.z_min = 0.9
        self.z_max = 1.1

        # ---------------------------------------------------------------
        # METALLICITY PRIOR
        # options: 'flat', 'massmet'
        # ---------------------------------------------------------------
        self.met_treatment = 'flat'
        self.Z_min = -1.5
        self.Z_max = 0.25
        self.massmet_width = 0.3

        # ---------------------------------------------------------------
        # DUST PRIOR
        # dust_model: 'Calzetti' or 'CF00'
        # dust_prior: 'flat' or 'exp'
        # ---------------------------------------------------------------
        self.dust_model = 'Calzetti'
        self.dust_prior = 'exp'
        self.Av_min = 0.0
        self.Av_max = 1.0
        self.Av_exp_scale = 1.0 / 3.0

        # ---------------------------------------------------------------
        # ORIGINAL DB SFH CONTROLS (tx-based)
        # ---------------------------------------------------------------
        # options: 'custom', 'TNGlike'
        self.sfh_treatment = 'custom'
        self.tx_alpha = 5.0
        self.Nparam = 3
        self.decouple_sfr = False
        self.decouple_sfr_time = 100  # Myr
        self.dynamic_decouple = True  # scale decouple time with redshift

        # ---------------------------------------------------------------
        # NEW: SFH FAMILY SELECTOR
        # 'gp' (default DB) or 'continuity'
        # ---------------------------------------------------------------
        self.sfh_type = 'gp'

        # ---------------------------------------------------------------
        # continuity-specific knobs
        # ---------------------------------------------------------------
        self.continuity_nbin = 7           # number of time bins
        self.continuity_df = 2.0           # Student-t DOF
        self.continuity_scale = 0.3        # scale of log-ratios

    # ====================================================================
    # BASIC PRIORS
    # ====================================================================

    def sample_mass_prior(self, size=1):
        massval = np.random.uniform(size=size) * (self.mass_max - self.mass_min) + self.mass_min
        # store for sSFR sampling
        self.massval = massval
        return massval

    def sample_sfr_prior(self, zval=1.0, size=1):
        """
        Returns log SFR *if* your SFH model needs an independent SFR draw.
        For continuity we don't actually need it, but we keep this for
        compatibility with the gp path.
        """
        if self.sfr_prior_type == 'SFRflat':
            return np.random.uniform(size=size) * (self.sfr_max - self.sfr_min) + self.sfr_min

        elif self.sfr_prior_type == 'sSFRflat':
            # draw log sSFR, then add logM
            return np.random.uniform(size=size) * (self.ssfr_max - self.ssfr_min) + self.ssfr_min + self.massval

        elif self.sfr_prior_type == 'sSFRlognormal':
            temp = np.random.lognormal(mean=self.ssfr_mean, sigma=self.ssfr_sigma, size=size)
            temp = temp - np.exp(self.ssfr_mean)  # center
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
            # very simple MZR-ish relation
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
            else:
                print('unknown dust_prior. options are flat and exp')

        elif self.dust_model == 'CF00':
            # not fully implemented here
            print('CF00 dust not fully implemented in this priors file.')
            return np.zeros(size=size) * np.nan

        else:
            print('dust model not implemented.')
            return np.zeros(size=size) * np.nan

    # ====================================================================
    # ORIGINAL DB / GP SFH SAMPLING
    # ====================================================================

    def sample_tx_prior(self, size=1):
        """
        Sample "tx" percentiles for the original DB SFH.
        If sfh_treatment == 'TNGlike', this tries to load lookup tables.
        """
        if self.sfh_treatment == 'TNGlike':
            # this uses the utility that DB ships with; keep as in your original file
            tng_zvals = np.load(get_file('train_data/alpha_lookup_tables', 'tng_alpha_zvals.npy'))
            tng_alphas = np.load(
                get_file('train_data/alpha_lookup_tables', f'tng_alpha_Nparam_{self.Nparam:.0f}.npy')
            )
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
        """
        Original DB tuple:
        [ logM*, logSFR_inst, Nparam, tx_1, tx_2, ... ]
        """
        sfh_tuple = np.zeros((self.Nparam + 3,))
        sfh_tuple[0] = self.sample_mass_prior()
        sfh_tuple[1] = self.sample_sfr_prior()
        sfh_tuple[2] = self.Nparam
        sfh_tuple[3:] = self.sample_tx_prior()
        return sfh_tuple

    # ====================================================================
    # CONTINUITY SFH SAMPLING (NEW)
    # ====================================================================

    def sample_continuity_ratios(self, nbin=None):
        """
        Draw log(SFR_i / SFR_{i-1}) from a Student-t distribution.
        This is the "continuity" part: adjacent bins prefer similar SFRs
        but can have heavy-tailed jumps.
        """
        if nbin is None:
            nbin = self.continuity_nbin

        # heavy-tailed around 0
        raw = np.random.standard_t(df=self.continuity_df, size=nbin - 1)
        log_sfr_ratios = raw * self.continuity_scale
        return log_sfr_ratios

    def sample_sfh_tuple_continuity(self):
        """
        Minimal, model-true continuity tuple:

            [ logM*, Nbin, log_sfr_ratio_0, ..., log_sfr_ratio_{Nbin-2} ]

        We do NOT store an independent instantaneous SFR here.
        """
        nbin = self.continuity_nbin

        # total mass formed
        logM = self.sample_mass_prior()

        # ratios
        log_sfr_ratios = self.sample_continuity_ratios(nbin=nbin)

        sfh_tuple = np.zeros((2 + (nbin - 1),))
        sfh_tuple[0] = logM
        sfh_tuple[1] = nbin
        sfh_tuple[2:] = log_sfr_ratios
        return sfh_tuple

    # ====================================================================
    # UNIFIED ENTRY POINTS (these are what atlas code should call)
    # ====================================================================

    def sample_sfh_tuple(self):
        if self.sfh_type == 'continuity':
            return self.sample_sfh_tuple_continuity()
        else:
            return self.sample_sfh_tuple_gp()

    def sample_all_params(self, random_seed=np.random.randint(1)):
        """
        Draw *one* full set of parameters from the priors:
        (sfh_tuple, metallicity, dust, z)
        """
        np.random.seed(random_seed)
        zval = self.sample_z_prior()
        sfh_tuple = self.sample_sfh_tuple()
        Z = self.sample_Z_prior()
        Av = self.sample_Av_prior()
        return sfh_tuple, Z, Av, zval

    def make_N_prior_draws(self, size=10, random_seed=np.random.randint(1)):
        """
        Draw many sets of parameters. Because continuity tuples and gp tuples
        have different lengths, we pad with NaN to a common length.
        """
        sfh_list = []
        Avs = np.zeros((size,))
        Zs = np.zeros((size,))
        zs = np.zeros((size,))

        for i in range(size):
            sfh_tuple, Zs[i], Avs[i], zs[i] = self.sample_all_params(
                random_seed=random_seed + i * 7
            )
            sfh_list.append(sfh_tuple)

        # pad to common length
        maxlen = max(len(x) for x in sfh_list)
        sfh_tuples = np.zeros((maxlen, size)) + np.nan
        for i, arr in enumerate(sfh_list):
            sfh_tuples[:len(arr), i] = arr

        return sfh_tuples, Zs, Avs, zs

    # ====================================================================
    # INFO / DIAGNOSTICS / PLOTTING
    # ====================================================================

    def print_priors(self):
        print('--------------Priors:--------------')
        print(f'log M* uniform: {self.mass_min:.1f} – {self.mass_max:.1f}')
        print(f'z uniform: {self.z_min:.1f} – {self.z_max:.1f}')
        print(f'log Z/Zsun uniform: {self.Z_min:.1f} – {self.Z_max:.1f}')
        print(f'dust: model={self.dust_model} prior={self.dust_prior} Av={self.Av_min:.1f}–{self.Av_max:.1f}')
        print('SFH family:', self.sfh_type)
        if self.sfh_type == 'continuity':
            print(f'  continuity_nbin   = {self.continuity_nbin}')
            print(f'  Student-t dof     = {self.continuity_df}')
            print(f'  log-ratio scale   = {self.continuity_scale}')
        else:
            print(f'  Nparam (gp)       = {self.Nparam}')
            print(f'  tx_alpha          = {self.tx_alpha}')
        print('-----------------------------------')

    def plot_prior_distributions(self, num_draws=20000):
        """
        Very rough corner plot of whatever parameters we can assemble.
        For continuity this will show the log-ratios.
        """
        sfh_tuples, Zs, Avs, zs = self.make_N_prior_draws(size=num_draws, random_seed=10)

        theta_list = []
        for i in range(num_draws):
            # drop NaNs from padded columns
            theta = [x for x in sfh_tuples[:, i] if not np.isnan(x)]
            theta += [Zs[i], Avs[i], zs[i]]
            theta_list.append(theta)

        theta_arr = np.array(theta_list, dtype=float)

        labels = []
        if self.sfh_type == 'continuity':
            labels.append('log M*')
            labels.append('Nbin')
            for k in range(self.continuity_nbin - 1):
                labels.append(f'ln r{k}')
        else:
            labels = ['log M*', 'log SFR', 'N_sfh']
            txs = ['t' + '%.0f' % i for i in quantile_names(self.Nparam)]
            labels[2:2] = txs

        labels += ['Z', 'Av', 'z']

        corner.corner(
            theta_arr,
            labels=labels,
            plot_datapoints=False,
            fill_contours=True,
            bins=50,
            smooth=1.0
        )
        plt.show()

    def plot_sfh_prior(self, numdraws=100, ref_mstar=10.0, zval=2.0):
        """
        Keep this mostly for the gp path. For continuity you'd usually
        just draw and plot directly using continuity_to_sfh.
        """
        if self.sfh_type == 'continuity':
            print("plot_sfh_prior: better to test continuity SFHs directly via gp_sfh.")
            return

        sfhs = np.zeros((1000, numdraws))
        sfh_tuples = np.zeros((self.Nparam + 3, numdraws))

        for i in range(numdraws):
            sfh_tuples[:, i], _, _, _ = self.sample_all_params(random_seed=i * 7)
            ssfr = sfh_tuples[1, i] - sfh_tuples[0, i]
            sfh_tuples[0, i] = ref_mstar
            sfh_tuples[1, i] = ssfr + ref_mstar
            sfhs[:, i], time = tuple_to_sfh(sfh_tuples[:, i], zval=zval)

        plt.figure(figsize=(12, 6))
        plt.plot((np.amax(time) - time), np.nanmedian(sfhs, 1), 'k', lw=3)
        plt.fill_between(
            (np.amax(time) - time),
            np.nanpercentile(sfhs, 40, axis=1),
            np.nanpercentile(sfhs, 60, axis=1),
            color='k',
            alpha=0.1
        )
        plt.fill_between(
            (np.amax(time) - time),
            np.nanpercentile(sfhs, 25, axis=1),
            np.nanpercentile(sfhs, 75, axis=1),
            color='k',
            alpha=0.1
        )
        plt.fill_between(
            (np.amax(time) - time),
            np.nanpercentile(sfhs, 16, axis=1),
            np.nanpercentile(sfhs, 84, axis=1),
            color='k',
            alpha=0.1
        )
        plt.ylabel('normalized SFR')
        plt.xlabel('lookback time [Gyr]')
        plt.show()

