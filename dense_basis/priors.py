# SED fitting priors and modeling assumptions, all in one place

import numpy as np
import matplotlib.pyplot as plt
import corner

# cosmology assumption
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# this should now contain:
# - tuple_to_sfh (existing DB)
# - continuity_to_sfh (you added)
# - make_continuity_agebins (you added)
from .gp_sfh import *

def quantile_names(N_params):
    return (np.round(np.linspace(0, 100, N_params + 2)))[1:-1]


class Priors(object):
    """
    A class that holds prior information for the various parameters
    during SED fitting - their distributions and bounds. Gets passed into
    generate_atlas methods.
    """

    def __init__(self):
        # ----- stellar mass -----
        self.mass_max = 12.0
        self.mass_min = 9.0

        # ----- SFR / sSFR -----
        # options: 'SFRflat', 'sSFRflat', 'sSFRlognormal'
        self.sfr_prior_type = 'sSFRflat'
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
        # options: 'flat', 'massmet'
        self.met_treatment = 'flat'
        self.Z_min = -1.5
        self.Z_max = 0.25
        self.massmet_width = 0.3

        # ----- dust -----
        # dust_model: 'Calzetti' or 'CF00'
        self.dust_model = 'Calzetti'
        # dust_prior: 'flat' or 'exp'
        self.dust_prior = 'exp'
        self.Av_min = 0.0
        self.Av_max = 1.0
        self.Av_exp_scale = 1.0 / 3.0

        # ----- GP / DB original SFH controls -----
        # options: 'custom', 'TNGlike'
        self.sfh_treatment = 'custom'
        self.tx_alpha = 5.0
        self.Nparam = 3
        self.decouple_sfr = False
        self.decouple_sfr_time = 100  # Myr
        self.dynamic_decouple = True

        # ----- NEW: SFH family selector -----
        # 'gp'       -> original DB behavior (tx-based / GP-like)
        # 'continuity' -> continuity SFH with SFR ratios
        self.sfh_type = 'continuity'

        # ----- continuity-specific knobs -----
        # number of time bins (like Prospector continuity_sfh)
        self.continuity_nbin = 7
        # Student-t degrees of freedom (lower -> heavier tails)
        self.continuity_df = 2.0
        # scale of log-ratios (like sigma)
        self.continuity_scale = 0.3

    # -------------------------------------------------------------------------
    # basic priors
    # -------------------------------------------------------------------------
    def sample_mass_prior(self, size=1):
        massval = np.random.uniform(size=size) * (self.mass_max - self.mass_min) + self.mass_min
        # store for sSFR sampling
        self.massval = massval
        return massval

    def sample_sfr_prior(self, zval=1.0, size=1):
        if self.sfr_prior_type == 'SFRflat':
            return np.random.uniform(size=size) * (self.sfr_max - self.sfr_min) + self.sfr_min

        elif self.sfr_prior_type == 'sSFRflat':
            # draw log sSFR, then add logM
            return np.random.uniform(size=size) * (self.ssfr_max - self.ssfr_min) + self.ssfr_min + self.massval

        elif self.sfr_prior_type == 'sSFRlognormal':
            # draw from lognormal, shift to sSFR(z), then add mass
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
            met = np.random.normal(scale=self.massmet_width, size=size) + (self.massval - 7) / (10.8 - 7) - 1.0
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
            # place-holder for CF00 path
            print('CF00 dust not fully implemented here.')
            return np.zeros(size=size) * np.nan

        else:
            print('dust model not implemented.')
            return np.zeros(size=size) * np.nan

    # -------------------------------------------------------------------------
    # original DB tx-based SFH sampler
    # -------------------------------------------------------------------------
    def sample_tx_prior(self, size=1):
        if self.sfh_treatment == 'TNGlike':
            # load TNG alpha tables
            tng_zvals = np.load(get_file('train_data/alpha_lookup_tables', 'tng_alpha_zvals.npy'))
            tng_alphas = np.load(get_file('train_data/alpha_lookup_tables',
                                          'tng_alpha_Nparam_%.0f.npy' % self.Nparam))
            tng_best_z_index = np.argmin(np.abs(tng_zvals - self.zval))
            self.tx_alpha = tng_alphas[0:, tng_best_z_index]

        if size == 1:
            temp_tx = np.cumsum(np.random.dirichlet(np.ones((self.Nparam + 1,)) * self.tx_alpha, size=size))[0:-1]
            return temp_tx
        else:
            temp_tx = np.cumsum(np.random.dirichlet(np.ones((self.Nparam + 1,)) * self.tx_alpha, size=size), axis=1)[
                      0:, 0:-1]
            return temp_tx

    def sample_sfh_tuple_gp(self):
        """
        Original DB behavior:
        [ logM*, logSFR_inst, Nparam, tx_1, tx_2, ... ]
        """
        sfh_tuple = np.zeros((self.Nparam + 3,))
        sfh_tuple[0] = self.sample_mass_prior()
        sfh_tuple[1] = self.sample_sfr_prior()
        sfh_tuple[2] = self.Nparam
        sfh_tuple[3:] = self.sample_tx_prior()
        return sfh_tuple

    # -------------------------------------------------------------------------
    # NEW: continuity SFH sampler with Student-t log-ratio prior
    # -------------------------------------------------------------------------
    def sample_continuity_ratios(self, nbin=None):
        """
        Draw log(SFR_i / SFR_{i-1}) from a Student-t distribution.
        """
        if nbin is None:
            nbin = self.continuity_nbin

        # standard_t(df, size) -> heavy-tailed around 0
        raw = np.random.standard_t(df=self.continuity_df, size=nbin - 1)
        log_sfr_ratios = raw * self.continuity_scale
        return log_sfr_ratios

    def sample_sfh_tuple_continuity(self):
        """
        Return a DB-shaped tuple:
        [ logM*, logSFR_inst, Nbin, log_sfr_ratio_0, ..., log_sfr_ratio_{Nbin-2} ]
        """
        nbin = self.continuity_nbin

        logM = self.sample_mass_prior()
        logSFR_inst = self.sample_sfr_prior()
        log_sfr_ratios = self.sample_continuity_ratios(nbin=nbin)

        sfh_tuple = np.zeros((3 + (nbin - 1),))
        sfh_tuple[0] = logM
        sfh_tuple[1] = logSFR_inst
        sfh_tuple[2] = nbin
        sfh_tuple[3:] = log_sfr_ratios
        return sfh_tuple

    # -------------------------------------------------------------------------
    # unified SFH sampler (DB will call this)
    # -------------------------------------------------------------------------
    def sample_sfh_tuple(self):
        if self.sfh_type == 'continuity':
            return self.sample_sfh_tuple_continuity()
        else:
            return self.sample_sfh_tuple_gp()

    # -------------------------------------------------------------------------
    # wrappers that DB expects
    # -------------------------------------------------------------------------
    def sample_all_params(self, random_seed=np.random.randint(1)):
        np.random.seed(random_seed)
        temp_z = self.sample_z_prior()
        sfh_tuple = self.sample_sfh_tuple()
        temp_Z = self.sample_Z_prior()
        temp_Av = self.sample_Av_prior()
        return sfh_tuple, temp_Z, temp_Av, temp_z

    def make_N_prior_draws(self, size=10, random_seed=np.random.randint(1)):
        """
        Returns:
            sfh_tuples: 2D array (padded with NaN if continuity tuples are longer)
            Zs, Avs, zs: 1D arrays
        """
        sfh_list = []
        Avs = np.zeros((size,))
        Zs = np.zeros((size,))
        zs = np.zeros((size,))

        for i in range(size):
            sfh_tuple, Zs[i], Avs[i], zs[i] = self.sample_all_params(random_seed=random_seed + i * 7)
            sfh_list.append(sfh_tuple)

        maxlen = max(len(x) for x in sfh_list)
        sfh_tuples = np.zeros((maxlen, size)) + np.nan
        for i, arr in enumerate(sfh_list):
            sfh_tuples[:len(arr), i] = arr

        return sfh_tuples, Zs, Avs, zs

    # -------------------------------------------------------------------------
    # plotting helpers (kept mostly as-is)
    # -------------------------------------------------------------------------
    def print_priors(self):
        print('--------------Priors:--------------')
        print('log M* uniform from %.1f to %.1f' % (self.mass_min, self.mass_max))
        print('z uniform from %.1f to %.1f' % (self.z_min, self.z_max))
        print('log Z/Zsun uniform from %.1f to %.1f' % (self.Z_min, self.Z_max))
        print('dust model:', self.dust_model, 'prior:', self.dust_prior,
              'Av: %.1f–%.1f' % (self.Av_min, self.Av_max))
        print('SFH type:', self.sfh_type)
        if self.sfh_type == 'continuity':
            print('  continuity_nbin =', self.continuity_nbin)
            print('  Student-t df     =', self.continuity_df)
            print('  log-ratio scale  =', self.continuity_scale)
        else:
            print('  Nparam (gp)      =', self.Nparam)
            print('  tx_alpha         =', self.tx_alpha)
        print('-----------------------------------')

    def plot_prior_distributions(self, num_draws=100000):
        sfh_tuples, Zs, Avs, zs = self.make_N_prior_draws(size=num_draws, random_seed=10)

        # flatten to something corner can handle; we’ll just drop NaNs
        theta_list = []
        for i in range(num_draws):
            theta = [x for x in sfh_tuples[:, i] if not np.isnan(x)]
            theta += [Zs[i], Avs[i], zs[i]]
            theta_list.append(theta)
        theta_arr = np.array(theta_list, dtype=float)

        # labels will depend on SFH type
        labels = ['log M*', 'log SFR', 'N_sfh']
        if self.sfh_type == 'continuity':
            for k in range(self.continuity_nbin - 1):
                labels.append(f'ln r{k}')
        else:
            txs = ['t' + '%.0f' % i for i in quantile_names(self.Nparam)]
            labels[2:2] = txs

        labels += ['Z', 'Av', 'z']

        figure = corner.corner(theta_arr,
                               labels=labels,
                               plot_datapoints=False,
                               fill_contours=True,
                               bins=50,
                               smooth=1.0)
        plt.show()


    def plot_sfh_prior(self, numdraws = 100, ref_mstar = 10.0, zval = 5.606):

        sfhs = np.zeros((1000, numdraws))
        sfh_tuples = np.zeros((self.Nparam+3, numdraws))
        sfr_error = np.zeros((numdraws,))
        mass_error = np.zeros((numdraws,))

        for i in (range(numdraws)):
            sfh_tuples[0:,i], _,_,_ = self.sample_all_params(random_seed = i*7)
            ssfr = sfh_tuples[1,i] - sfh_tuples[0,i]
            sfh_tuples[0,i] = ref_mstar
            sfh_tuples[1,i] = ssfr + ref_mstar
            sfhs[0:,i], time = tuple_to_sfh(sfh_tuples[0:,i], zval = zval)
            sfr_error[i] = np.log10(sfhs[-1,i]) - sfh_tuples[1,i]
            mass_error[i] = np.log10(np.trapz(x=time*1e9, y=sfhs[0:,i])) - sfh_tuples[0,i]

        # sfr_error[sfr_error<-2] = -2
        # sfr_error[sfr_error>2] = 2
        # plt.plot(sfh_tuples[1,0:], sfr_error,'.')
        # plt.show()
        # plt.hist(sfr_error,30)
        # plt.show()
        #
        # plt.plot(sfh_tuples[0,0:], mass_error,'.')
        # plt.show()
        # plt.hist(mass_error,30)
        # plt.show()

        plt.figure(figsize=(12,6))
        plt.plot((np.amax(time)-time), np.nanmedian(sfhs,1),'k',lw=3)
        plt.fill_between((np.amax(time)-time), np.nanpercentile(sfhs,40,axis=1),
                         np.nanpercentile(sfhs,60,axis=1),color='k',alpha=0.1)
        plt.fill_between((np.amax(time)-time), np.nanpercentile(sfhs,25,axis=1),
                         np.nanpercentile(sfhs,75,axis=1),color='k',alpha=0.1)
        plt.fill_between((np.amax(time)-time), np.nanpercentile(sfhs,16,axis=1),
                         np.nanpercentile(sfhs,84,axis=1),color='k',alpha=0.1)
        plt.plot((np.amax(time)-time), sfhs[0:,0:6],'k',lw=1, alpha=0.3)
        plt.ylabel('normalized SFR')
        plt.xlabel('lookback time [Gyr]')
        plt.show()

        txs = ['t'+'%.0f' %i for i in quantile_names(self.Nparam)]
        pg_labels = ['log sSFR']
        pg_labels[1:1] = txs

        arr = np.vstack((sfh_tuples[1,0:]-10, sfh_tuples[3:,0:]))
        corner.corner(arr.T, labels=pg_labels, plot_datapoints=False, fill_contours=True,
                bins=50, smooth=1.0, levels=[1 - np.exp(-(1/1)**2/2),1 - np.exp(-(2/1)**2/2)], label_kwargs={"fontsize": 30})
        plt.subplots_adjust(right=1.5,top=1.5)
        plt.show()
