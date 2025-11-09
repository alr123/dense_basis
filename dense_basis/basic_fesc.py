# basic_fesc.py
import numpy as np
import os
import hickle
from tqdm import tqdm

from .priors import *
from .pre_grid import *
from .gp_sfh import *
from .plotter import *
from .sed_fitter import *


def sample_fesc_prior():
    """Uniform prior between 0 and 1."""
    return np.random.uniform()


def get_k(f_esc=0.0, f_dust=0.0):
    """
    Factor for downscaling nebular line spectrum, from Inoue (2011),
    implemented as in Boquien+20 (A&A 622, A103).
    """
    Te = 1e4  # Kelvin
    alphaB_Te = 2.58e-19  # m^3/s
    alpha1_Te = 1.54e-19  # m^3/s
    alpharatio = alpha1_Te / alphaB_Te
    return (1 - f_esc - f_dust) / (1 + alpharatio * (f_esc + f_dust))


def makespec_fesc(
    specdetails,
    fesc,
    priors,
    sp,
    cosmo,
    filter_list=[],
    filt_dir=[],
    return_spec=False,
    peraa=False,
    input_sfh=False,
):
    """
    Create an SED or spectrum with an adjustable f_esc fraction.
    """

    # --- FSPS defaults ---
    sp.params['sfh'] = 3
    sp.params['cloudy_dust'] = True
    sp.params['gas_logu'] = -2
    sp.params['add_igm_absorption'] = True
    sp.params['add_neb_emission'] = True
    sp.params['add_neb_continuum'] = True
    sp.params['imf_type'] = 1  # Chabrier

    # --- Parse input ---
    if input_sfh:
        [sfh, timeax, dust, met, zval] = specdetails
    else:
        [sfh_tuple, dust, met, zval] = specdetails
        sfh, timeax = tuple_to_sfh(
            sfh_tuple,
            zval,
            decouple_sfr=priors.decouple_sfr,
            decouple_sfr_time=priors.decouple_sfr_time,
        )

    sp.set_tabular_sfh(timeax, sfh)
    sp.params['dust2'] = dust
    sp.params['logzsol'] = met
    sp.params['gas_logz'] = met
    sp.params['zred'] = zval

    # --- Non-nebular baseline ---
    sp.params['add_neb_emission'] = False
    sp.params['add_neb_continuum'] = False
    lam, spec_noneb = sp.get_spectrum(tage=cosmo.age(zval).value + 1e-4, peraa=peraa)

    # --- Full spectrum (nebular included) ---
    sp.params['add_neb_emission'] = True
    sp.params['add_neb_continuum'] = True
    lam, spec_full = sp.get_spectrum(tage=cosmo.age(zval).value + 1e-4, peraa=peraa)

    neb = (spec_full - spec_noneb) * get_k(f_esc=fesc)
    spec = spec_noneb + neb

    spec_ujy = convert_to_microjansky(spec, zval, cosmo)

    # --- Output modes ---
    if isinstance(return_spec, bool):
        if return_spec:
            return lam, spec_ujy
        else:
            filcurves, _, _ = make_filvalkit_simple(
                lam, zval, fkit_name=filter_list, filt_dir=filt_dir
            )
            return calc_fnu_sed_fast(spec_ujy, filcurves)

    elif hasattr(return_spec, "__len__") and len(return_spec) > 10:
        return convert_to_splined_spec(spec, lam, return_spec, zval)

    raise ValueError("Unknown argument for return_spec. Use True, False, or pass a wavelength grid.")


def generate_atlas_fesc(
    N_pregrid=10,
    priors=priors,
    initial_seed=42,
    store=True,
    filter_list='filter_list.dat',
    filt_dir='filters/',
    norm_method='median',
    z_step=0.01,
    sp=mocksp,
    cosmology=cosmo,
    fname=None,
    path='pregrids/',
    lam_array_spline=[],
    rseed=None,
):
    """
    Generate a pregrid including an f_esc parameter.
    """

    print(
        f"Generating atlas with: {priors.Nparam} tx parameters, {priors.sfr_prior_type} SFR prior, "
        f"{priors.sfh_treatment} SFH treatment, {priors.met_treatment} metallicity sampling, "
        f"{priors.dust_model} dust model, {priors.dust_prior} dust prior, "
        f"decoupled SFR={priors.decouple_sfr}."
    )

    if rseed is not None:
        np.random.seed(rseed)

    zval_all, sfh_tuple_all, sfh_tuple_rec_all = [], [], []
    norm_all, dust_all, met_all, sed_all = [], [], [], []
    mstar_all, sfr_all, fesc_all = [], [], []

    for i in tqdm(range(int(N_pregrid))):
        zval = priors.sample_z_prior()
        massval = priors.sample_mass_prior()
        sfrval = priors.sample_sfr_prior(zval=zval) if priors.sfr_prior_type == 'sSFRlognormal' else priors.sample_sfr_prior()
        txparam = priors.sample_tx_prior()
        sfh_tuple = np.hstack((massval, sfrval, priors.Nparam, txparam))
        norm = 1.0

        if priors.dynamic_decouple:
            priors.decouple_sfr_time = 100 * cosmo.age(zval).value / cosmo.age(0.1).value

        sfh, timeax = tuple_to_sfh(
            sfh_tuple,
            zval,
            decouple_sfr=priors.decouple_sfr,
            decouple_sfr_time=priors.decouple_sfr_time,
        )

        temp = calctimes(timeax, sfh, priors.Nparam)
        temptuple = calctimes_to_tuple(temp)
        dust = priors.sample_Av_prior()
        met = priors.sample_Z_prior()

        try:
            fesc = priors.sample_fesc_prior()
        except AttributeError:
            raise Exception("Define priors.sample_fesc_prior() before generating atlas.")

        specdetails = [sfh_tuple, dust, met, zval]

        if len(lam_array_spline) > 0:
            sed = makespec_fesc(
                specdetails, fesc, priors, sp, cosmology,
                filter_list, filt_dir, return_spec=lam_array_spline, peraa=True
            )
        else:
            lam, spec_ujy = makespec_fesc(
                specdetails, fesc, priors, sp, cosmology,
                filter_list, filt_dir, return_spec=True
            )
            if i == 0:
                fc_zgrid = np.arange(priors.z_min - z_step, priors.z_max + z_step, z_step)
                temp_fc, temp_lz, temp_lz_lores = make_filvalkit_simple(
                    lam, priors.z_min, fkit_name=filter_list, filt_dir=filt_dir
                )
                fcs = np.zeros((temp_fc.shape[0], temp_fc.shape[1], len(fc_zgrid)))
                for j, ztemp in enumerate(fc_zgrid):
                    fcs[:, :, j], _, _ = make_filvalkit_simple(
                        lam, ztemp, fkit_name=filter_list, filt_dir=filt_dir
                    )
            fc_index = np.argmin(np.abs(zval - fc_zgrid))
            sed = calc_fnu_sed_fast(spec_ujy, fcs[:, :, fc_index])

        # --- normalization ---
        if norm_method == 'none':
            norm_fac = 1
        elif norm_method == 'max':
            norm_fac = np.amax(sed)
        elif norm_method == 'median':
            norm_fac = np.nanmedian(sed)
        elif norm_method == 'area':
            norm_fac = 10 ** (massval - 9)
        else:
            raise ValueError('Undefined normalization argument.')

        sed /= norm_fac
        norm /= norm_fac
        mstar = np.log10(sp.stellar_mass / norm_fac)
        sfr = np.log10(sp.sfr / norm_fac)
        sfh_tuple[0:2] -= np.log10(norm_fac)
        temptuple[0:2] -= np.log10(norm_fac)

        # --- collect ---
        zval_all.append(zval)
        sfh_tuple_all.append(sfh_tuple)
        sfh_tuple_rec_all.append(temptuple)
        norm_all.append(norm)
        dust_all.append(dust)
        met_all.append(met)
        sed_all.append(sed)
        mstar_all.append(mstar)
        sfr_all.append(sfr)
        fesc_all.append(fesc)

    pregrid_dict = {
        'zval': np.array(zval_all),
        'sfh_tuple': np.array(sfh_tuple_all),
        'sfh_tuple_rec': np.array(sfh_tuple_rec_all),
        'norm': np.array(norm_all),
        'norm_method': norm_method,
        'mstar': np.array(mstar_all),
        'sfr': np.array(sfr_all),
        'dust': np.array(dust_all),
        'met': np.array(met_all),
        'fesc': np.array(fesc_all),
        'sed': np.array(sed_all),
    }

    if store:
        os.makedirs(path, exist_ok=True)
        if fname is None:
            fname = 'sfh_pregrid_size'
        outfile = os.path.join(path, f"{fname}_{N_pregrid}_Nparam_{priors.Nparam}.dbatlas")
        print(f"Saving atlas to: {outfile}")
        try:
            hickle.dump(pregrid_dict, outfile, compression='gzip', compression_opts=9)
        except Exception:
            print("storing without compression")
            hickle.dump(pregrid_dict, outfile)

    return pregrid_dict


def plot_posteriors_fesc(sedfit, truths=[], **kwargs):
    """Corner plot including f_esc parameter."""
    chi2_array = sedfit.chi2_array
    norm_fac = sedfit.norm_fac
    atlas = sedfit.atlas
    set_plot_style()

    sfrvals = atlas['sfr'].copy()
    sfrvals[sfrvals < -3] = -3

    pg_params = np.vstack([
        atlas['mstar'],
        sfrvals,
        atlas['sfh_tuple'][:, 3:].T,
        atlas['met'].ravel(),
        atlas['dust'].ravel(),
        atlas['fesc'].ravel(),
        atlas['zval'].ravel()
    ])
    txs = ['t' + f"{i:.0f}" for i in quantile_names(pg_params.shape[0] - 6)]
    pg_labels = ['log M$_*$', 'log SFR', *txs, 'Z', 'Av', 'f$_{esc}$', 'z']

    corner_params = pg_params.copy()
    corner_params[0, :] += np.log10(norm_fac)
    corner_params[1, :] += np.log10(norm_fac)

    fig = corner.corner(
        corner_params.T,
        weights=np.exp(-chi2_array / 2),
        labels=pg_labels,
        truths=(truths if len(truths) > 0 else None),
        plot_datapoints=False,
        fill_contours=True,
        bins=20,
        smooth=1.0,
        quantiles=(0.16, 0.84),
        levels=[1 - np.exp(-0.5), 1 - np.exp(-2)],
        label_kwargs={"fontsize": 30},
        show_titles=True,
        **kwargs
    )
    fig.subplots_adjust(right=1.5, top=1.5)
    return fig


def calc_fesc_posteriors(sedfit):
    """Add f_esc posterior percentiles to a SedFit object."""
    fesc_vals = get_quants_key('fesc', 50, sedfit.chi2_array, sedfit.atlas, sedfit.norm_fac)
    sedfit.fesc = fesc_vals
    return
