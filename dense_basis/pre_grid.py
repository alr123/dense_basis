import numpy as np
from tqdm import tqdm
import scipy.io as sio
import os
import pkg_resources
import hickle

# cosmology assumption
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

from .priors import Priors
from .gp_sfh import (
    tuple_to_sfh,              # original DB
    calctimes, calctimes_to_tuple,
    continuity_to_sfh,         # you added this
    make_continuity_agebins    # you added this
)

try:
    import fsps
    mocksp = fsps.StellarPopulation(
        compute_vega_mags=False,
        zcontinuous=1,
        sfh=0,
        imf_type=1,
        logzsol=0.0,
        dust_type=2,
        dust2=0.0,
        add_neb_emission=True
    )
    print('Starting dense_basis. please wait ~ a minute for the FSPS backend to initialize.')
except Exception:
    mocksp = None
    print('Starting dense_basis. Failed to load FSPS, only GP-SFH module will be available.')

priors = Priors()

# -----------------------------------------------------------------------
#                     Calculating spectra and SEDs
# -----------------------------------------------------------------------

def get_path(filename):
    return os.path.dirname(os.path.realpath(filename))

def convert_to_microjansky(spec, z, cosmology):
    """
    Convert FSPS Lnu to Fnu in microJy
    """
    temp = (
        (1 + z) * spec * 1e6 * 1e23 * 3.48e33 /
        (4 * np.pi * 3.086e+24 * 3.086e+24 *
         cosmology.luminosity_distance(z).value *
         cosmology.luminosity_distance(z).value)
    )
    return temp

def makespec_atlas(atlas, galid, priors, sp, cosmo, filter_list=[], filt_dir=[], return_spec=False):
    sfh_tuple = atlas['sfh_tuple'][galid, 0:]
    zval = atlas['zval'][galid]
    dust = atlas['dust'][galid]
    met = atlas['met'][galid]

    specdetails = [sfh_tuple, dust, met, zval]
    if priors.dynamic_decouple and priors.sfh_type != "continuity":
        priors.decouple_sfr_time = 100 * cosmo.age(zval).value / cosmo.age(0.1).value

    output = makespec(specdetails, priors, sp, cosmo, filter_list, filt_dir, return_spec)
    return output

def make_colours(specdetails, sp, cosmo):
    """
    generate standard rest frame colours (UV,VJ,NUVr,rJ, NUVU) using FSPS
    """
    sp.params['zred'] = 0

    sfh_tuple, dust, met, zval = specdetails

    if np.isnan(zval):
        rest_nuv = rest_r = rest_u = rest_v = rest_j = -99
    else:
        rest_nuv, rest_r, rest_u, rest_v, rest_j = sp.get_mags(
            tage=cosmo.age(zval).value,
            bands=['galex_nuv', 'sdss_r', 'u', 'v', '2mass_j']
        )

    nuvu = rest_nuv - rest_u
    nuvr = rest_nuv - rest_r
    uv = rest_u - rest_v
    vj = rest_v - rest_j
    rj = rest_r - rest_j

    return nuvu, nuvr, uv, vj, rj

def makespec(
    specdetails, priors, sp, cosmo,
    filter_list=[], filt_dir=[], return_spec=False,
    peraa=False, input_sfh=False
):
    """
    makespec() works in two ways to create spectra or SEDs.
    with input_sfh = False, specdetails = [sfh_tuple, dust, met, zval]
    with input_sfh = True,  specdetails = [sfh, timeax, dust, met, zval]
    """

    # hardcoded FSPS params
    sp.params['sfh'] = 3
    sp.params['cloudy_dust'] = True
    sp.params['gas_logu'] = -2
    sp.params['add_igm_absorption'] = True
    sp.params['add_neb_emission'] = True
    sp.params['add_neb_continuum'] = True
    sp.params['imf_type'] = 1  # Chabrier

    if input_sfh:
        [sfh, timeax, dust, met, zval] = specdetails
    else:
        [sfh_tuple, dust, met, zval] = specdetails

        # ---- branch on SFH family ----
        if priors.sfh_type == "continuity":
            nbin = int(sfh_tuple[1])
            log_sfr_ratios = sfh_tuple[2:2 + nbin - 1]
            agebins = make_continuity_agebins(zval, nbin)
            timeax, sfh, _ = continuity_to_sfh(
                zred=zval,
                logmass=sfh_tuple[0],
                log_sfr_ratios=log_sfr_ratios,
                agebins=agebins
            )
        else:
            sfh, timeax = tuple_to_sfh(
                sfh_tuple, zval,
                decouple_sfr=priors.decouple_sfr,
                decouple_sfr_time=priors.decouple_sfr_time
            )

    sp.set_tabular_sfh(timeax, sfh)
    sp.params['dust2'] = dust
    sp.params['logzsol'] = met
    sp.params['gas_logz'] = met  # match stellar and gas metallicity
    sp.params['zred'] = zval

    lam, spec = sp.get_spectrum(
        tage=cosmo.age(zval).value + 1e-4,
        peraa=peraa
    )
    spec_ujy = convert_to_microjansky(spec, zval, cosmo)

    if isinstance(return_spec, bool):
        if return_spec:
            return lam, spec_ujy
        else:
            filcurves, _, _ = make_filvalkit_simple(
                lam, zval,
                fkit_name=filter_list, filt_dir=filt_dir
            )
            sed = calc_fnu_sed_fast(spec_ujy, filcurves)
            return sed
    elif len(return_spec) > 10:
        return convert_to_splined_spec(spec, lam, return_spec, zval)
    else:
        raise ValueError('Unknown argument for return_spec. Use True or False, or pass a wavelength grid.')

def convert_to_splined_spec(spec_peraa, lam, lam_spline, redshift, cosmology=cosmo):
    spec = spec_peraa
    spec_ergsec = spec * 3.839e33 * 1e17 / (1 + redshift)
    lum_dist = cosmology.luminosity_distance(redshift).value
    spec_ergsec_cm2 = spec_ergsec / (4 * np.pi * 3.086e+24 * 3.086e+24 * lum_dist * lum_dist)
    spec_spline = np.interp(lam_spline, lam * (1 + redshift), spec_ergsec_cm2)
    return spec_spline

def make_sed_fast(sfh_tuple, metval, dustval, zval, filcurves,
                  igmval=True, return_lam=False, sp=mocksp, cosmology=cosmo):
    spec, logsfr, logmstar = make_spec(
        sfh_tuple, metval, dustval, zval,
        igmval=True, return_ms=True, return_lam=False,
        sp=sp, cosmology=cosmology
    )
    sed = calc_fnu_sed_fast(spec, filcurves)
    return sed, logsfr, logmstar

def make_filvalkit_simple(lam, z, fkit_name='filter_list.dat', vb=False, filt_dir='filters/'):
    lam_z = (1 + z) * lam
    lam_z_lores = np.arange(2000, 150000, 2000)

    if filt_dir == 'internal':
        resource_package = __name__
        resource_path = '/'.join(('filters', fkit_name))
        template = pkg_resources.resource_string(resource_package, resource_path)
        temp = template.split()
    else:
        if filt_dir[-1] == '/':
            f = open(filt_dir + fkit_name, 'r')
        else:
            f = open(filt_dir + '/' + fkit_name, 'r')
        temp = f.readlines()

    if vb:
        print('number of filters to be read in: ' + str(len(temp)))

    numlines = len(temp)
    if temp[-1] == '\n':
        numlines = len(temp) - 1

    filcurves = np.zeros((len(lam_z), numlines))
    filcurves_lores = np.zeros((len(lam_z_lores), numlines))

    if vb:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))

    for i in range(numlines):
        if (filt_dir == 'internal') and (fkit_name == 'filter_list_goodss.dat'):
            tempfilt = np.loadtxt(get_file('filters/filter_curves/goods_s', temp[i][22:].decode("utf-8")))
        elif (filt_dir == 'internal') and (fkit_name == 'filter_list_goodsn.dat'):
            tempfilt = np.loadtxt(get_file('filters/filter_curves/goods_n', temp[i][22:].decode("utf-8")))
        elif (filt_dir == 'internal') and (fkit_name == 'filter_list_cosmos.dat'):
            tempfilt = np.loadtxt(get_file('filters/filter_curves/cosmos', temp[i][21:].decode("utf-8")))
        elif (filt_dir == 'internal') and (fkit_name == 'filter_list_egs.dat'):
            tempfilt = np.loadtxt(get_file('filters/filter_curves/egs', temp[i][18:].decode("utf-8")))
        elif (filt_dir == 'internal') and (fkit_name == 'filter_list_uds.dat'):
            tempfilt = np.loadtxt(get_file('filters/filter_curves/uds', temp[i][18:].decode("utf-8")))
        else:
            if filt_dir[-1] == '/':
                filt_name = filt_dir + temp[i]
            else:
                filt_name = filt_dir + '/' + temp[i]
            if i == numlines - 1:
                tempfilt = np.loadtxt(filt_name[0:-1])
            else:
                if os.path.exists(filt_name[0:-1]):
                    tempfilt = np.loadtxt(filt_name[0:-1])
                else:
                    raise Exception('filters not found. are you sure the folder exists at the right relative path?')

        temp_lam_arr = tempfilt[:, 0]
        temp_response_curve = tempfilt[:, 1]

        bot_in = np.argmin(np.abs(lam_z - np.amin(temp_lam_arr)))
        top_in = np.argmin(np.abs(lam_z - np.amax(temp_lam_arr)))

        curve_small = np.interp(lam_z[bot_in+1:top_in-1], temp_lam_arr, temp_response_curve)
        splinedcurve = np.zeros(lam_z.shape)
        splinedcurve[bot_in+1:top_in-1] = curve_small
        if np.amax(splinedcurve) > 1:
            splinedcurve = splinedcurve / np.amax(splinedcurve)

        filcurves[:, i] = splinedcurve

        if vb:
            plt.plot(np.log10(lam_z), splinedcurve, 'k--', label=filt_name[0:-1])

    if (filt_dir != 'internal'):
        f.close()

    if vb:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel(r'log $\lambda [\AA]$')
        plt.ylabel('Filter transmission')
        plt.axis([3.5, 5, 0, 1])
        plt.show()

    return filcurves, lam_z, lam_z_lores

def calc_fnu_sed(spec, z, lam, fkit_name='filter_list.dat', filt_dir='filters/'):
    filcurves, lam_z, lam_z_lores = make_filvalkit_simple(
        lam, z, fkit_name=fkit_name, filt_dir=filt_dir
    )
    fnuspec = spec
    filvals = np.zeros((filcurves.shape[1],))
    for tindex in range(filcurves.shape[1]):
        temp1 = filcurves[np.where(filcurves[:, tindex] > 0), tindex]
        temp2 = fnuspec[np.where(filcurves[:, tindex] > 0)]
        filvals[tindex] = np.sum(temp1.T[:, 0] * temp2) / np.sum(filcurves[:, tindex])
    return filvals

def calc_fnu_sed_fast(fnuspec, filcurves):
    filvals = np.zeros((filcurves.shape[1],))
    for tindex in range(filcurves.shape[1]):
        temp1 = filcurves[np.where(filcurves[:, tindex] > 0), tindex]
        temp2 = fnuspec[np.where(filcurves[:, tindex] > 0)]
        filvals[tindex] = np.sum(temp1.T[:, 0] * temp2) / np.sum(filcurves[:, tindex])
    return filvals

def generate_atlas(
    N_pregrid=10, priors=priors, initial_seed=42, store=True,
    filter_list='filter_list.dat', filt_dir='filters/', norm_method='median',
    z_step=0.01, sp=mocksp, cosmology=cosmo, fname=None, path='pregrids/',
    lam_array_spline=[], rseed=None
):
    """
    Generate a pregrid of galaxy properties and their corresponding SEDs.
    Now works for both gp and continuity SFHs.
    """

    print('generating atlas with:')
    print('  SFH type:', priors.sfh_type)
    print('  SFR sampling:', priors.sfr_prior_type)
    print('  SFH treatment:', priors.sfh_treatment)
    print('  met sampling:', priors.met_treatment)
    print('  dust:', priors.dust_model, priors.dust_prior)
    print('  decouple SFR:', priors.decouple_sfr)

    if rseed is not None:
        print('setting random seed to :', rseed)
        np.random.seed(rseed)

    zval_all = []
    sfh_tuple_all = []
    sfh_tuple_rec_all = []
    norm_all = []
    dust_all = []
    met_all = []
    sed_all = []
    mstar_all = []
    sfr_all = []
    nuvu_all = []
    uv_all = []
    vj_all = []
    rj_all = []
    nuvr_all = []

    # filter grids: build once
    fcs = None
    fc_zgrid = None

    for i in tqdm(range(int(N_pregrid))):

        # draw everything at once (works for gp and continuity)
        sfh_tuple, met, dust, zval = priors.sample_all_params(
            random_seed=initial_seed + i * 7
        )

        # for gp: adjust decouple time
        if priors.dynamic_decouple and priors.sfh_type != "continuity":
            priors.decouple_sfr_time = 100 * cosmo.age(zval).value / cosmo.age(0.1).value

        # for bookkeeping: reconstruct SFH (or just keep tuple for continuity)
        if priors.sfh_type == "continuity":
            nbin = int(sfh_tuple[1])
            agebins = make_continuity_agebins(zval, nbin)
            timeax, sfh, _ = continuity_to_sfh(
                zred=zval,
                logmass=sfh_tuple[0],
                log_sfr_ratios=sfh_tuple[2:2 + nbin - 1],
                agebins=agebins
            )
            temptuple = sfh_tuple.copy()  # we can just store original
        else:
            sfh, timeax = tuple_to_sfh(
                sfh_tuple, zval,
                decouple_sfr=priors.decouple_sfr,
                decouple_sfr_time=priors.decouple_sfr_time
            )
            temp = calctimes(timeax, sfh, priors.Nparam)
            temptuple = calctimes_to_tuple(temp)

        specdetails = [sfh_tuple, dust, met, zval]

        # make SED
        if len(lam_array_spline) > 0:
            sed = makespec(
                specdetails, priors, sp, cosmology,
                filter_list, filt_dir,
                return_spec=lam_array_spline, peraa=True
            )
        else:
            lam, spec_ujy = makespec(
                specdetails, priors, sp, cosmology,
                filter_list, filt_dir, return_spec=True
            )

            if i == 0:
                fc_zgrid = np.arange(priors.z_min - z_step, priors.z_max + z_step, z_step)
                temp_fc, temp_lz, temp_lz_lores = make_filvalkit_simple(
                    lam, priors.z_min,
                    fkit_name=filter_list, filt_dir=filt_dir
                )
                fcs = np.zeros((temp_fc.shape[0], temp_fc.shape[1], len(fc_zgrid)))
                for j, zfc in enumerate(fc_zgrid):
                    fcs[:, :, j], _, _ = make_filvalkit_simple(
                        lam, zfc,
                        fkit_name=filter_list, filt_dir=filt_dir
                    )

            fc_index = np.argmin(np.abs(zval - fc_zgrid))
            sed = calc_fnu_sed_fast(spec_ujy, fcs[:, :, fc_index])

        # normalization
        if norm_method == 'none':
            norm_fac = 1
        elif norm_method == 'max':
            norm_fac = np.nanmax(sed)
        elif norm_method == 'median':
            norm_fac = np.nanmedian(sed)
        elif norm_method == 'area':
            norm_fac = 10 ** (sfh_tuple[0] - 9)
        else:
            raise ValueError('undefined normalization argument')

        sed = sed / norm_fac
        norm = 1.0 / norm_fac

        mstar = np.log10(sp.stellar_mass / norm_fac)
        sfr = np.log10(sp.sfr / norm_fac)

        # adjust tuple for normalization:
        sfh_tuple_normed = sfh_tuple.copy()
        sfh_tuple_normed[0] = sfh_tuple_normed[0] - np.log10(norm_fac)
        if priors.sfh_type != "continuity":
            # gp version has logSFR at index 1
            sfh_tuple_normed[1] = sfh_tuple_normed[1] - np.log10(norm_fac)

        temptuple_normed = temptuple.copy()
        if len(temptuple_normed) > 1:
            temptuple_normed[0] = temptuple_normed[0] - np.log10(norm_fac)
            if priors.sfh_type != "continuity":
                temptuple_normed[1] = temptuple_normed[1] - np.log10(norm_fac)

        # colours
        nuvu, nuvr, uv, vj, rj = make_colours(specdetails, sp, cosmo)

        zval_all.append(zval)
        sfh_tuple_all.append(sfh_tuple_normed)
        sfh_tuple_rec_all.append(temptuple_normed)
        norm_all.append(norm)
        dust_all.append(dust)
        met_all.append(met)
        sed_all.append(sed)
        mstar_all.append(mstar)
        sfr_all.append(sfr)
        nuvu_all.append(nuvu)
        nuvr_all.append(nuvr)
        uv_all.append(uv)
        vj_all.append(vj)
        rj_all.append(rj)

    # pad variable-length tuples
    maxlen_tuple = max(len(x) for x in sfh_tuple_all)
    sfh_tuple_arr = np.full((len(sfh_tuple_all), maxlen_tuple), np.nan)
    for i, arr in enumerate(sfh_tuple_all):
        sfh_tuple_arr[i, :len(arr)] = arr

    maxlen_rec = max(len(x) for x in sfh_tuple_rec_all)
    sfh_tuple_rec_arr = np.full((len(sfh_tuple_rec_all), maxlen_rec), np.nan)
    for i, arr in enumerate(sfh_tuple_rec_all):
        sfh_tuple_rec_arr[i, :len(arr)] = arr

    pregrid_dict = {
        'zval': np.array(zval_all),
        'sfh_tuple': sfh_tuple_arr,
        'sfh_tuple_rec': sfh_tuple_rec_arr,
        'norm': np.array(norm_all),
        'norm_method': norm_method,
        'mstar': np.array(mstar_all),
        'sfr': np.array(sfr_all),
        'dust': np.array(dust_all),
        'met': np.array(met_all),
        'sed': np.array(sed_all),
        'nuvu': np.array(nuvu_all),
        'nuvr': np.array(nuvr_all),
        'uv': np.array(uv_all),
        'vj': np.array(vj_all),
        'rj': np.array(rj_all),
        'sfh_type': priors.sfh_type,
        'continuity_nbin': getattr(priors, 'continuity_nbin', None)
    }

    if store:
        if fname is None:
            fname = 'sfh_pregrid_size'

        # pick a sensible suffix for filename
        if priors.sfh_type == "continuity":
            suffix = priors.continuity_nbin
        else:
            suffix = priors.Nparam

        if not os.path.exists(path):
            os.mkdir(path)
            print('Created directory and saved atlas at : ' + path)
        outname = f"{path}{fname}_{N_pregrid}_Nparam_{suffix}.dbatlas"
        print('Saved atlas at : ' + outname)
        try:
            hickle.dump(pregrid_dict, outname,
                        compression='gzip', compression_opts=9)
        except Exception:
            print('storing without compression')
            hickle.dump(pregrid_dict, outname)
        return

    return pregrid_dict

def load_atlas(fname, N_pregrid, N_param, path='pregrids/'):
    fname_full = path + fname + '_' + str(N_pregrid) + '_Nparam_' + str(N_param) + '.dbatlas'
    cat = hickle.load(fname_full)
    return cat

def quantile_names(N_params):
    return (np.round(np.linspace(0, 100, N_params + 2)))[1:-1]
