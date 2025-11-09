# parallelization.py

import os
import time
import glob
import numpy as np

try:
    from schwimmbad import SerialPool, MultiPool
    from functools import partial
except Exception:
    print('running without parallelization.')
    MultiPool = None
    from functools import partial

import hickle
from astropy.table import Table
import pylab as pl
from IPython import display

from .pre_grid import generate_atlas
from .sed_fitter import evaluate_sed_likelihood, get_quants
from .priors import Priors


# ---------------------------------------------------------------------
# helpers for padding variable-length 2D arrays
# ---------------------------------------------------------------------
def _vstack_pad(a, b, fill=np.nan):
    """
    Stack two 2D arrays with possibly different column counts
    by padding the smaller one with `fill`.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    na, ma = a.shape
    nb, mb = b.shape
    m = max(ma, mb)

    a_pad = np.full((na, m), fill)
    b_pad = np.full((nb, m), fill)
    a_pad[:, :ma] = a
    b_pad[:, :mb] = b
    return np.vstack((a_pad, b_pad))


# ---------------------------------------------------------------------
# chunk generator
# ---------------------------------------------------------------------
def gen_pg_parallel(data_i, atlas_vals):
    """
    Generate the i-th chunk in parallel
    """
    fname, priors, pg_folder, filter_list, filt_dir, N_pregrid = atlas_vals
    fname_full = f"{fname}_chunk_{data_i}"

    generate_atlas(
        N_pregrid=N_pregrid,
        priors=priors,
        fname=fname_full,
        store=True,
        path=pg_folder,
        filter_list=filter_list,
        filt_dir=filt_dir,
        rseed=(N_pregrid * data_i + 1),
    )
    return


def generate_atlas_in_parallel_chunking(
    chunksize,
    nchunks,
    fname='temp_parallel_atlas',
    filter_list='filter_list_goodss.dat',
    filt_dir='internal',
    priors=None,
    pg_folder='pregrids/',
):
    """
    Generate chunks of an atlas in parallel and combine them into one big atlas.
    Works for both gp-style and continuity-style priors.
    """
    if priors is None:
        priors = Priors()

    chunk_path = os.path.join(pg_folder, 'atlaschunks')
    store_path = pg_folder

    os.makedirs(chunk_path, exist_ok=True)

    atlas_vals = [fname, priors, chunk_path, filter_list, filt_dir, chunksize]

    time_start = time.time()
    data = np.arange(nchunks)

    if MultiPool is not None:
        try:
            with MultiPool() as pool:
                list(pool.map(partial(gen_pg_parallel, atlas_vals=atlas_vals), data))
        finally:
            pass
    else:
        # serial fallback
        for d in data:
            gen_pg_parallel(d, atlas_vals)

    print('Generated pregrid (%d chunks, %d sedsperchunk)' % (nchunks, chunksize))
    print('time taken: %.2f mins.' % ((time.time() - time_start)/60.0))

    combine_pregrid_chunks(
        fname_base=fname,
        N_chunks=nchunks,
        N_pregrid=chunksize,
        N_param=getattr(priors, "Nparam", 3),
        chunk_path=chunk_path,
        store_path=store_path,
    )

    return


# ---------------------------------------------------------------------
# chunk combiner
# ---------------------------------------------------------------------
def _find_chunk_file(chunk_path, fname_base, i):
    """
    Find a file for chunk i. We try a couple of patterns because the exact suffix
    (Nparam or #bins) depends on the SFH family.
    """
    # most generic: anything that starts with this
    pattern = os.path.join(chunk_path, f"{fname_base}_chunk_{i}*.dbatlas")
    matches = glob.glob(pattern)
    if len(matches) == 0:
        raise FileNotFoundError(f"Could not find chunk {i} in {chunk_path} with base {fname_base}")
    # pick the first one (they should all be for this chunk)
    return matches[0]


def combine_pregrid_chunks(
    fname_base,
    N_chunks,
    N_pregrid,
    N_param,
    chunk_path='pregrids/atlaschunks/',
    store_path='pregrids/',
):
    """
    Combine chunks of a large pregrid generated in parallel.

    IMPORTANT: newer pregrids may have variable-length SFH arrays (continuity),
    so we pad 2D arrays to the widest width we see before stacking.
    """

    # load first chunk
    first_file = _find_chunk_file(chunk_path, fname_base, 0)
    atlas_big = hickle.load(first_file)

    for i in range(1, N_chunks):
        this_file = _find_chunk_file(chunk_path, fname_base, i)
        atlas = hickle.load(this_file)

        for key in atlas.keys():
            if key == 'norm_method':
                continue

            try:
                a_big = atlas_big[key]
                a_new = atlas[key]

                # 0D / 1D -> hstack
                if not hasattr(a_big, "shape") or np.ndim(a_big) <= 1:
                    atlas_big[key] = np.hstack((np.array(a_big), np.array(a_new)))
                else:
                    # 2D or higher
                    if a_big.ndim == 2 and np.ndim(a_new) == 2:
                        atlas_big[key] = _vstack_pad(a_big, a_new)
                    else:
                        # fallback — try hstack
                        atlas_big[key] = np.hstack((a_big, a_new))
            except Exception as e:
                print('didnt combine ', key, 'because', e)

    # store combined
    os.makedirs(store_path, exist_ok=True)
    totsize = N_pregrid * N_chunks

    # we don't know the final "Nparam" in the filename if continuity, so keep what caller gave
    outname = os.path.join(
        store_path,
        f"{fname_base}_combined_{totsize}_Nparam_{N_param}.dbatlas"
    )
    print('Saved atlas at : ' + outname)
    try:
        hickle.dump(atlas_big, outname, compression='gzip', compression_opts=9)
    except Exception:
        print('storing without compression')
        hickle.dump(atlas_big, outname)

    return


# ---------------------------------------------------------------------
# single-galaxy fitter (used by pools)
# ---------------------------------------------------------------------
def fit_gals(gal_id, catvals):
    """
    Fit a single galaxy SED to an atlas.
    """
    if len(catvals) == 3:
        cat_seds, cat_errs, atlas = catvals
        fit_mask = []
    elif len(catvals) == 4:
        cat_seds, cat_errs, fit_mask, atlas = catvals
    else:
        raise ValueError('wrong number of arguments supplied to fitter')

    gal_sed = cat_seds[gal_id, :].copy()
    gal_err = cat_errs[gal_id, :].copy()

    chi2, norm_fac = evaluate_sed_likelihood(
        gal_sed,
        gal_err,
        atlas,
        fit_mask=fit_mask,
        zbest=None,
        deltaz=None
    )

    quants = get_quants(chi2, atlas, norm_fac)

    return quants, chi2


# ---------------------------------------------------------------------
# deprecated: redshift-sliced parallel generation
# ---------------------------------------------------------------------
def make_atlas_parallel(zval, atlas_params):
    """
    Make a single atlas given a redshift value and a list of parameters (including a priors object).
    Atlas Params: [N_pregrid, priors, fname, store, path, filter_list, filt_dir, z_bw]
    """
    N_pregrid, priors, fname, store, path, filter_list, filt_dir, z_bw = atlas_params

    priors.z_max = zval + z_bw/2
    priors.z_min = zval - z_bw/2

    fname = fname + '_zval_%.0f_' % (zval * 10000)

    generate_atlas(
        N_pregrid=N_pregrid,
        priors=priors,
        fname=fname,
        store=True,
        path=path,
        filter_list=filter_list,
        filt_dir=filt_dir,
        rseed=int(zval * 100000)
    )

    return


def generate_atlas_in_parallel_zgrid(zgrid, atlas_params, dynamic_decouple=True):
    """
    Make a set of atlases given a redshift grid and a list of parameters (including a priors object).
    Atlas Params: [N_pregrid, priors, fname, store, path, filter_list, filt_dir, z_bw]
    """

    time_start = time.time()

    if MultiPool is not None:
        try:
            with MultiPool() as pool:
                list(pool.map(partial(make_atlas_parallel, atlas_params=atlas_params), zgrid))
        finally:
            pass
    else:
        # serial fallback
        for z in zgrid:
            make_atlas_parallel(z, atlas_params)

    time_end = time.time()
    print('time taken [parallel]: %.2f min.' % ((time_end - time_start)/60))


# ---------------------------------------------------------------------
# big catalog fitter (your original logic, slightly tidied)
# ---------------------------------------------------------------------
def fit_catalog(
    fit_cat,
    atlas_path,
    atlas_fname,
    output_fname,
    N_pregrid=10000,
    N_param=3,
    z_bw=0.05,
    f160_cut=100,
    fit_mask=[],
    zgrid=[],
    sfr_uncert_cutoff=2.0
):
    """
    Fit an entire catalog by slicing in redshift and calling the atlas fitter in parallel.
    This will work for continuity atlases too, because sed_fitter.get_quants()
    already branches on atlas['sfh_type'].
    """

    cat_id, cat_zbest, cat_seds, cat_errs, cat_f160, cat_class_star = fit_cat

    if isinstance(zgrid, np.ndarray) is False:
        zgrid = np.arange(np.amin(cat_zbest), np.amax(cat_zbest), z_bw)

    fit_id = cat_id.copy()
    fit_logM_50 = np.zeros_like(cat_zbest)
    fit_logM_MAP = np.zeros_like(cat_zbest)
    fit_logM_16 = np.zeros_like(cat_zbest)
    fit_logM_84 = np.zeros_like(cat_zbest)
    fit_logSFRinst_50 = np.zeros_like(cat_zbest)
    fit_logSFRinst_MAP = np.zeros_like(cat_zbest)
    fit_logSFRinst_16 = np.zeros_like(cat_zbest)
    fit_logSFRinst_84 = np.zeros_like(cat_zbest)

    fit_logZsol_50 = np.zeros_like(cat_zbest)
    fit_logZsol_16 = np.zeros_like(cat_zbest)
    fit_logZsol_84 = np.zeros_like(cat_zbest)
    fit_Av_50 = np.zeros_like(cat_zbest)
    fit_Av_16 = np.zeros_like(cat_zbest)
    fit_Av_84 = np.zeros_like(cat_zbest)

    fit_zfit_50 = np.zeros_like(cat_zbest)
    fit_zfit_16 = np.zeros_like(cat_zbest)
    fit_zfit_84 = np.zeros_like(cat_zbest)
    fit_logMt_50 = np.zeros_like(cat_zbest)
    fit_logMt_16 = np.zeros_like(cat_zbest)
    fit_logMt_84 = np.zeros_like(cat_zbest)
    fit_logSFR100_50 = np.zeros_like(cat_zbest)
    fit_logSFR100_16 = np.zeros_like(cat_zbest)
    fit_logSFR100_84 = np.zeros_like(cat_zbest)
    fit_nparam = np.zeros_like(cat_zbest)
    fit_t25_50 = np.zeros_like(cat_zbest)
    fit_t25_16 = np.zeros_like(cat_zbest)
    fit_t25_84 = np.zeros_like(cat_zbest)
    fit_t50_50 = np.zeros_like(cat_zbest)
    fit_t50_16 = np.zeros_like(cat_zbest)
    fit_t50_84 = np.zeros_like(cat_zbest)
    fit_t75_50 = np.zeros_like(cat_zbest)
    fit_t75_16 = np.zeros_like(cat_zbest)
    fit_t75_84 = np.zeros_like(cat_zbest)

    fit_nbands = np.zeros_like(cat_zbest)
    fit_f160w = np.zeros_like(cat_zbest)
    fit_stellarity = np.zeros_like(cat_zbest)
    fit_chi2 = np.zeros_like(cat_zbest)
    fit_flags = np.zeros_like(cat_zbest)

    for i in (range(len(zgrid))):

        print('loading atlas at', zgrid[i])

        zval = zgrid[i]

        z_mask = (
            (cat_zbest < (zval + z_bw/2)) &
            (cat_zbest > (zval - z_bw/2)) &
            (cat_f160 < f160_cut)
        )
        fit_ids = np.arange(len(cat_zbest))[z_mask]

        print('starting parallel fitting for Ngals = ', len(fit_ids), ' at redshift ', str(zval))

        try:
            # load atlas for this redshift slice
            # original naming convention
            fname = atlas_fname + '_zval_%.0f_' % (zgrid[i] * 10000)
            fname_full = os.path.join(
                atlas_path,
                f"{fname}{N_pregrid}_Nparam_{N_param}.dbatlas"
            )
            if not os.path.exists(fname_full):
                # try glob, in case continuity changed the suffix
                pattern = os.path.join(atlas_path, f"{fname}*.dbatlas")
                matches = glob.glob(pattern)
                if len(matches) == 0:
                    raise FileNotFoundError(f"could not find atlas file for z ~ {zgrid[i]}")
                fname_full = matches[0]

            atlas = hickle.load(fname_full)
            print('loaded atlas')

            if MultiPool is not None:
                with MultiPool() as pool:
                    if isinstance(fit_mask, np.ndarray) is False:
                        all_quants = list(pool.map(partial(fit_gals, catvals=(cat_seds, cat_errs, atlas)), fit_ids))
                    else:
                        all_quants = list(pool.map(partial(fit_gals, catvals=(cat_seds, cat_errs, fit_mask, atlas)), fit_ids))
            else:
                # serial
                all_quants = []
                for gid in fit_ids:
                    if isinstance(fit_mask, np.ndarray) is False:
                        all_quants.append(fit_gals(gid, (cat_seds, cat_errs, atlas)))
                    else:
                        all_quants.append(fit_gals(gid, (cat_seds, cat_errs, fit_mask, atlas)))

            print('finished fitting parallel zbest chunk at z=%.3f' % zval)

            print('starting to put values in arrays')
            for ii, gal_id in enumerate(fit_ids):

                gal_sed = cat_seds[gal_id, 0:]
                gal_err = cat_errs[gal_id, 0:]

                quants = all_quants[ii][0]
                fit_likelihood = all_quants[ii][1]

                fit_logM_50[gal_id] = quants[0][0]
                fit_logM_16[gal_id] = quants[0][1]
                fit_logM_84[gal_id] = quants[0][2]
                fit_logSFRinst_50[gal_id] = quants[1][0]
                fit_logSFRinst_16[gal_id] = quants[1][1]
                fit_logSFRinst_84[gal_id] = quants[1][2]

                fit_Av_50[gal_id] = quants[2][0]
                fit_Av_16[gal_id] = quants[2][1]
                fit_Av_84[gal_id] = quants[2][2]

                fit_logZsol_50[gal_id] = quants[3][0]
                fit_logZsol_16[gal_id] = quants[3][1]
                fit_logZsol_84[gal_id] = quants[3][2]

                fit_zfit_50[gal_id] = quants[4][0]
                fit_zfit_16[gal_id] = quants[4][1]
                fit_zfit_84[gal_id] = quants[4][2]

                # quants[5] is the SFH tuple percentiles array — shape (3, ncols)
                fit_logMt_50[gal_id] = quants[5][0][0]
                fit_logMt_16[gal_id] = quants[5][1][0]
                fit_logMt_84[gal_id] = quants[5][2][0]

                # original code assumed gp ordering; keep it for backward compatibility
                if quants[5].shape[1] > 1:
                    fit_logSFR100_50[gal_id] = quants[5][0][1]
                    fit_logSFR100_16[gal_id] = quants[5][1][1]
                    fit_logSFR100_84[gal_id] = quants[5][2][1]
                if quants[5].shape[1] > 2:
                    fit_nparam[gal_id] = quants[5][0][2]
                if quants[5].shape[1] > 3:
                    fit_t25_50[gal_id] = quants[5][0][3]
                    fit_t25_16[gal_id] = quants[5][1][3]
                    fit_t25_84[gal_id] = quants[5][2][3]
                if quants[5].shape[1] > 4:
                    fit_t50_50[gal_id] = quants[5][0][4]
                    fit_t50_16[gal_id] = quants[5][1][4]
                    fit_t50_84[gal_id] = quants[5][2][4]
                if quants[5].shape[1] > 5:
                    fit_t75_50[gal_id] = quants[5][0][5]
                    fit_t75_16[gal_id] = quants[5][1][5]
                    fit_t75_84[gal_id] = quants[5][2][5]

                fit_nbands[gal_id] = np.sum(gal_sed > 0)
                fit_f160w[gal_id] = cat_f160[gal_id]
                fit_stellarity[gal_id] = cat_class_star[gal_id]
                fit_chi2[gal_id] = np.amin(fit_likelihood)

                # flagging
                if np.isnan(quants[0][0]):
                    fit_flags[gal_id] = 1.0
                elif (np.abs(fit_logSFRinst_84[gal_id] - fit_logSFRinst_16[gal_id]) > sfr_uncert_cutoff):
                    fit_flags[gal_id] = 2.0
                elif (cat_class_star[gal_id] > 0.5):
                    fit_flags[gal_id] = 3.0
                elif (fit_chi2[gal_id] > 1000):
                    fit_flags[gal_id] = 4.0
                else:
                    fit_flags[gal_id] = 0.0

        except Exception as e:
            print("couldn't fit with pool at z=", zval, "because", e)

        print('finishing that')
        pl.clf()
        pl.figure(figsize=(12,6))
        pl.hist(cat_zbest[cat_zbest>0], np.arange(0,6,z_bw), color='black', alpha=0.3)
        pl.hist(cat_zbest[fit_zfit_50>0], np.arange(0,6,z_bw), color='royalblue')
        pl.title('fit %.0f/%.0f galaxies' % (np.sum(fit_zfit_50>0), len(cat_zbest)))
        pl.xlabel('redshift');pl.ylabel('# galaxies')

        display.clear_output(wait=True)
        display.display(pl.gcf())

    pl.show()

    fit_mdict = {
        'ID':fit_id,
        'logM_50':fit_logM_50, 'logM_16':fit_logM_16,'logM_84':fit_logM_84,
        'logSFRinst_50':fit_logSFRinst_50, 'logSFRinst_16':fit_logSFRinst_16, 'logSFRinst_84':fit_logSFRinst_84,
        'logZsol_50':fit_logZsol_50, 'logZsol_16':fit_logZsol_16, 'logZsol_84':fit_logZsol_84,
        'Av_50':fit_Av_50, 'Av_16':fit_Av_16, 'Av_84':fit_Av_84,
        'zfit_50':fit_zfit_50, 'zfit_16':fit_zfit_16, 'zfit_84':fit_zfit_84,
        'logMt_50':fit_logMt_50, 'logMt_16':fit_logMt_16, 'logMt_84':fit_logMt_84,
        'logSFR100_50':fit_logSFR100_50, 'logSFR100_16':fit_logSFR100_16, 'logSFR100_84':fit_logSFR100_84,
        't25_50':fit_t25_50, 't25_16':fit_t25_16, 't25_84':fit_t25_84,
        't50_50':fit_t50_50, 't50_16':fit_t50_16, 't50_84':fit_t50_84,
        't75_50':fit_t75_50, 't75_16':fit_t75_16, 't75_84':fit_t75_84,
        'nparam':fit_nparam,
        'nbands':fit_nbands,
        'F160w':fit_f160w,
        'stellarity':fit_stellarity,
        'chi2': fit_chi2,
        'fit_flags':fit_flags
    }

    fit_cat = Table(fit_mdict)
    fit_cat.write(output_fname, format='ascii.commented_header')

    return
