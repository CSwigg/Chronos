#!/usr/bin/env python3
"""
Multiprocessing version of the Chronos cluster analysis notebook.
Processes multiple stellar clusters in parallel using multiprocessing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import arviz as az
import multiprocessing as mp
from functools import partial
import os
import time
from datetime import datetime
from tqdm import tqdm
import warnings
import psutil  # For memory monitoring

# Suppress warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
plt.ioff()  # Turn off interactive plotting

# Add Chronos to path
sys.path.append('/Users/cam/Desktop/astro_research/radcliffe/chronos/Chronos/')
from utils.ExtinctionPrior import ExtinctionPrior
from bayes_fitting.ChronosSkewedCauchy_bayes import ChronosSkewCauchyBayes


def check_existing_results(output_file, all_results_file, cluster_ids):
    """
    Check existing results and return clusters that still need to be processed.

    Parameters:
    -----------
    output_file : str
        Path to the main output CSV file
    all_results_file : str
        Path to the all results CSV file
    cluster_ids : list
        List of all cluster IDs to process

    Returns:
    --------
    tuple
        (clusters_to_process, existing_results_df, existing_all_results_df)
    """
    existing_results_df = pd.DataFrame()
    existing_all_results_df = pd.DataFrame()

    # Check if results files exist
    if os.path.exists(all_results_file):
        try:
            existing_all_results_df = pd.read_csv(all_results_file)
            completed_clusters = set(existing_all_results_df['name'].tolist())
            print(f"Found existing results for {len(completed_clusters)} clusters")
        except Exception as e:
            print(f"Warning: Could not read existing results file: {e}")
            completed_clusters = set()
    else:
        completed_clusters = set()

    if os.path.exists(output_file):
        try:
            existing_results_df = pd.read_csv(output_file)
        except Exception as e:
            print(f"Warning: Could not read existing output file: {e}")

    # Determine which clusters still need processing
    all_clusters = set(cluster_ids)
    clusters_to_process = list(all_clusters - completed_clusters)

    if len(completed_clusters) > 0:
        print("Resuming analysis:")
        print(f"  Total clusters: {len(all_clusters)}")
        print(f"  Already completed: {len(completed_clusters)}")
        print(f"  Remaining to process: {len(clusters_to_process)}")
    else:
        print(f"Starting fresh analysis with {len(clusters_to_process)} clusters")

    return clusters_to_process, existing_results_df, existing_all_results_df


def save_results_batch(results_batch, df_clusters, output_file, all_results_file):
    """
    Save a batch of results to files efficiently.

    Parameters:
    -----------
    results_batch : list
        List of result dictionaries from processing clusters
    df_clusters : pd.DataFrame
        Original cluster data for merging
    output_file : str
        Path to main output file
    all_results_file : str
        Path to all results file
    """
    try:
        if not results_batch:
            return

        print(f"Saving batch of {len(results_batch)} results...")

        # Save to all results file
        if os.path.exists(all_results_file):
            all_results_df = pd.read_csv(all_results_file)
            # Remove test entries
            all_results_df = all_results_df[all_results_df['name'] != 'TEST_ENTRY']
        else:
            all_results_df = pd.DataFrame()

        # Add new results
        new_results_df = pd.DataFrame(results_batch)
        if len(all_results_df) > 0:
            # Remove any duplicates (in case of rerun)
            new_names = set(new_results_df['name'].tolist())
            all_results_df = all_results_df[~all_results_df['name'].isin(new_names)]
            all_results_df = pd.concat([all_results_df, new_results_df], ignore_index=True)
        else:
            all_results_df = new_results_df

        # Save all results
        all_results_df.to_csv(all_results_file, index=False)
        print(f"Saved all results to: {all_results_file}")

        # Save successful results with cluster data
        successful_results = [r for r in results_batch if r['status'] == 'success']
        if successful_results:
            print(f"Processing {len(successful_results)} successful results for main output...")

            # Load existing successful results
            if os.path.exists(output_file):
                existing_df = pd.read_csv(output_file)
                # Remove test entries
                existing_df = existing_df[existing_df['name'] != 'TEST_ENTRY']
            else:
                existing_df = pd.DataFrame()

            # Create DataFrame from successful results
            successful_df = pd.DataFrame(successful_results).drop('status', axis=1)

            # Get cluster data for merging (handle missing data_source column)
            if 'data_source' in df_clusters.columns:
                cluster_data = df_clusters[df_clusters['data_source'] == 'hunt'].copy()
            else:
                cluster_data = df_clusters.copy()

            # Filter to only clusters we have results for
            cluster_data = cluster_data[cluster_data['name'].isin(successful_df['name'])]

            # Merge cluster data with chronos results
            merged_df = pd.merge(cluster_data, successful_df, on='name', how='left')

            if len(existing_df) > 0:
                # Remove any existing entries for these clusters and combine
                existing_df = existing_df[~existing_df['name'].isin(merged_df['name'])]
                final_df = pd.concat([existing_df, merged_df], ignore_index=True)
            else:
                final_df = merged_df

            final_df.to_csv(output_file, index=False)
            print(f"Saved {len(successful_results)} successful results to: {output_file}")

        print("Batch save complete!")

    except Exception as e:
        print(f"Error saving batch: {e}")
        import traceback
        traceback.print_exc()


def mode_reals(array, bins=100):
    """Calculate the mode of a real-valued array using histogram."""
    counts, bin_edges = np.histogram(array, bins=bins)
    # Take left edges as approximation for bin midpoint
    bins_left_edges = bin_edges[:-1]
    return bins_left_edges[np.argmax(counts)]


def process_cluster_simple(args):
    """
    Simple wrapper function that processes a cluster.
    This needs to be at module level for multiprocessing.
    """
    cluster_id, df, ext, output_dir_posterior, output_dir_isochrone = args

    print(f"Processing cluster: {cluster_id}")
    result = process_single_cluster(cluster_id, df, ext, output_dir_posterior, output_dir_isochrone)
    print(f"Finished cluster: {cluster_id} - Status: {result['status']}")

    return result


def process_single_cluster(cluster_id, data, ext, output_dir_posterior, output_dir_isochrone):
    """
    Process a single cluster with Chronos Bayesian fitting.

    Parameters:
    -----------
    cluster_id : str
        Unique identifier for the cluster
    data : pd.DataFrame
        Full dataset containing all clusters
    ext : ExtinctionPrior
        Extinction prior object
    output_dir_posterior : str
        Directory to save posterior plots
    output_dir_isochrone : str
        Directory to save isochrone plots

    Returns:
    --------
    dict
        Dictionary containing the fitted parameters for this cluster
    """
    import gc  # Garbage collection for memory management

    try:
        # Get cluster data
        df_group = data.groupby('label').get_group(cluster_id)
        extinction_prior = ext.compute_prior(df_group['ra'], df_group['dec'], distance=df_group['distance_50'])

        age_hunt_myr = df_group['age_myr'].values[0]

        # Set up Chronos fitting
        kwargs = dict(
            data=df_group,
            use_grp=False,
            models='parsec',
            abs_Gmag_name='g_abs_mag',
            color_bprp_name='bp-rp',
            color_grp_name='g-rp'
        )
        cbayes = ChronosSkewCauchyBayes(**kwargs)

        # Set parameter bounds
        #age_lo, age_hi = age_hunt_myr - 50, age_hunt_myr + 50
        age_lo, age_hi = 1, 500
        if age_lo < 1:
            age_lo = 1

        av_r = 0, 5.
        lA_r = np.log10(age_lo * 10**6), np.log10(age_hi * 10**6)
        bayes_bounds = dict(
            logAge_range=lA_r,
            #feh_range=(-3, 1e-3),
            feh_range = (-1e-5, 1e-5),
            av_range=av_r,
            skewness_range=(0.5, 0.99),
            scale_range=(0.001, 0.1)
        )

        # Set fitting kwargs and bounds
        cbayes.set_fitting_kwargs(fit_range=(-np.infty, 10), do_mass_normalize=False, weights=None)
        cbayes.set_bounds(**bayes_bounds)

        # Fit the model with reduced parameters to save memory
        sampler, best_fit, samples_bprp = cbayes.fit_bayesian(nwalkers=40, nsteps=400, burnin=100)

        # Get samples from posterior
        logAge, feh, A_V, skewness, scale = samples_bprp.T
        to_plot = 10**logAge / 10**6, A_V, skewness, scale
        names = 'Age (Myr)', 'AV (mag)', 'Skewness', 'Scale'

        # Create posterior plots
        fig, axes = plt.subplots(1, len(names), figsize=(len(names) * 4, 5))
        for name, data2plot, ax in zip(names, to_plot, axes):
            ax.hist(data2plot, bins=50, histtype='step', color='k')
            ax.hist(data2plot, bins=50, histtype='stepfilled', color='k', alpha=0.25)
            mode_hist = mode_reals(data2plot, bins=100)
            lo, hi = az.hdi(data2plot, hdi_prob=0.64)
            for al in [mode_hist, lo, hi]:
                ax.axvline(al, c='k', alpha=0.5)
            ax.set_xlabel(name)

        posterior_file = os.path.join(output_dir_posterior, f'{cluster_id}_fit.png')
        plt.savefig(posterior_file, bbox_inches='tight', dpi=300)
        plt.close()
        plt.clf()

        # Calculate final parameters
        hdi_prob = 0.68
        ages_myr = 10**logAge / 10**6
        age_mode = mode_reals(ages_myr, bins=100)
        age_hdi = az.hdi(ages_myr, hdi_prob=hdi_prob)
        age_lo, age_hi = age_hdi[0], age_hdi[1]

        av_mode = mode_reals(A_V, bins=100)
        av_lo, av_hi = az.hdi(A_V, hdi_prob=hdi_prob)

        # Compute fit info
        _, masses, _ = cbayes.compute_fit_info(
            logAge=np.log10(age_mode * 10**6), feh=0, A_V=av_mode, g_rp=cbayes.use_grp, signed_distance=True
        )

        # Create isochrone plot
        size = 15
        plt.figure(figsize=(6, 9))
        plt.scatter(*cbayes.distance_handler.fit_data['hrd'].T, s=50, c='tab:purple',
                    edgecolors='tab:purple', alpha=0.9)
        plt.ylim(14, -4)
        plt.xlim(-1, 5)
        plt.xlabel(r'$G_{BP} - G_{RP}$', size=size)
        plt.ylabel(r'$M_G$', size=size)
        plt.xticks(size=size)
        plt.yticks(size=size)

        # Plot isochrone
        isochrone = cbayes.isochrone_handler.model(
            logAge=np.log10(age_mode * 10**6), feh=0, A_V=av_mode, g_rp=cbayes.use_grp
        )
        plt.plot(*isochrone.T,
                 label=r'${{{:.1f}}}^{{+{:.1f}}}_{{{:.1f}}}$ Myr'.format(
                     age_mode, age_hi - age_mode, age_lo - age_mode),
                 c='k', alpha=0.7, zorder=0)
        plt.title(r'AV = ${{{:.1f}}}^{{{:.1f}}}_{{{:.1f}}}$ mag'.format(
            av_mode, av_hi - av_mode, av_lo - av_mode), size=size)
        plt.annotate(r'${{{:.1f}}}^{{+{:.1f}}}_{{{:.1f}}}$ Myr'.format(
            age_mode, age_hi - age_mode, age_lo - age_mode),
            (0.98, 0.98), xycoords='axes fraction', ha='right', va='top', size=size)

        isochrone_file = os.path.join(output_dir_isochrone, f'{cluster_id}.png')
        plt.savefig(isochrone_file, bbox_inches='tight', dpi=300)
        plt.close()
        plt.clf()

        # Force garbage collection to free memory
        gc.collect()

        # Return results
        result = {
            'name': cluster_id,
            'age_chronos_mode': age_mode,
            'age_chronos_lo': age_lo,
            'age_chronos_hi': age_hi,
            'av_chronos_mode': av_mode,
            'av_chronos_lo': av_lo,
            'av_chronos_hi': av_hi,
            'status': 'success'
        }

        return result

    except Exception as e:
        return {
            'name': cluster_id,
            'age_chronos_mode': np.nan,
            'age_chronos_lo': np.nan,
            'age_chronos_hi': np.nan,
            'av_chronos_mode': np.nan,
            'av_chronos_lo': np.nan,
            'av_chronos_hi': np.nan,
            'status': f'error: {str(e)}'
        }


def main():
    """Main function to run the multiprocessing cluster analysis."""
    print("Starting Chronos multiprocessing cluster analysis...")
    print(f"Start time: {datetime.now()}")

    # Define output files first
    output_file = '/Users/cam/Downloads/hunt_sample_chronos_ages_multiprocessing_feb_2026.csv'
    all_results_file = '/Users/cam/Downloads/all_clusters_chronos_results_feb_2026.csv'

    # Load data
    print("Loading data...")
    df_stars = pd.read_csv('/Users/cam/Downloads/members-2.csv')
    #df_clusters = pd.read_csv('/Users/cam/Downloads/hunt_new_vels.csv')
    df_clusters = pd.read_csv('/Users/cam/Downloads/hunt_partII_partIII_merged.csv')

    #### Make cuts
    df_clusters = df_clusters.loc[
        # (df_clusters['x'].between(-2000, 2000)) &
        # (df_clusters['y'].between(-2000, 2000)) &
        # (df_clusters['z'].between(-2000, 2000)) &
        (df_clusters['age_myr'] < 200)
        # (df_clusters['class_50'] > 0.5) &
        # (df_clusters['cst'] > 5)
    ]
    print(f"Number of clusters after cuts: {len(df_clusters)}")

    # Sort clusters by age (youngest first)
    df_clusters = df_clusters.sort_values(by='age_myr', ascending=True).reset_index(drop=True)

    # Process stellar data
    df_stars = df_stars.loc[df_stars['name'].isin(df_clusters['name'])]
    df_stars['g_abs_mag'] = df_stars['phot_g_mean_mag'] + 5 - 5 * np.log10(1000 / df_stars['parallax'])
    df = df_stars[['name', 'ra', 'dec', 'parallax', 'phot_g_mean_mag', 'phot_bp_mean_mag',
                   'phot_rp_mean_mag', 'g_abs_mag', 'bp_rp', 'g_rp']]
    df = df.rename(columns={'bp_rp': 'bp-rp', 'g_rp': 'g-rp'})

    # Handle distance column - try distance_50 first, fall back to parallax
    if 'distance_50' in df_stars.columns:
        df['distance_50'] = df_stars['distance_50']
        print("Using 'distance_50' column for distances")
    else:
        df['distance_50'] = 1000 / df['parallax']
        print("Warning: 'distance_50' not found, using parallax-derived distances")

    df['label'] = df['name']

    # Merge age_myr from df_clusters into df for processing
    df = pd.merge(df, df_clusters[['name', 'age_myr']], on='name', how='left')

    # Add result columns
    df['age_lo'] = np.nan
    df['age_hi'] = np.nan
    df['av'] = np.nan
    df['av_lo'] = np.nan
    df['av_hi'] = np.nan

    # Initialize extinction prior
    print("Loading extinction prior...")
    fname_ext = '/Users/cam/Downloads/mean_and_std_healpix.fits'
    ext = ExtinctionPrior(fname_ext)

    # Create output directories
    output_dir_posterior = '/Users/cam/Desktop/astro_research/radcliffe/chronos/Chronos/posterior_plots_feb_2026'
    output_dir_isochrone = '/Users/cam/Desktop/astro_research/radcliffe/chronos/Chronos/isochrone_fit_plots_feb_2026'

    os.makedirs(output_dir_posterior, exist_ok=True)
    os.makedirs(output_dir_isochrone, exist_ok=True)

    # Get all cluster IDs and check what's already been processed
    all_cluster_ids = df.label.unique()
    clusters_to_process, existing_results_df, existing_all_results_df = check_existing_results(
        output_file, all_results_file, all_cluster_ids
    )

    if len(clusters_to_process) == 0:
        print("All clusters have already been processed!")
        print(f"Results are available in: {output_file}")
        return

    # Set up multiprocessing with safety limits
    n_processes = mp.cpu_count()
    print(f"Using {n_processes} processes")

    # Add memory management
    import psutil
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    print(f"Available memory: {available_memory_gb:.1f} GB")

    # Prepare arguments for multiprocessing
    process_args = [
        (cluster_id, df, ext, output_dir_posterior, output_dir_isochrone)
        for cluster_id in clusters_to_process
    ]

    print("Starting cluster processing with batch saving every 5 clusters...")

    # Process clusters in batches
    batch_size = 5
    all_results = []

    # Process clusters in parallel
    start_time = time.time()

    try:
        with mp.Pool(processes=n_processes) as pool:
            for i, result in enumerate(tqdm(
                pool.imap(process_cluster_simple, process_args, chunksize=1),
                total=len(clusters_to_process),
                desc="Processing clusters",
                unit="cluster"
            )):
                all_results.append(result)

                # Save in batches of 5
                if (i + 1) % batch_size == 0:
                    batch_to_save = all_results[-batch_size:]
                    save_results_batch(batch_to_save, df_clusters, output_file, all_results_file)

                # Also save if we've reached the end
                elif (i + 1) == len(clusters_to_process):
                    remaining_batch = all_results[-(len(all_results) % batch_size):]
                    if remaining_batch:
                        save_results_batch(remaining_batch, df_clusters, output_file, all_results_file)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Saving current results...")
        if all_results:
            # Save whatever we have processed so far
            unsaved_start = len(all_results) - (len(all_results) % batch_size)
            if unsaved_start < len(all_results):
                remaining_batch = all_results[unsaved_start:]
                save_results_batch(remaining_batch, df_clusters, output_file, all_results_file)
            print(f"Saved {len(all_results)} processed results before exit")
    except Exception as e:
        print(f"\nError during processing: {e}")
        if all_results:
            save_results_batch(all_results, df_clusters, output_file, all_results_file)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nProcessing complete! Total time: {total_time:.2f} seconds")

    # Optional: create summary
    try:
        if all_results:
            results_df = pd.DataFrame(all_results)
            successful_results = results_df[results_df['status'] == 'success']
            print(f"Summary: {len(successful_results)} clusters processed successfully out of {len(all_results)} total")

            if len(successful_results) > 0:
                print(f"Age range: {successful_results['age_chronos_mode'].min():.1f} - {successful_results['age_chronos_mode'].max():.1f} Myr")
                print(f"Extinction range: {successful_results['av_chronos_mode'].min():.3f} - {successful_results['av_chronos_mode'].max():.3f}")
        else:
            print("No results to summarize")
    except Exception as e:
        print(f"Could not create summary: {e}")

    print("Results saved to:")
    print(f"   - Main results: {output_file}")
    print(f"   - All results: {all_results_file}")
    print(f"   - Posterior plots: {output_dir_posterior}")
    print(f"   - Isochrone plots: {output_dir_isochrone}")

    # Force garbage collection
    import gc
    gc.collect()

    print("All done!")


if __name__ == "__main__":
    # Additional warning suppression for multiprocessing
    import logging
    logging.getLogger().setLevel(logging.ERROR)

    main()
