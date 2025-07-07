#! /usr/bin/env python

import asteca
import os
import sys
import pyabc
import logging
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path
from tempfile import gettempdir
from warnings import filterwarnings
from WDPhotTools.fitter import WDfitter
from astroquery.gaia import Gaia
# from WDPhotTools import plotter

plt.rcParams.update({"font.family": "IBM Plex Mono"})
filterwarnings("ignore")

logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

# TODO: create separate modules for each section
#       plots
#       isochrone fitting
#       cooling model fitting

# phot cut parameters
LOGG_MIN, LOGG_MAX = 7.0, 10.0

# ASteCA parameters
# TODO: need fixing for different isochrones
LOG_AGE_MIN, LOG_AGE_MAX = 7.0, 9.5
MET_MIN, MET_MAX = 0.01, 0.02
DIST_MOD_MIN, DIST_MOD_MAX = 0.8, 10.5
EXTINC_AV_MIN, EXTINC_AV_MAX = 0.0, 2.0

WD_ATMOSPHERE = "H"
ERROR_FACTOR = 2.5 / np.log(10)
POP_SIZE = 100  # popluation size for ASteCA ABCSMC


def main():
    if len(sys.argv) != 2:
        logger.error("provide scv file of cluster memebers!")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    try:
        data = pd.read_csv(input_file, on_bad_lines="skip")
        logger.info(f"successfully loaded data from {input_file}.")
        logger.info(f"found {len(data)} sources in {input_file}.")
    except FileNotFoundError:
        logger.error(f"{input_file} does not exist!")
        sys.exit(1)
    except Exception:
        logger.error(f"loading {input_file} failed!", exc_info=True)
        sys.exit(1)

    source_id = data["GaiaDR3"].astype(int).tolist()
    source_id_str = ",".join(str(each_id) for each_id in source_id)
    query = f"""SELECT
      source_id,
      phot_g_mean_flux_over_error,
      phot_bp_mean_flux_over_error,
      phot_rp_mean_flux_over_error,
      ag_gspphot,
      ag_gspphot_lower,
      ag_gspphot_upper
    FROM gaiadr3.gaia_source
    WHERE source_id IN ({source_id_str})"""

    # request data from Gaia
    try:
        job = Gaia.launch_job_async(query=query, verbose=False)
        table = job.get_results()
        extra = table.to_pandas()
    except Exception:
        logger.error("failed to get extra tables from Gaia", exc_info=True)
        sys.exit(1)

    data["Gabs"] = data["Gmag"] + 5.0 * np.log10(data["Plx"] / 1000) + 5.0

    data["Dist"] = 1000 / data["Plx"]
    logger.info(f"mean distance {np.mean(data["Dist"]):.4f} pc")

    g_mag_err = ERROR_FACTOR * np.reciprocal(
        extra["phot_g_mean_flux_over_error"].to_numpy()
    )
    bp_mag_err = ERROR_FACTOR * np.reciprocal(
        extra["phot_bp_mean_flux_over_error"].to_numpy()
    )
    rp_mag_err = ERROR_FACTOR * np.reciprocal(
        extra["phot_rp_mean_flux_over_error"].to_numpy()
    )

    z_min = extra["ag_gspphot_lower"].to_numpy()
    z_max = extra["ag_gspphot_upper"].to_numpy()

    # NOTE: begining of ASteCA analysis
    cluster = asteca.Cluster(
        ra=data["RA_ICRS"],
        dec=data["DE_ICRS"],
        pmra=data["pmRA"],
        pmde=data["pmDE"],
        plx=data["Plx"],
        e_pmra=data["e_pmRA"],
        e_pmde=data["e_pmDE"],
        e_plx=data["e_Plx"],
        magnitude=data["Gmag"],
        e_mag=g_mag_err,
        color=data["BP-RP"],
        e_color=np.sqrt(bp_mag_err**2 + rp_mag_err**2)
    )

    cluster.get_center()
    ra_c, dec_c = cluster.radec_c
    cluster.get_nmembers()
    membership = asteca.Membership(cluster)

    prob = 0.8
    prob_fastmp = membership.fastmp()
    data["Prob_Fastmp"] = prob_fastmp
    prob_mask = data["Prob_Fastmp"] > prob

    directories = ["prob", "cmd", "ifmr", "loc"]
    for directory in directories:
        try:
            dir = Path(f"plots/{directory}")
            dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.error(f"failed to create {dir.stem}!", exc_info=True)
            sys.exit(1)

    # membership prob plots
    plt.figure(figsize=(8, 8), dpi=300)
    plt.title(
        f"{input_file.stem} Membership Probabilities {prob}",
        fontsize=18
    )
    sns.scatterplot(
        data=data,
        x=data["BP-RP"],
        y=data["Gmag"],
        c="gray",
        ec=None,
        alpha=0.25,
        label=f"prob < {prob}"
    )
    sns.scatterplot(
        data=data[prob_mask],
        x=data["BP-RP"][prob_mask],
        y=data["Gmag"][prob_mask],
        hue=prob_fastmp[prob_mask],
        ec="k",
        alpha=0.6,
        lw=1,
        palette="viridis",
    )
    plt.gca().invert_yaxis()
    plt.xlabel("BP-RP", fontsize=16)
    plt.ylabel("Gmag", fontsize=16)
    plt.legend(fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"./plots/prob/prob_{input_file.stem}.png")
    plt.close()

    # NOTE: isochrone fit
    isochrones = asteca.Isochrones(
        model="PARSEC",
        isochs_path=f"isochrones/",
        magnitude="Gmag",
        color=("G_BPmag", "G_RPmag"),
        magnitude_effl=6390.21,
        color_effl=(5182.58, 7825.08),
        verbose=3
    )

    synthetic_cluster = asteca.Synthetic(isochrones, seed=457304)
    synthetic_cluster.calibrate(cluster)
    likelihood = asteca.Likelihood(cluster)

    # WARN: do not edit key names for this dict
    #       it fucks up the isochrone fitting
    priors = pyabc.Distribution(
        {
            "met": pyabc.RV("uniform", MET_MIN, MET_MAX-MET_MIN),
            "loga": pyabc.RV("uniform", LOG_AGE_MIN, LOG_AGE_MAX-LOG_AGE_MIN),
            "dm": pyabc.RV("uniform", DIST_MOD_MIN, DIST_MOD_MAX-DIST_MOD_MIN),
            "Av": pyabc.RV("uniform", EXTINC_AV_MIN, EXTINC_AV_MAX-EXTINC_AV_MIN)
        }
    )

    # helper functons for pyabc
    def model(parameters):
        return {"data": synthetic_cluster.generate(parameters)}

    def distance(synthetic_cluster_dict, _):
        return likelihood.get(synthetic_cluster_dict["data"])

    abc = pyabc.ABCSMC(
        model,
        priors,
        distance,
        population_size=POP_SIZE,
    )

    timestamp = time.strftime("%H:%M:%S", time.localtime())
    db_path = "sqlite:///" + os.path.join(
        gettempdir(),
        f"{input_file.stem}-{timestamp}.db"
    )

    # TODO: naming databases by timestamp -> Done
    #       add a cleanup routine
    #       see if it is possible to reuse databases
    abc.new(db_path)
    abc_history = abc.run(minimum_epsilon=0.01, max_nr_populations=20)

    df, w = abc_history.get_distribution()
    parameters = {}
    for k in df.keys():
        _median = pyabc.weighted_statistics.weighted_median(df[k].values, w)
        parameters[k] = _median
        # _std = pyabc.weighted_statistics.weighted_std(df[k].values, w)

    # synthetic_cluster_arr = synthetic_cluster.generate(parameters)
    isochrone_arr = synthetic_cluster.get_isochrone(parameters)

    final_min_dist = pyabc.inference_util.eps_from_hist(abc_history)
    logger.info(f"final minimized distance: {100*final_min_dist:.0f}%")
    ess = pyabc.weighted_statistics.effective_sample_size(w)
    logger.info(f"effective sample size: {ess:.0f}")

    cluster_age = 10 ** parameters["loga"]
    logger.info(f"cluster age: {cluster_age:.0f} years")
    # NOTE: end of ASteCA analysis

    # Gentile 2019 photometric cuts
    candidates = data[
        (data["Gabs"] > 5.0) &
        (data["Gabs"] > 5.93 + 5.047*data["BP-RP"]) &
        (data["Gabs"] > 6.0*data["BP-RP"]**3
            - 21.77*data["BP-RP"]**2
            + 27.91*data["BP-RP"] + 0.897) &
        (data["BP-RP"] < 1.7)
    ]
    logger.info(
        f"selected {len(candidates)} potentinal WD from photometric cuts."
    )

    # fit cooling model
    wd_counter = 0
    identified_wds = np.zeros(len(data), dtype=int)
    wd_best_fits_list = []
    cooling_ages = []
    mass_fitted = []

    filters = ["G3", "G3_BP", "G3_RP"]
    fitter = WDfitter()

    for index, star in candidates.iterrows():
        mags = [star["Gmag"], star["BPmag"], star["RPmag"]]
        mag_errors = [g_mag_err[index], bp_mag_err[index], rp_mag_err[index]]
        try:
            fitter.set_extinction_mode(
                mode="total",
                z_min=z_min[index],
                z_max=z_max[index]
            )
            fitter.fit(
                atmosphere=WD_ATMOSPHERE,
                filters=filters,
                mags=mags,
                mag_errors=mag_errors,
                distance=data["Dist"][index],
                # ebv=0.0,
                rv=0.0,
                method="least_squares",
            )
            fitted_params = fitter.best_fit_params["H"]
            logg_fitted = fitted_params["logg"]
            mass_fitted.append(fitted_params["mass"])
            cooling_ages.append(fitted_params["age"])
            if LOGG_MIN < logg_fitted < LOGG_MAX:
                wd_counter += 1
                wd_best_fits_list.append(fitter.best_fit_params["H"].copy())
                identified_wds[index] = 1
                fitter.show_best_fit(
                    atmosphere="H",
                    title=f"{input_file.stem}_{index}",
                    display=False,
                    savefig=True,
                    folder=f"plots/{input_file.stem}",
                    filename=f"fit_{input_file.stem}_{index}",
                )
            plt.close()

        except Exception:
            logger.error("cooling model fit failed!", exc_info=True)

    if len(wd_best_fits_list) != 0:
        pd.DataFrame(
            wd_best_fits_list
        ).to_csv(f"./result/{input_file.stem}_fitted_model.csv")

    # progenitor stuff
    progenitor_age = cluster_age - np.array(cooling_ages)
    logger.info(f"progenitor age: {progenitor_age} years")
    progenitor_mass = (progenitor_age / (10**10)) ** 2.5
    logger.info(f"progenitor mass: {progenitor_mass}")
    logger.info(f"final mass: {mass_fitted}")

    pd.DataFrame({
        "prog_mass": progenitor_mass,
        "final_mass": wd_best_fits_list[0]["mass"]
    }).to_csv(f"./result/masses/{input_file.stem}_init_final_masses.csv")

    # ifmr plots
    plt.figure(figsize=(8, 8), dpi=300)
    plt.scatter(
        progenitor_mass,
        mass_fitted,
        color="orange",
        linewidth=1.0,
        edgecolor="black",
        # alpha=0.6,
        label="IFMR"
    )
    plt.xlabel("initial mass", fontsize=18)
    plt.ylabel("final mass", fontsize=18)
    plt.title(f"{input_file.stem}", fontsize=18)
    plt.grid()
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(f"./plots/ifmr/ifmr_{input_file.stem}.png")
    plt.close()

    data["Category"] = np.where(identified_wds == 1, "White Dwarfs", "Members")

    # cmd plots
    # fig , ax = plt.subplots(figsize=(8,8), dpi=300)
    plt.figure(figsize=(8, 8), dpi=300)
    # plotter.plot_atmosphere_model(
    #     x="G3_BP-G3_RP",
    #     y="G3",
    #     atmosphere="H",
    #     savefig=False
    # )
    sns.scatterplot(
        data=data,
        x="BP-RP",
        y="Gmag",
        hue="Category",
        marker="o",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.6
    )
    plt.plot(
        isochrone_arr[1],
        isochrone_arr[0],
        color="purple",
        linewidth=3.0,
        alpha=0.6,
        label="Isochrone Fit"
    )
    # ax.scatter(
    #     x=data["BP-RP"],
    #     y=data["Gmag"],
    #     c=identified_wds,
    #     cmap="plasma",
    #     marker="o",
    #     edgecolor="black",
    #     linewidth=1,
    #     alpha=0.6
    # )
    plt.xlabel("BP-RP", fontsize=18)
    plt.ylabel("Gmag", fontsize=18)
    plt.gca().invert_yaxis()
    plt.title(f"{input_file.stem}", fontsize=18)
    plt.grid()
    plt.legend(fontsize=18)
    plt.tight_layout()
    Path("plots/cmd").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"./plots/cmd/cmd_{input_file.stem}.png")
    plt.close()

    # loc plots
    mask = identified_wds == 1
    plt.figure(figsize=(8, 8), dpi=300)
    # sns.scatterplot(
    #     data=data,
    #     x="RA_ICRS",
    #     y="DE_ICRS",
    #     hue="Category",
    #     marker="o",
    #     edgecolor="black",
    #     linewidth=1,
    #     alpha=0.6
    # )
    sns.scatterplot(
        data=data[~mask],
        x=data["RA_ICRS"][~mask],
        y=data["DE_ICRS"][~mask],
        c="gray",
        ec=None,
        alpha=0.4,
        label="Members"
    )
    sns.scatterplot(
        data=data[mask],
        x=data["RA_ICRS"][mask],
        y=data["DE_ICRS"][mask],
        ec="k",
        lw=1,
        c="orange",
        alpha=0.6,
        label="White Dwarfs"
    )
    plt.scatter(
        ra_c,
        dec_c,
        marker="+",
        s=100,
        color="red",
        label="Cluster Center"
    )
    # plt.plot(
    #     isochrone_arr[1],
    #     isochrone_arr[0],
    #     color="purple",
    #     linewidth=3.0,
    #     alpha=0.6
    plt.xlabel("ra", fontsize=18)
    plt.ylabel("dec", fontsize=18)
    plt.title(f"{input_file.stem}", fontsize=18)
    plt.grid()
    plt.legend(fontsize=18)
    plt.tight_layout()
    Path("plots/cmd").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"./plots/loc/loc_{input_file.stem}.png")
    plt.close()

    num_found_wds = np.count_nonzero(identified_wds == 1)
    logger.info(f"number of WD found: {num_found_wds}.")

    return None


if __name__ == "__main__":
    main()

