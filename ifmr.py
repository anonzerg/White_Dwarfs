#! /usr/bin/env python

import glob
import logging

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from warnings import filterwarnings

plt.rcParams.update({"font.family": "IBM Plex Mono"})
filterwarnings("ignore")

logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

def main() -> None:
    csv_files = glob.glob("./result/masses/*.csv")
    df_list = [pd.read_csv(file) for file in csv_files]
    df_comb = pd.concat(df_list, ignore_index=True)
    df_comb.to_csv("./result/masses/comb_masses.csv")

    plt.figure(figsize=(8, 8), dpi=300)
    plt.title(
        f"Initial Final Mass Relation",
        fontsize=18
    )
    sns.scatterplot(
        data=df_comb,
        x=df_comb["prog_mass"],
        y=df_comb["final_mass"],
        color="orange",
        edgecolor="black",
        linewidth=1.0,
    )
    plt.xlabel("Initial Mass", fontsize=18)
    plt.ylabel("Final Mass", fontsize=18)
    plt.grid()
    plt.tight_layout()
    plt.savefig("./plots/ifmr.png")
    plt.close()

    return None


if __name__ == "__main__":
    main()
