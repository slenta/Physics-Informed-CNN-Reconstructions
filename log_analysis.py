# file to look at and analyze network logs and plot training error
import numpy as np
import config as cfg
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import evaluation_og as evalu


def trainings_error_plot(part=cfg.save_part, smoothing=50, mode="Training"):

    error_path = f"../Asi_maskiert/pdfs/validation/{part}/errors/"

    if mode == "Training":
        file_valid = f"{error_path}run-.-tag-loss_valid.csv"
        file_hole = f"{error_path}run-.-tag-loss_hole.csv"
    elif mode == "Validation":
        file_valid = f"{error_path}run-.-tag-loss_valid.csv"
        file_hole = f"{error_path}run-.-tag-loss_hole.csv"

    pd_valid = pd.read_csv(file_valid)
    pd_hole = pd.read_csv(file_hole)

    error_valid = evalu.running_mean_std(
        np.array(pd_valid["Value"]), mode="mean", del_t=smoothing
    )
    error_hole = evalu.running_mean_std(
        np.array(pd_hole["Value"]), mode="mean", del_t=smoothing
    )
    length = error_hole.shape[0]
    steps = pd_hole["Step"][:length]

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(f"Development of {mode} Loss")
    plt.subplot(1, 2, 1)
    plt.title(f"{mode} Loss Valid")
    plt.plot(steps, error_valid, label=f"{mode} Loss Valid")
    plt.legend()
    plt.grid()
    plt.xlabel("Trainings Iteration")
    plt.ylabel("Loss Values")
    plt.subplot(1, 2, 2)
    plt.title(f"{mode} Loss Hole")
    plt.plot(steps, error_hole, label=f"{mode} Loss Hole")
    plt.legend()
    plt.grid()
    plt.xlabel("Trainings Iteration")
    plt.ylabel("Loss Values")
    plt.savefig(f"{error_path}{mode}_loss.pdf")
    plt.show()


trainings_error_plot("part_19")
