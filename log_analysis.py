# file to look at and analyze network logs and plot training error
import numpy as np
import config as cfg
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import evaluation_og as evalu


def trainings_error_plot(part=cfg.save_part, depth=0, smoothing=50, mode="Training"):

    error_path = f"../Asi_maskiert/logs/{part}/"

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
    steps = pd_hole["Step"]
    step = np.linspace(0, max(steps), length)

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(f"Development of {mode} Loss: Depth Layer {depth}")
    plt.subplot(1, 2, 1)
    plt.title(f"{mode} Loss Valid")
    plt.plot(step, error_valid, label=f"{mode} Loss Valid")
    plt.legend()
    plt.grid()
    plt.xlabel("Trainings Iteration")
    plt.ylabel("Loss Values")
    plt.subplot(1, 2, 2)
    plt.title(f"{mode} Loss Hole")
    plt.plot(step, error_hole, label=f"{mode} Loss Hole")
    plt.legend()
    plt.grid()
    plt.xlabel("Trainings Iteration")
    plt.ylabel("Loss Values")
    plt.savefig(f"{error_path}{mode}_smoothing_{smoothing}_loss.pdf")
    plt.show()

    return error_hole, error_valid, steps


matplotlib.use("Agg")
length = 1000
overall_loss_hole = np.zeros(shape=(20, length))
overall_loss_valid = np.zeros(shape=(20, length))
smoothing = 1

for i in np.arange(80, 100):
    part = "part_" + str(i)
    depth = i - 80
    error_hole, error_valid, steps = trainings_error_plot(
        part, depth=depth, smoothing=smoothing
    )

    overall_loss_hole[depth, :] = error_hole
    overall_loss_valid[depth, :] = error_valid

loss_hole_mean = np.nanmean(overall_loss_hole, axis=0)
loss_valid_mean = np.nanmean(overall_loss_valid, axis=0)

print(
    steps[np.where(loss_hole_mean == min(loss_hole_mean))[0]],
    steps[np.where(loss_valid_mean == min(loss_valid_mean))[0]],
)


step = np.linspace(0, max(steps), length)
fig = plt.figure(figsize=(14, 6))
fig.suptitle(f"Development of Overall Training Loss")
plt.subplot(1, 2, 1)
plt.title(f"Training Loss Valid")
plt.plot(step, loss_valid_mean, label=f"Training Loss Valid")
plt.legend()
plt.grid()
plt.xlabel("Trainings Iteration")
plt.ylabel("Loss Values")
plt.subplot(1, 2, 2)
plt.title(f"Training Loss Hole")
plt.plot(step, loss_hole_mean, label=f"Training Loss Hole")
plt.legend()
plt.grid()
plt.xlabel("Trainings Iteration")
plt.ylabel("Loss Values")
plt.savefig(
    f"../Asi_maskiert/pdfs/validation/part_18/Training_loss_overall_smoothing_{smoothing}.pdf"
)
plt.show()
