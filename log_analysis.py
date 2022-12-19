# file to look at and analyze network logs and plot training error
import numpy as np
import config as cfg
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd


def trainings_error_plot(part=cfg.save_part, mode="Training"):
    file_valid = f"../Asi_maskiert/pdfs/validation/{part}/run-.-tag-loss_valid.csv"
    file_hole = f"../Asi_maskiert/pdfs/validation/{part}/run-.-tag-loss_hole.csv"

    pd_valid = pd.read_csv(file_valid)
    pd_hole = pd.read_csv(file_hole)

    print(pd_hole, pd_valid)


trainings_error_plot("part_19")
