# script plotting all relevant master thesis plots

import visualisation as vs
import numpy as np
import config as cfg
import matplotlib.pyplot as plt


cfg.set_train_args()

part_argo = "part_18"
part_full = "part_19"
resume_argo = 565000
resume_full = 550000
plotting = "argo"

if plotting == "argo":
    # Argo plot masked assimilation
    vs.plot_val_error(
        part_argo, 600000, interval=5000, combine_start=80, in_channels=20
    )
    vs.timeseries_plotting(
        part_argo, resume_argo, obs=False, del_t=1, argo="argo", mask_argo="anhang"
    )
    vs.timeseries_plotting(
        part_argo, resume_argo, obs=False, del_t=12, argo="argo", mask_argo="anhang"
    )
    vs.hc_plotting(part_argo, resume_argo, time=[180, 192], obs=False)
    vs.hc_plotting(part_argo, resume_argo, time=[178, 180], obs=False)
    vs.pattern_corr_plot(part_argo, del_t=1, obs=False)
    vs.pattern_corr_plot(part_argo, del_t=12, obs=False)
elif plotting == "full":
    # Full masked assimilation plots
    # vs.plot_val_error(
    #    part_full, resume_full, interval=25000, combine_start=60, in_channels=20
    # )
    vs.timeseries_plotting(
        part_full, resume_full, obs=False, del_t=1, argo="full", mask_argo="anhang"
    )
    vs.timeseries_plotting(
        part_full, resume_full, obs=False, del_t=12, argo="full", mask_argo="anhang"
    )
    vs.hc_plotting(part_full, resume_full, time=[0, 120], obs=False)
    vs.hc_plotting(part_full, resume_full, time=[720, 732], obs=False)
    vs.pattern_corr_plot(part_full, del_t=1, obs=False)
    vs.pattern_corr_plot(part_full, del_t=12, obs=False)

    # Full direct reconstructions
    vs.timeseries_plotting(
        part_full, resume_full, obs=True, del_t=1, argo="full", mask_argo="full"
    )
    vs.timeseries_plotting(
        part_full, resume_full, obs=True, del_t=12, argo="full", mask_argo="full"
    )
    vs.hc_plotting(part_full, resume_full, time=[0, 120], obs=True)
    vs.hc_plotting(part_full, resume_full, time=[720, 732], obs=True)
    vs.pattern_corr_plot(part_full, del_t=1, obs=True)
    vs.pattern_corr_plot(part_full, del_t=12, obs=True)

elif plotting == "anhang":
    # Argo direct Anhang reconstructions
    vs.timeseries_plotting(
        part_argo, resume_argo, obs=False, del_t=1, argo="anhang", mask_argo="anhang"
    )
    vs.timeseries_plotting(
        part_argo, resume_argo, obs=False, del_t=12, argo="anhang", mask_argo="anhang"
    )
    vs.hc_plotting(part_argo, resume_argo, time=[202, 214], obs=False)
    vs.hc_plotting(part_argo, resume_argo, time=[202, 214], obs=True)
    vs.pattern_corr_plot(part_argo, del_t=1, obs=True)
    vs.pattern_corr_plot(part_argo, del_t=12, obs=True)

elif plotting == "all":
    # Argo plot masked assimilation
    vs.plot_val_error(
        part_argo, 600000, interval=5000, combine_start=80, in_channels=20
    )
    vs.timeseries_plotting(
        part_argo, resume_argo, obs=False, del_t=1, argo="argo", mask_argo="argo"
    )
    vs.timeseries_plotting(
        part_argo, resume_argo, obs=False, del_t=12, argo="argo", mask_argo="argo"
    )
    vs.hc_plotting(part_argo, resume_argo, time=[180, 192], obs=False)
    vs.hc_plotting(part_argo, resume_argo, time=[178, 180], obs=False)
    vs.pattern_corr_plot(part_argo, del_t=1, obs=False)
    vs.pattern_corr_plot(part_argo, del_t=12, obs=False)

    # Full masked assimilation plots
    vs.plot_val_error(
        part_full, 1000000, interval=25000, combine_start=60, in_channels=20
    )
    vs.timeseries_plotting(
        part_full, resume_full, obs=False, del_t=1, argo="full", mask_argo="full"
    )
    vs.timeseries_plotting(
        part_full, resume_full, obs=False, del_t=12, argo="full", mask_argo="full"
    )
    vs.hc_plotting(part_full, resume_full, time=[0, 120], obs=False)
    vs.hc_plotting(part_full, resume_full, time=[720, 732], obs=False)
    vs.pattern_corr_plot(part_full, del_t=1, obs=False)
    vs.pattern_corr_plot(part_full, del_t=12, obs=False)

    # Full direct reconstructions
    vs.timeseries_plotting(
        part_full, resume_full, obs=True, del_t=1, argo="full", mask_argo="full"
    )
    vs.timeseries_plotting(
        part_full, resume_full, obs=True, del_t=12, argo="full", mask_argo="full"
    )
    vs.hc_plotting(part_full, resume_full, time=[0, 120], obs=True)
    vs.hc_plotting(part_full, resume_full, time=[720, 732], obs=True)
    vs.pattern_corr_plot(part_full, del_t=1, obs=True)
    vs.pattern_corr_plot(part_full, del_t=12, obs=True)

    # Argo direct Anhang reconstructions
    vs.timeseries_plotting(
        part_argo, resume_argo, obs=False, del_t=1, argo="anhang", mask_argo="anhang"
    )
    vs.timeseries_plotting(
        part_argo, resume_argo, obs=False, del_t=12, argo="anhang", mask_argo="anhang"
    )
    vs.hc_plotting(part_argo, resume_argo, time=[202, 214], obs=False)
    vs.hc_plotting(part_argo, resume_argo, time=[202, 214], obs=True)
    vs.pattern_corr_plot(part_argo, del_t=1, obs=True)
    vs.pattern_corr_plot(part_argo, del_t=12, obs=True)
