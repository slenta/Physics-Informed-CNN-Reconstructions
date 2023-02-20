# script plotting all relevant master thesis plots

import visualisation as vs
import numpy as np
import config as cfg
import matplotlib.pyplot as plt


cfg.set_train_args()

part_argo = "part_18"
part_full = "part_19"
part_depth = "part_13"
resume_argo = 585000
resume_full = 550000
resume_depth = 1000000
plotting = "special"

if plotting == "special":

    vs.timeseries_plotting(
        part_full, resume_full, obs=False, del_t=12, argo="full", mask_argo="anhang"
    )

    vs.timeseries_plotting(
        part_full, resume_full, obs=False, del_t=1, argo="full", mask_argo="anhang"
    )

    vs.timeseries_plotting(
        part_full, resume_full, obs=False, del_t=12, argo="full", mask_argo="anhang_nw"
    )

    vs.hc_plotting(
        part_full, resume_full, time=[0,60], obs=False, mask_argo="anhang_nw"
    )

    vs.hc_plotting(
        part_full, resume_full, time=[0,60], obs=False, mask_argo="anhang"
    )

elif plotting == "argo":
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
    vs.timeseries_plotting(
        part_depth, resume_depth, del_t=1, argo="full", mask_argo="anhang"
    )

    vs.hc_plotting(part_argo, resume_argo, time=600, obs=False, mask_argo="anhang")
    vs.hc_plotting(
        part_argo, resume_argo, time=[720, 732], obs=False, mask_argo="anhang"
    )
    vs.hc_plotting(
        part_argo, resume_argo, time=[732, 744], obs=False, mask_argo="anhang"
    )
    vs.pattern_corr_plot(
        part_argo,
        resume_iter=resume_argo,
        del_t=1,
        obs=False,
        argo="argo",
        mask_argo="anhang",
    )
    vs.pattern_corr_plot(
        part_argo,
        resume_iter=resume_argo,
        del_t=12,
        obs=False,
        argo="argo",
        mask_argo="anhang",
    )
    vs.pattern_corr_plot(
        part_depth,
        resume_iter=resume_depth,
        del_t=12,
        obs=False,
        argo="argo",
        mask_argo="anhang",
    )

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
    # vs.uncertainty_plot(part_full, resume_full, length=764, del_t=12)
    vs.hc_plotting(part_full, resume_full, time=[0, 120], obs=False, mask_argo="anhang")
    vs.hc_plotting(part_full, resume_full, time=[0, 24], obs=False, mask_argo="anhang")
    vs.hc_plotting(
        part_full, resume_full, time=[120, 180], obs=False, mask_argo="anhang"
    )
    vs.hc_plotting(
        part_full, resume_full, time=[720, 732], obs=False, mask_argo="anhang"
    )
    vs.hc_plotting(
        part_full, resume_full, time=[732, 744], obs=False, mask_argo="anhang"
    )
    vs.correlation_plotting(
        part_full, resume_full, obs=False, starts=[0, 180], ends=[180, 552]
    )
    vs.pattern_corr_plot(
        part_full,
        resume_iter=resume_full,
        del_t=1,
        obs=False,
        argo="full",
        mask_argo="anhang",
    )
    vs.pattern_corr_plot(
        part_full,
        resume_iter=resume_full,
        del_t=12,
        obs=False,
        argo="full",
        mask_argo="anhang",
    )

elif plotting == "direct":
    # Full direct reconstructions
    vs.timeseries_plotting(
        part_full, resume_full, obs=True, del_t=1, argo="full", mask_argo="anhang"
    )
    vs.timeseries_plotting(
        part_full,
        resume_full,
        obs=True,
        del_t=1,
        argo="full",
        mask_argo="anhang",
    )
    vs.timeseries_plotting(
        part_full,
        resume_full,
        obs=True,
        del_t=12,
        argo="full",
        mask_argo="anhang",
        compare=True,
    )
    vs.hc_plotting(part_full, resume_full, time=[0, 120], obs=True, mask_argo="anhang")
    vs.hc_plotting(
        part_full, resume_full, time=[720, 732], obs=True, mask_argo="anhang"
    )

    vs.pattern_corr_plot(
        part_full,
        resume_iter=resume_full,
        del_t=1,
        obs=True,
        argo="full",
        mask_argo="anhang",
    )
    vs.pattern_corr_plot(
        part_full,
        resume_iter=resume_full,
        del_t=12,
        obs=True,
        argo="full",
        mask_argo="anhang",
    )


elif plotting == "anhang":
    # Argo direct Anhang reconstructions
    vs.timeseries_plotting(
        part_argo,
        resume_argo,
        obs=True,
        del_t=1,
        argo="anhang",
        mask_argo="anhang",
        compare=True,
    )

    vs.timeseries_plotting(
        part_argo, resume_argo, obs=False, del_t=1, argo="anhang", mask_argo="anhang"
    )
    vs.hc_plotting(
        part_argo, resume_argo, time=[754, 764], obs=False, mask_argo="anhang"
    )
    vs.hc_plotting(
        part_argo, resume_argo, time=[754, 764], obs=True, mask_argo="anhang"
    )
    vs.pattern_corr_plot(
        part_argo,
        resume_iter=resume_argo,
        del_t=1,
        obs=True,
        argo="anhang",
        mask_argo="anhang",
    )
    vs.pattern_corr_plot(
        part_argo,
        resume_iter=resume_argo,
        del_t=12,
        obs=True,
        argo="anhang",
        mask_argo="anhang",
    )

elif plotting == "all":
    # Argo plot masked assimilation
    # vs.plot_val_error(
    #    part_argo, 600000, interval=5000, combine_start=80, in_channels=20
    # )
    vs.timeseries_plotting(
        part_argo, resume_argo, obs=False, del_t=1, argo="argo", mask_argo="anhang"
    )
    vs.timeseries_plotting(
        part_argo, resume_argo, obs=False, del_t=12, argo="argo", mask_argo="anhang"
    )
    vs.hc_plotting(part_argo, resume_argo, time=600, obs=False, mask_argo="anhang")
    vs.hc_plotting(
        part_argo, resume_argo, time=[720, 732], obs=False, mask_argo="anhang"
    )
    vs.hc_plotting(
        part_argo, resume_argo, time=[732, 744], obs=False, mask_argo="anhang"
    )
    vs.pattern_corr_plot(
        part_argo,
        resume_iter=resume_argo,
        del_t=1,
        obs=False,
        argo="argo",
        mask_argo="anhang",
    )
    vs.pattern_corr_plot(
        part_argo,
        resume_iter=resume_argo,
        del_t=12,
        obs=False,
        argo="argo",
        mask_argo="anhang",
    )

    # Full masked assimilation plots
    vs.plot_val_error(
        part_full, 1000000, interval=25000, combine_start=60, in_channels=20
    )
    vs.timeseries_plotting(
        part_full, resume_full, obs=False, del_t=1, argo="full", mask_argo="anhang"
    )
    vs.timeseries_plotting(
        part_full, resume_full, obs=False, del_t=12, argo="full", mask_argo="anhang"
    )
    # vs.uncertainty_plot(part_full, resume_full, length=764, del_t=12)
    vs.hc_plotting(part_full, resume_full, time=[0, 120], obs=False, mask_argo="anhang")
    vs.hc_plotting(part_full, resume_full, time=[0, 24], obs=False, mask_argo="anhang")
    vs.hc_plotting(
        part_full, resume_full, time=[120, 180], obs=False, mask_argo="anhang"
    )
    vs.hc_plotting(
        part_full, resume_full, time=[720, 732], obs=False, mask_argo="anhang"
    )
    vs.hc_plotting(
        part_full, resume_full, time=[732, 744], obs=False, mask_argo="anhang"
    )
    vs.pattern_corr_plot(
        part_full,
        resume_iter=resume_full,
        del_t=1,
        obs=False,
        argo="full",
        mask_argo="anhang",
    )
    vs.pattern_corr_plot(
        part_full,
        resume_iter=resume_full,
        del_t=12,
        obs=False,
        argo="full",
        mask_argo="anhang",
    )

    # Full direct reconstructions
    vs.timeseries_plotting(
        part_full, resume_full, obs=True, del_t=1, argo="full", mask_argo="anhang"
    )
    vs.timeseries_plotting(
        part_full,
        resume_full,
        obs=True,
        del_t=12,
        argo="full",
        mask_argo="anhang",
        compare=True,
    )
    vs.hc_plotting(part_full, resume_full, time=[0, 120], obs=True, mask_argo="anhang")
    vs.hc_plotting(
        part_full, resume_full, time=[720, 732], obs=True, mask_argo="anhang"
    )

    vs.pattern_corr_plot(
        part_full,
        resume_iter=resume_full,
        del_t=1,
        obs=True,
        argo="full",
        mask_argo="anhang",
    )
    vs.pattern_corr_plot(
        part_full,
        resume_iter=resume_full,
        del_t=12,
        obs=True,
        argo="full",
        mask_argo="anhang",
    )

    # Argo direct Anhang reconstructions
    vs.timeseries_plotting(
        part_full, resume_full, obs=True, del_t=1, argo="anhang", mask_argo="anhang"
    )
    vs.timeseries_plotting(
        part_full, resume_full, obs=True, del_t=12, argo="anhang", mask_argo="anhang"
    )
    vs.hc_plotting(part_full, resume_full, time=760, obs=True, mask_argo="anhang")
    vs.hc_plotting(
        part_full, resume_full, time=[754, 764], obs=True, mask_argo="anhang"
    )
    vs.pattern_corr_plot(
        part_full,
        resume_iter=resume_full,
        del_t=1,
        obs=True,
        argo="anhang",
        mask_argo="anhang",
    )
    vs.pattern_corr_plot(
        part_full,
        resume_iter=resume_full,
        del_t=12,
        obs=True,
        argo="anhang",
        mask_argo="anhang",
    )
