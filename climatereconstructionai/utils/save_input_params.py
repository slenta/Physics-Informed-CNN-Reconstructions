import os
import json


def save_input_params(cfg, dir, filename="input_params_summary.json"):
    """
    Save the input parameters to a JSON file in the snapshot directory.

    Parameters:
    cfg (module): The configuration module containing the parameters.
    filename (str): The name of the summary file to save.
    """
    # Collect relevant input parameters
    input_params = {
        "lr": cfg.lr,
        "n_filters": cfg.n_filters,
        "out_channels": cfg.out_channels,
        "encoding_layers": cfg.encoding_layers,
        "pooling_layers": cfg.pooling_layers,
        "n_channel_steps": cfg.n_channel_steps,
    }

    # Write the parameters to a JSON file in the snapshot directory
    summary_file = os.path.join(dir, filename)
    with open(summary_file, "w") as f:
        json.dump(input_params, f, indent=4)
