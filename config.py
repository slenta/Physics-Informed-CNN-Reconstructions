import argparse
import torch

MEAN = [0.485, 0.456, 0.406, 0.406, 0.406, 0.406, 0.406, 0.406, 0.406, 0.406]
STD = [0.229, 0.224, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225]

# LAMBDA_DICT_IMG_INPAINTING = {
#    'hole': 6.0, 'tv': 0.1, 'valid': 1.0, 'prc': 0.05, 'style': 120.0
# }
LAMBDA_DICT_IMG_INPAINTING = {
    "hole": 60.0,
    "tv": 0.1,
    "valid": 60.0,
    "prc": 0.05,
    "style": 10.0,
    "total": 0.0,
}
LAMBDA_DICT_HOLE = {"hole": 1.0}

PDF_BINS = [0, 0.01, 0.02, 0.1, 1, 2, 10, 100]

# def get_format(dataset_name):
#    json_data = pkgutil.get_data(__name__, "static/dataset_format.json")
#    dataset_format = json.loads(json_data)
#
#    return dataset_format[str(dataset_name)]


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parser.parse_args(open(values).read().split(), namespace)


def str_list(arg):
    return arg.split(",")


def int_list(arg):
    return list(map(int, arg.split(",")))


def lim_list(arg):
    lim = list(map(float, arg.split(",")))
    assert len(lim) == 2
    return lim


def global_args(parser, arg_file=None, prog_func=None):
    import torch

    if arg_file is None:
        import sys

        argv = sys.argv[1:]
    else:
        argv = ["--load-from-file", arg_file]

    global progress_fwd
    progress_fwd = prog_func

    args = parser.parse_args(argv)

    args_dict = vars(args)
    for arg in args_dict:
        globals()[arg] = args_dict[arg]

    torch.backends.cudnn.benchmark = True
    globals()[device] = torch.device(device)

    # globals()["dataset_format"] = get_format(args.dataset_name)

    global skip_layers
    global gt_channels
    global recurrent_steps

    # gt_channels = []
    # for i in range(out_channels):
    #    gt_channels.append((i + 1) * channel_steps + i * (channel_steps + 1))
    gt_channels = out_channels

    if lstm_steps:
        recurrent_steps = lstm_steps
    # elif gru_steps:
    #    recurrent_steps = gru_steps
    else:
        recurrent_steps = 0


data_types = None
mask_name = None
im_name = None
evaluation_dirs = None
partitions = None
infill = None
create_images = None
create_video = None
create_report = None
log_dir = None
save_dir = None
im_dir = None
mask_dir = None
resume_iter = None
device = None
batch_size = None
n_threads = None
finetune = None
lr = None
lr_finetune = None
max_iter = None
log_interval = None
save_model_interval = None
lstm_steps = None
prev_next_steps = None
encoding_layers = None
pooling_layers = None
eval_names = None
eval_threshold = None
eval_range = None
ts_range = None
eval_timesteps = None
out_channels = None
gt_channels = None
channel_reduction_rate = None
save_snapshot_image = None
loss_criterion = None
mask_year = None
im_year = None
in_channels = None
save_part = None
image_size = None
attention = None
smoothing_factor = None
weights = None
skip_layers = None
vis_interval = None
prepro_mode = None
lon1 = None
lon2 = None
lat1 = None
lat2 = None
latlons = None
save_part = None
attribute_depth = None
attribute_anomaly = None
attribute_argo = None
mask_argo = None
eval_im_year = None
val_interval = None
val_dir = None
eval_mask_year = None
ensemble_member = None
n_filters = None
disable_first_bn = None
depth = None
val_cut = None
combine_layers = None
vis_step = None
combine_start = None
eval_full = None
nw_corner = None


def set_train_args(arg_file=None):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--log_dir", type=str, default="logs/")
    arg_parser.add_argument("--save_dir", type=str, default="../Asi_maskiert/results/")
    arg_parser.add_argument(
        "--im_dir", type=str, default="../Asi_maskiert/original_image/"
    )
    arg_parser.add_argument(
        "--mask_dir", type=str, default="../Asi_maskiert/original_masks/"
    )
    arg_parser.add_argument("--im_name", type=str, default="Image_")
    arg_parser.add_argument("--mask_name", type=str, default="Maske_")
    arg_parser.add_argument("--mask_year", type=str, default="1958_2021_newgrid")
    arg_parser.add_argument("--im_year", type=str, default="r3_14_newgrid")
    arg_parser.add_argument("--ensemble_member", type=int, default=2)
    arg_parser.add_argument("--resume_iter", type=int)
    arg_parser.add_argument("--device", type=str, default="cpu")
    arg_parser.add_argument("--batch_size", type=int, default=4)
    arg_parser.add_argument("--n_threads", type=int, default=32)
    arg_parser.add_argument("--finetune", action="store_true")
    arg_parser.add_argument("--lr", type=float, default=2e-4)
    arg_parser.add_argument("--lr-finetune", type=float, default=5e-5)
    arg_parser.add_argument("--max_iter", type=int, default=500000)
    arg_parser.add_argument("--log_interval", type=int, default=10)
    arg_parser.add_argument("--save_model_interval", type=int, default=25000)
    arg_parser.add_argument("--lstm_steps", type=int, default=0)
    arg_parser.add_argument("--prev-next-steps", type=int, default=0)
    arg_parser.add_argument("--encoding_layers", type=int, default=4)
    arg_parser.add_argument("--pooling_layers", type=int, default=2)
    arg_parser.add_argument("--out_channels", type=int, default=20)
    arg_parser.add_argument("--in_channels", type=int, default=20)
    arg_parser.add_argument("--loss_criterion", type=int, default=0)
    arg_parser.add_argument("--eval-timesteps", type=str, default="0,1,2,3,4")
    arg_parser.add_argument("--channel-reduction-rate", type=int, default=1)
    arg_parser.add_argument("--save_part", type=str, default="part_1")
    arg_parser.add_argument("--image_size", type=int, default=128)
    arg_parser.add_argument("--weights", type=str, default=None)
    arg_parser.add_argument("--attention", action="store_true")
    arg_parser.add_argument("--disable_skip_layers", action="store_true")
    arg_parser.add_argument("--vis_interval", type=int, default=50000)
    arg_parser.add_argument("--eval_im_year", type=str, default="r2_full_newgrid")
    arg_parser.add_argument("--prepro_mode", type=str, default="none")
    arg_parser.add_argument("--attribute_anomaly", type=str, default="anomalies")
    arg_parser.add_argument("--attribute_depth", type=str, default="depth")
    arg_parser.add_argument("--attribute_argo", type=str, default="argo")
    arg_parser.add_argument("--mask_argo", type=str, default="argo")
    arg_parser.add_argument("--lon1", type=int, default=-60)
    arg_parser.add_argument("--lon2", type=int, default=-10)
    arg_parser.add_argument("--lat1", type=int, default=45)
    arg_parser.add_argument("--lat2", type=int, default=60)
    arg_parser.add_argument("--val_interval", type=int, default=50000)
    arg_parser.add_argument(
        "--val_dir", type=str, default="../Asi_maskiert/results/validation/"
    )
    arg_parser.add_argument("--eval_mask_year", type=str, default="1958_2021_newgrid")
    arg_parser.add_argument("--n_filters", type=int, default=None)
    arg_parser.add_argument("--disable_first_bn", action="store_true")
    arg_parser.add_argument("--depth", type=int, default=0)
    arg_parser.add_argument("--val_cut", action="store_true")
    arg_parser.add_argument("--vis_step", type=int, default=600)
    arg_parser.add_argument("--combine_layers", action="store_true")
    arg_parser.add_argument("--eval_full", action="store_true")
    arg_parser.add_argument("--combine_start", type=int, default=60)
    arg_parser.add_argument("--nw_corner", action="store_true")
    global_args(arg_parser, arg_file)
    args = arg_parser.parse_args()

    global im_name
    global mask_name
    global mask_year
    global im_year
    global log_dir
    global save_dir
    global im_dir
    global mask_dir
    global resume_iter
    global device
    global batch_size
    global n_threads
    global finetune
    global lr
    global lr_finetune
    global max_iter
    global log_interval
    global save_model_interval
    global lstm_steps
    global prev_next_steps
    global encoding_layers
    global pooling_layers
    global image_size
    global eval_timesteps
    global in_channels
    global out_channels
    global gt_channels
    global channel_reduction_rate
    global save_snapshot_image
    global loss_criterion
    global save_part
    global image_size
    global attention
    global skip_layers
    global weights
    global vis_interval
    global eval_im_year
    global prepro_mode
    global lon1
    global lon2
    global lat1
    global lat2
    global attribute_anomaly
    global attribute_argo
    global attribute_depth
    global mask_argo
    global val_interval
    global val_dir
    global eval_mask_year
    global ensemble_member
    global n_filters
    global disable_first_bn
    global depth
    global val_cut
    global combine_layers
    global combine_start
    global eval_full
    global vis_step
    global nw_corner

    im_name = args.im_name
    mask_name = args.mask_name
    eval_timesteps = args.eval_timesteps.split(",")
    log_dir = args.log_dir
    save_dir = args.save_dir
    im_dir = args.im_dir
    mask_dir = args.mask_dir
    resume_iter = args.resume_iter
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    batch_size = args.batch_size
    n_threads = args.n_threads
    finetune = args.finetune
    lr = args.lr
    lr_finetune = args.lr_finetune
    max_iter = args.max_iter
    log_interval = args.log_interval
    save_model_interval = args.save_model_interval
    lstm_steps = args.lstm_steps
    prev_next_steps = args.prev_next_steps
    encoding_layers = args.encoding_layers
    pooling_layers = args.pooling_layers
    channel_reduction_rate = args.channel_reduction_rate
    out_channels = args.out_channels
    gt_channels = []
    loss_criterion = args.loss_criterion
    for i in range(out_channels):
        gt_channels.append((i + 1) * prev_next_steps + i * (prev_next_steps + 1))
    in_channels = args.in_channels
    im_year = args.im_year
    mask_year = args.mask_year
    save_part = args.save_part
    image_size = args.image_size
    attention = args.attention
    weights = args.weights
    if args.disable_skip_layers:
        skip_layers = 0
    else:
        skip_layers = 1
    for i in range(out_channels):
        gt_channels.append((i + 1) * prev_next_steps + i * (prev_next_steps + 1))
    vis_interval = args.vis_interval
    eval_im_year = args.eval_im_year
    prepro_mode = args.prepro_mode
    lon1 = args.lon1
    lon2 = args.lon2
    lat1 = args.lat1
    lat2 = args.lat2
    attribute_depth = args.attribute_depth
    attribute_anomaly = args.attribute_anomaly
    attribute_argo = args.attribute_argo
    mask_argo = args.mask_argo
    val_interval = args.val_interval
    val_dir = args.val_dir
    eval_mask_year = args.eval_mask_year
    ensemble_member = args.ensemble_member
    n_filters = args.n_filters
    disable_first_bn = args.disable_first_bn
    depth = args.depth
    val_cut = args.val_cut
    combine_layers = args.combine_layers
    vis_step = args.vis_step
    combine_start = args.combine_start
    eval_full = args.eval_full
    nw_corner = args.nw_corner


def set_evaluation_args(arg_file=None):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--log_dir", type=str, default="logs/")
    arg_parser.add_argument("--save_dir", type=str, default="../Asi_maskiert/results/")
    arg_parser.add_argument(
        "--im_dir", type=str, default="../Asi_maskiert/original_image/"
    )
    arg_parser.add_argument(
        "--mask_dir", type=str, default="../Asi_maskiert/original_masks/"
    )
    arg_parser.add_argument("--im_name", type=str, default="Image_")
    arg_parser.add_argument("--mask_name", type=str, default="Maske_")
    arg_parser.add_argument("--mask_year", type=str, default="1958_2021_newgrid")
    arg_parser.add_argument("--im_year", type=str, default="r16_newgrid")
    arg_parser.add_argument("--ensemble_member", type=int, default=2)
    arg_parser.add_argument("--resume_iter", type=int)
    arg_parser.add_argument("--lr", type=float, default=2e-4)
    arg_parser.add_argument("--device", type=str, default="cpu")
    arg_parser.add_argument("--batch_size", type=int, default=4)
    arg_parser.add_argument("--n_threads", type=int, default=32)
    arg_parser.add_argument("--lstm_steps", type=int, default=0)
    arg_parser.add_argument("--encoding_layers", type=int, default=4)
    arg_parser.add_argument("--pooling_layers", type=int, default=2)
    arg_parser.add_argument("--out_channels", type=int, default=20)
    arg_parser.add_argument("--in_channels", type=int, default=20)
    arg_parser.add_argument("--save_part", type=str, default="part_1")
    arg_parser.add_argument("--vis_interval", type=int, default=50000)
    arg_parser.add_argument("--eval_im_year", type=str, default="r16_newgrid")
    arg_parser.add_argument("--attribute_anomaly", type=str, default="anomalies")
    arg_parser.add_argument("--attribute_depth", type=str, default="depth")
    arg_parser.add_argument("--attribute_argo", type=str, default="argo")
    arg_parser.add_argument("--mask_argo", type=str, default="argo")
    arg_parser.add_argument("--lonlats", type=list, default=[-60, -10, 45, 60])
    arg_parser.add_argument("--image_size", type=int, default=128)
    arg_parser.add_argument("--vis_step", type=int, default=600)
    arg_parser.add_argument("--prepro_mode", type=str, default="none")
    arg_parser.add_argument(
        "--val_dir", type=str, default="../Asi_maskiert/results/validation/"
    )
    arg_parser.add_argument("--eval_mask_year", type=str, default="1958_2021_newgrid")
    arg_parser.add_argument("--n_filters", type=int, default=None)
    arg_parser.add_argument("--disable_first_bn", action="store_true")
    arg_parser.add_argument("--depth", type=int, default=0)
    arg_parser.add_argument("--val_cut", action="store_true")
    arg_parser.add_argument("--combine_layers", action="store_true")
    arg_parser.add_argument("--eval_full", action="store_true")
    arg_parser.add_argument("--combine_start", type=int, default=60)
    global_args(arg_parser, arg_file)
    args = arg_parser.parse_args()

    global im_name
    global mask_name
    global mask_year
    global im_year
    global log_dir
    global save_dir
    global im_dir
    global mask_dir
    global resume_iter
    global device
    global batch_size
    global n_threads
    global lstm_steps
    global encoding_layers
    global pooling_layers
    global prepro_mode
    global in_channels
    global out_channels
    global save_part
    global eval_im_year
    global lon1
    global lon2
    global lat1
    global lat2
    global attribute_anomaly
    global attribute_argo
    global attribute_depth
    global mask_argo
    global vis_step
    global val_dir
    global eval_mask_year
    global ensemble_member
    global n_filters
    global disable_first_bn
    global depth
    global val_cut
    global combine_layers
    global combine_start
    global eval_full
    global lr
    global image_size

    im_name = args.im_name
    mask_name = args.mask_name
    log_dir = args.log_dir
    save_dir = args.save_dir
    im_dir = args.im_dir
    mask_dir = args.mask_dir
    resume_iter = args.resume_iter
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    batch_size = args.batch_size
    n_threads = args.n_threads
    lstm_steps = args.lstm_steps
    encoding_layers = args.encoding_layers
    pooling_layers = args.pooling_layers
    out_channels = args.out_channels
    in_channels = args.in_channels
    prepro_mode = args.prepro_mode
    im_year = args.im_year
    mask_year = args.mask_year
    save_part = args.save_part
    eval_im_year = args.eval_im_year
    lon1 = args.lonlats[0]
    lon2 = args.lonlats[1]
    lat1 = args.lonlats[2]
    lat2 = args.lonlats[3]
    attribute_depth = args.attribute_depth
    attribute_anomaly = args.attribute_anomaly
    attribute_argo = args.attribute_argo
    mask_argo = args.mask_argo
    val_dir = args.val_dir
    eval_mask_year = args.eval_mask_year
    ensemble_member = args.ensemble_member
    n_filters = args.n_filters
    disable_first_bn = args.disable_first_bn
    depth = args.depth
    val_cut = args.val_cut
    combine_layers = args.combine_layers
    vis_step = args.vis_step
    combine_start = args.combine_start
    eval_full = args.eval_full
    lr = args.lr
    image_size = args.image_size
