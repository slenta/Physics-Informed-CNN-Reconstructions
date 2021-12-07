import sys

sys.path.append('./')

from model.PConvLSTM import PConvLSTM
from utils.evaluator import PConvLSTMEvaluator
from utils.netcdfloader import NetCDFLoader
from utils.io import load_ckpt
import config as cfg

cfg.set_evaluation_args()

evaluator = PConvLSTMEvaluator(cfg.evaluation_dirs[0], cfg.mask_dir+cfg.mask_names[0], cfg.data_root_dir + 'test_large/', cfg.data_types[0])

if cfg.infill:
    """dataset_val = NetCDFLoader(cfg.data_root_dir, cfg.img_names, cfg.mask_dir, cfg.mask_names, cfg.infill, cfg.data_types, cfg.lstm_steps)
    lstm = True
    if cfg.lstm_steps == 0:
        lstm = False
    model = PConvLSTM(image_size=cfg.image_size,
                      num_enc_dec_layers=cfg.encoding_layers,
                      num_pool_layers=cfg.pooling_layers,
                      num_in_channels=len(cfg.data_types),
                      lstm=lstm).to(cfg.device)

    load_ckpt(cfg.snapshot_dir, [('model', model)], cfg.device)

    model.eval()"""

    evaluator.infill(None, None, None)

if cfg.create_images:
    start_date = cfg.create_images.split(',')[0]
    end_date = cfg.create_images.split(',')[1]
    create_video = False
    if cfg.create_video:
        create_video = True
    evaluator.create_evaluation_images('image.nc', create_video, start_date, end_date)
    evaluator.create_evaluation_images('gt.nc', create_video, start_date, end_date)
    evaluator.create_evaluation_images('output.nc', create_video, start_date, end_date)
    evaluator.create_evaluation_images('output_comp.nc', create_video, start_date, end_date)

if cfg.create_report:
    evaluator.create_evaluation_report()