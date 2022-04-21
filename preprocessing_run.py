from preprocessing import preprocessing
import config as cfg

cfg.set_preprocessing_args()

if cfg.mode == 'image':
    print(cfg.attribute0, cfg.attribute1)
    dataset = preprocessing(cfg.image_dir, cfg.image_name, cfg.image_size, 'image', cfg.depth, cfg.attribute_depth, cfg.attribute_anomaly, cfg.attribute_argo, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
    dataset.save_data()
elif cfg.mode == 'mask':
    dataset = preprocessing(cfg.mask_dir, cfg.mask_name, cfg.image_size, 'mask', cfg.depth, cfg.attributes)
    dataset.save_data()
elif cfg.mode == 'both':
    dataset = preprocessing(cfg.image_dir, cfg.image_name, cfg.image_size, 'image', cfg.depth, cfg.attributes, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
    dataset1 = preprocessing(cfg.mask_dir, cfg.mask_name, cfg.image_size, 'mask', cfg.depth, cfg.attributes, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
    dataset.save_data()
    dataset1.save_data()
