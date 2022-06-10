# Infilling spatial precipitation recordings with a memory assisted CNN

## Requirements
- Python 3.7+

```
pip install -r requirements.txt
```

## Usage

### Preprocess 
Download climate data. The dataset should contain a directory with training images and one with masks

To process them into the required input format for the dataloader of the network, execute

`python preprocessing.py`

The argument `--mode` specifies wether image, masks or both still require preprocessing. To specify additional attributes of the temperature profile input data, here are some additional args:

- `--attribute_argo` -> specifies from which time frame the data should be taken
- `--attribute_depth` -> wether to reconstruct only SSTs or also subsurface temperatures
- `--attribute_anomalies` -> input data changed into monthly anomalies using baseline climatology of input data
- `--in_channels` -> amount of depth steps (one input channel of NN for one depth profile)
- `--im_dir, --im_name, --im_year` specify the exact location and name of the input data
- `--lon/--lat` optional preprocessing of cropping the image to specified longitudes, latitudes



### Training
The training process can be started by executing 

`python train.py`

To specify additional args such as the data root directory, use `--arg arg_value`.
Here are some important args:
- `--root_dir` -> root directory of training and validation data
- `--save_dir` -> directory of training checkpoints
- `--mask_dir` -> directory of mask files
- `--img_name` -> comma separated list of training data files stored in the data root directory, have to be same shape! First image is ground truth
- `--mask_name` -> comma separated list of mask files stored in the mask directory, need to correspond to order in img-names
- `--data_types` -> comma separated list of types of variable, need to correspond to order in img-names and mask-names
- `--device` -> cuda or cpu
- `--lstm_steps` -> Number of considered sequences for lstm, set to zero, if lstm module should be deactivated
- `--encoding_layers` -> number of encoding layers in the CNN
- `--pooling_layers` -> number of pooling layers in the CNN
- `--image_size` -> size of image, must be of shape NxN

### Evaluate
The evaluation process can be started by executing

`python train_and_evaluate/evaluate.py`

Important args:
- `--evaluation-dir` -> directory where evaluations will be stored
- `--data-root-dir` -> root directory of test data
- `--mask-dir` -> directory of mask files
- `--img-names` -> comma separated list of training data files stored in the data root directory, have to be same shape!
- `--mask-names` -> comma separated list of mask files stored in the mask directory, need to correspond to order in img-names
- `--data-types` -> comma separated list of types of variable, need to correspond to order in img-names and mask-names
- `--device` -> cuda or cpu
- `--lstm-steps` -> Number of considered sequences for lstm, set to zero, if lstm module should be deactivated
- `--infill` -> 'test', if mask order is irrelevant, 'infill', if mask order is relevant
- `--create-images` -> creates images for time window in format 'YYY-MM-DD-HH:MM,YYYY-MM-DD-HH:MM'
- `--create-video` -> creates video. Images need to be created as well
- `--create-report` -> creates evaluation report for total test data set
