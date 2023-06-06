# SRSRT: Spacetime Realtime Super Resolution Transformer

# Running SRSRT:

## Dataset Download
First download the full 82GB vimeo septuplet 90K dataset from [here](http://toflow.csail.mit.edu/)
Extract the contents of the zip and place into the root of this repository. The directory should be named `vimeo_septuplet`, and it should contain a `sequences` subdirectory. 

## Environment
First install the required python packages
`pip install -r requirements.txt`

## Data Preparation
Data preparation must be run before training or evaluation is run:
``python SRSRT.py prepare_data``

## Training
``python SRSRT.py train <model_name> <training_set_path>`` 

``python SRSRT.py evaluate <model_name> <evaluation_set_path>``
