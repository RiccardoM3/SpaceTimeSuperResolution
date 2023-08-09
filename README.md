# SRSRT: Spacetime Realtime Super Resolution Transformer

# Running SRSRT:

## Dataset Download
First download the full 82GB vimeo septuplet 90K dataset from [here](http://toflow.csail.mit.edu/)
Extract the contents of the zip and place into the root of this repository. The directory should be named `vimeo_septuplet`, and it should contain a `sequences` subdirectory. 

## Environment
First install the required python packages
`pip install -r requirements.txt`

Then use the below command to install torch with CUDA:
`pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`

## Data Preparation
Data preparation must be run before training or evaluation is run:
``python SRSRT.py prepare_data``

## Commands
``python SRSRT.py train <model_name> <training_set_path>`` 

``python SRSRT.py eval <model_name> <evaluation_set_path>``

``python SRSRT.py display <model_name> <evaluation_set_path>``

``python SRSRT.py display_one <model_name> <vimeo path> <sequence path> <optional: input sequence length>``

``python SRSRT.py observe_log <tag> <log_path>``

## Examples
``python SRSRT.py train paper_model_final .\vimeo_septuplet\sep_trainlist.txt`` 

``python SRSRT.py eval paper_model_final .\vimeo_septuplet\sep_testlist.txt``

``python SRSRT.py display paper_model_final .\vimeo_septuplet\sep_testlist.txt``

``python SRSRT.py display_one paper_model_final .\vimeo_septuplet 00096/0674 2``

``python SRSRT.py observe_log Loss .\logs\2023-08-05-10-33-11_paper_model_final.txt``