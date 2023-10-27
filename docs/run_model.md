# Running LVSRFIT:

Note: Everything in this file is all relative to the `model` directory.

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
``python LVSRFIT.py prepare_data``

## Commands
Train the model:
- Command: ``python LVSRFIT.py train <model_name> <training_set_path>`` 
- Example: ``python LVSRFIT.py train paper_model_final .\vimeo_septuplet\sep_trainlist.txt`` 

Evaluate the model's accuracy:
- Command: ``python LVSRFIT.py eval <model_name> <evaluation_set_path>``
- Example: ``python LVSRFIT.py eval paper_model_final .\vimeo_septuplet\sep_testlist.txt``

Continuously display inference outputs:
- Command: ``python LVSRFIT.py display <model_name> <evaluation_set_path>``
- Example: ``python LVSRFIT.py display paper_model_final .\vimeo_septuplet\sep_testlist.txt``

Display inference outputs for a specified Vimeo file:
- Command: ``python LVSRFIT.py display_one <model_name> <vimeo path> <sequence path> <optional: input sequence length>``
- Example: ``python LVSRFIT.py display_one paper_model_final .\vimeo_septuplet 00096/0674 2``

Calculate the number of FPS the model can run at
- Command: ``python LVSRFIT.py fps_test <model_name> <test_set_path>``
- Example: ``python LVSRFIT.py fps_test paper_model_final .\vimeo_septuplet\sep_testlist.txt``

View a plot of all the Loss values in a log file
- Command: ``python LVSRFIT.py observe_log <tag> <log_path>``
- Example: ``python LVSRFIT.py observe_log Loss .\logs\2023-08-05-10-33-11_paper_model_final.txt``
