# SRSRT: Spacetime Realtime Super Resolution Transformer
# Introduces a u-net style transformer in both the spacial and temporal domain
# Employs a modification of the swin transformer




TODO:
i need to see if attention k,v dictionaries can be convolved usefully to lower resolutions

# Running SRSRT:
## Data preparation
python SRSRT.py 

## Training
``python SRSRT.py prepare_training <training_set_path>``
``python SRSRT.py prepare_evaluation <evaluation_set_path>``
``python SRSRT.py train <model_name> <training_set_path>``
``python SRSRT.py evaluate <model_name> <evaluation_set_path>``
