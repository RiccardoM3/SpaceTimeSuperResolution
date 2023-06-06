# SRSRT: Spacetime Realtime Super Resolution Transformer
- Introduces a u-net style transformer in both the spacial and temporal domain
- Employs a modification of swin transformer blocks
    - check if its in the ablation study
- list of papers and things that i can run inference on
- size of swin transformer blocks
- masks with non transformers

# Running SRSRT:
## Data preparation
``python SRSRT.py prepare_training <training_set_path>``

``python SRSRT.py prepare_evaluation <evaluation_set_path>``

## Training
``python SRSRT.py train <model_name> <training_set_path>`` 

``python SRSRT.py evaluate <model_name> <evaluation_set_path>``
