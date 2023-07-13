SRSRT_SETTINGS_DEFAULT = {
    "scale": 4,
    "num_frames": 7,
    "batch_size": 4,
    "num_epochs": 200,
    "max_iters_per_epoch": 100000,
    "training_save_interval": 1000,
    "training": {
        "learning_rate": 0.0002,
        "beta1": 0.9,
        "beta2": 0.99,
        "warmup_iter": -1, # 5000  # -1: no warm up
        "T_period": [20000, 40000, 60000, 80000],
        "restarts": [20000, 40000, 60000],
        "restart_weights": [0.5, 0.5, 0.5],
        "weight_decay": 0,
        "eta_min": 1e-7
    }
}