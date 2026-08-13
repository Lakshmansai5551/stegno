# config.py

MODEL_CONFIG = {
    "in_channels": 6,
    "out_channels": 3,
    "base_channels": 32,   # Change to 64 in Colab
    "depth": 4,            # Change to 5 later
    "use_batchnorm": True,
    "use_residual": True
}

TRAIN_CONFIG = {
    "image_size": 128,     # Change to 256 in Colab
    "batch_size": 2,       # Increase later
    "lr": 1e-4,
    "epochs": 10,
    "device": "cuda"
}