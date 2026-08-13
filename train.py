import torch
from models.hiding_unet import HidingUNet
from config import MODEL_CONFIG, TRAIN_CONFIG

device = "cpu"

model = HidingUNet(MODEL_CONFIG).to(device)

cover = torch.randn(2, 3, 128, 128)
secret = torch.randn(2, 3, 128, 128)

output = model(cover, secret)

print("Output shape:", output.shape)