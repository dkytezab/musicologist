from diffusion.load_model import get_diff_model
from torchinfo import summary

model, _ = get_diff_model(model_name="stable-diffusion", device="cuda")
print(summary(model))
print(model)
print(model.config)
