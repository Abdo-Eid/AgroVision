from agrovision_core.utils.io import load_config
from agrovision_core.train.train import train

cfg = load_config("config/config.yaml")

model, train_metrics = train(cfg)
print("✅ Training finished")
print(train_metrics)
