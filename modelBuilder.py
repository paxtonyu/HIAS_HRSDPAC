import argparse
import yaml

from models.baseCNN import baseCNN
from models.CNN_plus_KAN import CNN_plus_KAN
from models.ViT import ViT
from models.HSCNN import HSCNN

class Config:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.MODEL = self.config.get("MODEL", {})
        self.SOLVER = self.config.get("SOLVER", {})
        self.OUTPUT_DIR = self.config.get("OUTPUT_DIR", "")
    
    def load_config(self, config_path):
        """loading the config file from yaml"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def update_from_args(self, args):
        """Update configuration with command-line arguments."""
        if args.mode:
            self.SOLVER["mode"] = args.mode


def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--mode", type=str, choices=["train", "test", "predict"], help="Training, testing, or prediction mode.")
    return parser.parse_args()

def init_my_model(config):
    """initialize the model using the yaml config file"""
    model_name = config.MODEL.get("name")
    
    if "baseCNN" in model_name:
        model = baseCNN()
    elif "CNN_plus_KAN" in model_name:
        model = CNN_plus_KAN(
            in_channels=config.SOLVER["pca_components"], 
            C1_num=config.MODEL["C1_num"], 
            num_classes=config.MODEL["num_classes"],
            kan_list=config.MODEL["kan_list"],
            dropout=config.MODEL["dropout"]
        )
    elif "ViT" == model_name:
        model = ViT(
            image_size=config.MODEL["patch_size"],
            patch_size=1,
            num_classes=config.MODEL["num_classes"],
            dim=config.MODEL["dim"],
            depth=config.MODEL["depth"],
            heads=config.MODEL["heads"],
            mlp_dim=config.MODEL["mlp_dim"],
            pool=config.MODEL["pool"],
            channels=config.SOLVER["pca_components"],
            dim_head=config.MODEL["dim_head"],
            dropout=config.MODEL["dropout"],
            emb_dropout=config.MODEL["emb_dropout"],
        )
    elif "HSCNN" == model_name:
        model = HSCNN(config)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model