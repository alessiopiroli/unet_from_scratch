import argparse
from unet.utils.misc import load_config
from unet.utils.trainer import Trainer

def main(args):
    config = load_config(args.config)
    trainer = Trainer(config)
    trainer.evaluate_model(args.ckpt)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default="unet/config/unet_config.yaml", help="Config path")
    parser.add_argument("ckpt", type=str)
    args = parser.parse_args()
    main(args)
