import argparse
from unet.utils.misc import load_config
from unet.utils.trainer import Trainer


def main(args):
    config = load_config(args.config)
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default="unet/config/unet_config.yaml", help="Config path")
    args = parser.parse_args()
    main(args)
