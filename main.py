from trainer import Trainer
from infenrence import BeamSearcher
import config
import warnings
warnings.simplefilter('error')  # treat warnings as exceptions

def main():
    if config.train:
        trainer = Trainer()
        trainer.train()
    if config.test:
        beamsearcher = BeamSearcher(config.model_path, config.output_dir)
        beamsearcher.decode()


if __name__ == "__main__":
    main()
