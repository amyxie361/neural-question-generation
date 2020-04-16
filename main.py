from trainer import Trainer
from infenrence import BeamSearcher
import config


def main():
    if config.train:
        if config.load_model:
            trainer = Trainer(config.model_path)
        else:
            trainer = Trainer()
        trainer.train()
    if config.test:
        beamsearcher = BeamSearcher(config.model_path, config.output_dir)
        beamsearcher.decode()
    if config.interface:
        beamsearcher = BeamSearcher(config.model_path, config.output_dir)
        sent = input("Please input:")
        while len(sent):
            questions = beamsearcher.interactive(sent)
            print([" ".join(q) for q in questions])
            sent = input("Please input:")

if __name__ == "__main__":
    main()
