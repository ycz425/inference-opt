import argparse
from experiment import ExperimentRunner
from pathlib import Path
from config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=Path, default='config.yaml')
    parser.add_argument('-o', '--output', type=Path, default='experiments')
    args = parser.parse_args()

    runner = ExperimentRunner(args.output)
    config = load_config(args.config)

    runner.run(config)




if __name__ == '__main__':
    main()







