from optimizer import Optimizer, load_grid, load_constraint
from config import load_config
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=Path, default='config.yaml')
    parser.add_argument('-g', '--grid', type=Path, default='grid.yaml')
    parser.add_argument('-x', '--constraint', type=Path, default=None)
    parser.add_argument('-m', '--metric', type=str, default='output_tokens_per_sec')
    parser.add_argument('-o', '--output', type=Path, default='optimizations')
    
    args = parser.parse_args()

    config = load_config(args.config)
    grid = load_grid(args.grid)

    if args.constraint is None:
        constraint = None
    else:
        constraint = load_constraint(args.constraint)

    optimizer = Optimizer(config=config, output_dir=args.output)
    optimizer.search(grid=grid, metric=args.metric, constraint=constraint)


if __name__ == "__main__":
    main()