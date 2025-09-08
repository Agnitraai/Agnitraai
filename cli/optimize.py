"""CLI entry point for optimizing a model."""

import argparse
from agnitra.demo import DemoNet


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize a model using Agnitra SDK")
    parser.add_argument("--model", default="demo-model", help="Name of the model to optimize")
    args = parser.parse_args()

    net = DemoNet()
    result = net.optimize(model=args.model)
    print(result)


if __name__ == "__main__":
    main()
