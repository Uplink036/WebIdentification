import argparse
import os
import wandb
from dotenv import load_dotenv
from sweep_train import run_sweep_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a W&B sweep agent")
    parser.add_argument(
        "--sweep-id",
        required=True,
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    agent_kwargs = {
        "sweep_id": args.sweep_id,
        "function": run_sweep_train,
        "count": args.count,
        "project": "WebIdentification",
    }
    wandb.agent(**agent_kwargs)


if __name__ == "__main__":
    main()