from __future__ import annotations

import argparse
from pathlib import Path

from .training import RunConfig, evaluate_checkpoint, train_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train or evaluate the bidirectional GRU leave-one-session-out area "
            "experiment (5 July-30 sessions only). Architecture/temporal-mode/target/"
            "evaluation-scheme are fixed -- this experiment isolates one variable."
        )
    )
    parser.add_argument(
        "--held-out-session",
        help=(
            "One of: mint-only-horizontal-run01, "
            "caret-mint-left-lavender-right-run01, "
            "caret-lavender-left-mint-right-run01, "
            "lavender-only-horizontal-run01, "
            "inverted-caret-lavender-left-mint-right-run01"
        ),
    )
    parser.add_argument(
        "--architecture-version",
        choices=("v1", "v2"),
        default="v1",
        help=(
            "v1: hidden=64, no dropout/normalization (the original single-variable-"
            "isolation model). v2: hidden=161 (~matches the TCN's parameter count), "
            "+dropout, +LayerNorm on the GRU output."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-directory")
    parser.add_argument("--experiment-root")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--aggregate-report", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    if arguments.aggregate_report:
        from .report import generate_report

        print(generate_report())
        return
    if arguments.eval_only:
        if arguments.checkpoint is None:
            parser.error("--eval-only requires --checkpoint")
        evaluate_checkpoint(
            arguments.checkpoint,
            output_directory=(
                None
                if arguments.output_directory is None
                else Path(arguments.output_directory).expanduser().resolve()
            ),
        )
        return
    if not arguments.held_out_session:
        parser.error("training requires --held-out-session")
    configuration_values = {
        "held_out_session": arguments.held_out_session,
        "architecture_version": arguments.architecture_version,
        "seed": arguments.seed,
        "epochs": arguments.epochs,
        "batch_size": arguments.batch_size,
        "learning_rate": arguments.learning_rate,
        "weight_decay": arguments.weight_decay,
        "minimum_learning_rate": arguments.minimum_learning_rate,
        "gradient_clip": arguments.gradient_clip,
        "device": arguments.device,
        "output_directory": arguments.output_directory,
    }
    if arguments.experiment_root is not None:
        configuration_values["experiment_root"] = arguments.experiment_root
    output = train_run(RunConfig(**configuration_values))
    print(output)


if __name__ == "__main__":
    main()
