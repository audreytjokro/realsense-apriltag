from __future__ import annotations

import argparse
from pathlib import Path

from .training import RunConfig, evaluate_checkpoint, train_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or evaluate smell-only spatial sequence models."
    )
    parser.add_argument("--architecture", choices=("transformer", "temporal-cnn"))
    parser.add_argument("--temporal-mode", choices=("causal", "bidirectional"))
    parser.add_argument(
        "--target",
        choices=("position", "distance", "area", "height", "velocity"),
    )
    parser.add_argument("--session")
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument(
        "--evaluation-scheme",
        choices=("within-session", "leave-one-session-out"),
        default="within-session",
    )
    parser.add_argument("--held-out-session")
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
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--aggregate-report", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--render-position-video", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    if arguments.aggregate_report:
        if arguments.eval_only or arguments.checkpoint or arguments.resume:
            parser.error("--aggregate-report cannot be combined with training/evaluation flags")
        from .manifest import MANIFEST_PATH, validate_run_manifest

        manifest_path = arguments.manifest or MANIFEST_PATH
        runs = validate_run_manifest(manifest_path)
        if runs and runs[0].study == "canonical":
            from .reporting import generate_report

            report = generate_report(manifest_path)
        else:
            from .study_reporting import generate_study_report

            report = generate_study_report(manifest_path)
        print(report)
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
            selector_overrides={
                "architecture": arguments.architecture,
                "temporal_mode": arguments.temporal_mode,
                "target": arguments.target,
                "session": arguments.session,
            },
            render_video=arguments.render_position_video,
        )
        return
    missing = [
        flag
        for flag, value in (
            ("--architecture", arguments.architecture),
            ("--temporal-mode", arguments.temporal_mode),
            ("--target", arguments.target),
        )
        if value is None
    ]
    if missing:
        parser.error("training requires " + ", ".join(missing))
    configuration_values = {
        "architecture": arguments.architecture,
        "temporal_mode": arguments.temporal_mode,
        "target": arguments.target,
        "session": arguments.session,
        "seed": arguments.seed,
        "epochs": arguments.epochs,
        "batch_size": arguments.batch_size,
        "learning_rate": arguments.learning_rate,
        "weight_decay": arguments.weight_decay,
        "minimum_learning_rate": arguments.minimum_learning_rate,
        "gradient_clip": arguments.gradient_clip,
        "device": arguments.device,
        "output_directory": arguments.output_directory,
        "sequence_length": arguments.sequence_length,
        "evaluation_scheme": arguments.evaluation_scheme,
        "held_out_session": arguments.held_out_session,
    }
    if arguments.experiment_root is not None:
        configuration_values["experiment_root"] = arguments.experiment_root
    config = RunConfig(**configuration_values)
    output = train_run(
        config,
        resume_checkpoint=arguments.resume,
        render_position_video=arguments.render_position_video,
    )
    print(output)


if __name__ == "__main__":
    main()
