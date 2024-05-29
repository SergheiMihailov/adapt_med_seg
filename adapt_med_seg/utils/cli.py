import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="segvol_baseline",
        choices=[
            "segvol_baseline",
            "segvol_lora",
            "segvol_moe",
            "segvol_context_prior",
        ],
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets",
        help="Path to the dataset(s)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for training"
    )

    # TODO: Check if this is correct with Miki
    parser.add_argument(
        "--modalities",
        type=str,
        default=["MRI"],
        nargs="*",
        help="List of modalities to use",
    )
    parser.add_argument('--max_train_samples',
                        type=int,
                        default=None,
                        help='Use only a specific number of samples for training/validation.')
    parser.add_argument('--max_val_samples',
                        type=int,
                        default=None,
                        help='Use only a specific number of samples for testing.')
    parser.add_argument('--max_test_samples',
                        type=int,
                        default=None,
                        help='Use only a specific number of samples for testing.')
    parser.add_argument("--prompt_types", type=str, default=["point", "text"], nargs="*",
                        choices=["point", "bbox", "text"],
                        help="List of prompt types to use in evaluation. Only works with `eval.py`,"+\
                                "not with `train.py`. Default is all prompt types.")
    parser.add_argument("--perturb_bbox", type=float, default=None,
                        help="Apply random perturbation to the bounding box prompt by"+\
                                "randomly shifting the corners of it." +\
                                "Argument is the maximum voxel shift in each direction.")

    parser.add_argument("--test_at_end", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--num_sanity_val_steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches to accumulate before backprop. In theory, this could reduce training time and simulate larger batches (which we can't do with the current models).",
    )
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--target_modules", type=str, default=None, nargs="*")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--log_every_n_steps", type=int, default=50) # high default to avoid spamming
    parser.add_argument("--wandb_project", type=str, default="dl2_g33")
    parser.add_argument("--lr", "--learning_rate", type=float, default=5e-5)
    parser.add_argument(
        "--betas", type=tuple[float, float], default=(0.9, 0.999), nargs=2
    )
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--train_only_vit", action="store_true")
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--version_id", type=str, default=None)

    args = parser.parse_args()
    # dataclass can, but argparse can't handle union types,
    # so we need to do this manually
    if isinstance(args.target_modules, list) and len(args.target_modules) == 1:
        args.target_modules = args.target_modules[0]
    return args
