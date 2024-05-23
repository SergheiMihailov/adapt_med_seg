from itertools import accumulate
from adapt_med_seg.data.dataset import MedSegDataset
from adapt_med_seg.models.lightning_model import SegVolLightning
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
import argparse

from lightning.pytorch.loggers import WandbLogger
import wandb

api = wandb.Api()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="segvol_baseline",
        choices=["segvol_baseline", "segvol_lora", "segvol_context_prior"],
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
    parser.add_argument("--cls_idx", type=int, default=0)
    parser.add_argument("--test_at_end", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--num_sanity_val_steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--num_workers",
        help="Number of worker processes to use",
        type=int,
        default=1,
        action="store",
    )
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
    parser.add_argument("--target_modules", type=str, default=None, nargs="+")
    parser.add_argument("--tasks", type=str, default=[], nargs="*")
    parser.add_argument("--modalities", default=["MRI", "CT"], nargs="+")
    args = parser.parse_args()

    print(args.use_wandb)

    # if args.use_wandb:
    wandb_logger = WandbLogger(project="dl2_g33")

    seed_everything(args.seed)
    model = SegVolLightning(
        test_mode=False,
        **vars(args),
    )

    _dataset = MedSegDataset(
        processor=model._model.processor,
        dataset_path=args.dataset_path,
        modalities=args.modalities,
        train=True,
    )

    model.set_dataset(_dataset, cls_idx=args.cls_idx)
    train_dataloader, val_dataloader = _dataset.get_train_val_dataloaders()

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator=args.device,
        # deterministic=True,
        num_sanity_val_steps=args.num_sanity_val_steps,
        precision="bf16-mixed" if args.bf16 else "16-mixed" if args.fp16 else 32,
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=wandb_logger if args.use_wandb else None,
        callbacks=[lr_monitor],
    )
    trainer.fit(model, train_dataloader, val_dataloader)

    if args.test_at_end:
        # Reinitialize the dataset for testing
        test_dataloader = MedSegDataset(
            processor=model._model.processor,
            dataset_path=args.dataset_path,
            modalities=args.modalities,
            train=False,
        )
        model.set_dataset(test_dataloader, cls_idx=args.cls_idx)
        trainer.test(model, test_dataloader)


if __name__ == "__main__":
    main()
