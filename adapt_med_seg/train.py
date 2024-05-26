from adapt_med_seg.data.dataset import MedSegDataset
from adapt_med_seg.models.lightning_model import SegVolLightning
from adapt_med_seg.utils.cli import parse_arguments
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import wandb

api = wandb.Api()


import wandb

api = wandb.Api()


def main():
    args = parse_arguments()
    kwargs = vars(args)

    seed_everything(args.seed)
    model_name = kwargs.pop("model_name")
    modalities = kwargs.pop("modalities")
    model = SegVolLightning(
        model_name=model_name, modalities=modalities, test_mode=False, **kwargs
    )

    _dataset = MedSegDataset(
        processor=model._model.processor,
        dataset_path=args.dataset_path,
        modalities=modalities,
        train=True,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=None,
    )

    model.set_dataset(_dataset)
    train_dataloader, val_dataloader = _dataset.get_train_val_dataloaders(batch_size=args.batch_size)

    loggers = [TensorBoardLogger(args.log_dir)]
    if args.use_wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            save_dir=args.log_dir,
            log_model="all",
            name=f"{model_name}_{args.dataset_path}",
        )
        wandb_logger.watch(model, log="all", log_freq=10)
        loggers.append(wandb_logger)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(monitor="val_dice_score", mode="max")
    early_stopping = EarlyStopping(monitor="val_dice_score", mode="max", patience=5)

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator=args.device,
        # logger=loggers,
        # deterministic=True,
        num_sanity_val_steps=args.num_sanity_val_steps,
        precision="bf16-mixed" if args.bf16 else "16-mixed" if args.fp16 else 32,
        gradient_clip_val=0.5,
        accumulate_grad_batches=args.accumulate_grad_batches,
        # override the default if we have less than 50 samples
        log_every_n_steps=min(args.max_len_samples, 50) if args.max_len_samples else 50,
        callbacks=[
            lr_monitor,
            checkpoint_callback,
            early_stopping,
            StochasticWeightAveraging(swa_lrs=1e-2),
        ],
    )
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=args.ckpt_path)

    if args.test_at_end:
        # Reinitialize the dataset for testing
        test_data = MedSegDataset(
            processor=model._model.processor,
            dataset_path=args.dataset_path,
            modalities=modalities,
            train=False,
            max_train_samples=None,
            max_val_samples=None,
            max_test_samples=args.max_test_samples,
        )
        model.set_dataset(test_data)
        test_dataloader = test_data.get_test_dataloader(batch_size=args.batch_size)
        trainer.test(model, test_dataloader)


if __name__ == "__main__":
    main()
