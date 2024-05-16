from adapt_med_seg.data.dataset import MedSegDataset
from adapt_med_seg.pipelines.train import SegVolLightning, TrainingArgs

from pytorch_lightning import Trainer, seed_everything

seed_everything(42)

model_names = ["segvol_lora"]
dataset_paths = ["datasets/chaos000"]

for model_name in model_names:
    for dataset in dataset_paths:
        training_args = TrainingArgs(
            dataset_number=dataset,
            model_name=model_name,
            cls_idx=0,
            device="cpu",
            # use_wandb=True,
        )

        train_pipeline = SegVolLightning(model_name, test_mode=False)

        _dataset = MedSegDataset(
            processor=train_pipeline._model.processor,
            dataset_path=dataset,
            modalities=["CT"],
            train=True,
        )

        train_pipeline.set_dataset(_dataset)
        train_dataloader, val_dataloader = _dataset.get_train_val_dataloaders()

        trainer = Trainer(
            max_epochs=10,
            accelerator=training_args.device,
            deterministic=True,
            num_sanity_val_steps=1,
        )
        trainer.fit(train_pipeline, train_dataloader, val_dataloader)
        # results = train_pipeline.run()

        # print(f"Results for {model_name} on dataset {dataset_number}:\n{results}")