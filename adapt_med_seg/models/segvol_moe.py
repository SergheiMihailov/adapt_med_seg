from adapt_med_seg.models.segvol_lora import SegVolLoRA


class SegVolMoE(SegVolLoRA):
    """
    Baseline model for SegVol.
    """

    def forward_test(
        self,
        image,
        zoomed_image=None,
        text_prompt=None,
        bbox_prompt_group=None,
        point_prompt_group=None,
        use_zoom=True,
        modality=None,
    ):
        if modality != "1" and modality != 1 and modality != "MRI":
            # CT mode
            with self.model.disable_adapters():
                return super().forward_test(
                    image,
                    zoomed_image,
                    text_prompt,
                    bbox_prompt_group,
                    point_prompt_group,
                    use_zoom,
                    modality,
                )
        else:
            # MRI mode
            return super().forward_test(
                image,
                zoomed_image,
                text_prompt,
                bbox_prompt_group,
                point_prompt_group,
                use_zoom,
                modality,
            )

    def forward_train(self, image, tasks, train_labels, modality):
        if modality != "1" and modality != 1 and modality != "MRI":
            # CT mode
            with self.disable_adapters():
                return super().forward_train(image, tasks, train_labels, modality)
        else:
            # MRI mode
            return super().forward_train(image, tasks, train_labels, modality)
