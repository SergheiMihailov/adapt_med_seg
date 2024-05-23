# SegEVOLution: Enhanced Medical Image Segmentation with Multimodality Learning


Medical image segmentation (MIS) is a significant challenge in the field of computer vision due to several complexities. Firstly, the data comprises various image modalities, such as Computed Tomography (CT), Magnetic Resonance Imaging (MRI), Endoscopy, and Ultrasound (US), each employing fundamentally different techniques. Secondly, these medical images target different parts of the human anatomy, resulting in substantial variations in label space and data distribution. Thirdly, unlike conventional images, acquiring large-scale medical data is challenging due to the high costs of annotation and privacy concerns. This difficulty is exacerbated in the case of volumetric (3D) medical images, which are particularly hard to obtain, store, and annotate, and require significant computational resources for processing (Du et al., 2024). Consequently, developing a universal model that demonstrates robust and consistent performance across the entire MIS domain is exceptionally challenging.

This study focuses on SegVol, a foundation model designed for MIS and pre-trained on CT volumes. Notably, SegVol exhibits promising zero-shot performance on MRI data. Our objective is to evaluate the transferability of SegVol's CT pre-training to MRI data through fine-tuning techniques. Our approach comprises two main strategies: First, we will quantitatively evaluate SegVol's performance under controlled input distribution shifts within the MRI modality. Second, we aim to enhance SegVol's performance in the MRI domain by employing advanced prompts, parameter-efficient fine-tuning, and modality-specific priors. Additionally, we will assess the impact of fine-tuning on SegVol's performance in the CT domain. Broadly, this work seeks to provide insights into the adaptability of MIS models across different medical imaging modalities.

## Background

Recently, several large-scale models have been proposed to address the challenges of universality and robustness in image segmentation across both natural and medical domains. Notably, the Segment Anything Model (SAM) (Kirillov et al., 2023) is a large pre-trained foundation model specifically designed for image segmentation, demonstrating impressive results on various tasks, including segmentation of out-of-distribution samples. However, subsequent studies have revealed that despite SAM’s exceptional performance on natural images, it underperforms on most medical image segmentation (MIS) tasks, such as organ, tumor, and lesion segmentation across CT, MRI, and Ultrasound modalities (Ma et al., 2024; G.-P. Ji et al., 2023; Sheng et al., 2023; W. Ji et al., 2024; Roy et al., n.d.; Y. Ji et al., 2022).

To address these limitations, several new methods have been proposed to adapt SAM for improved performance on MIS tasks (Wu et al., 2023; Cheng et al., 2023; Haoyu Wang et al., 2023). SAM-Med2D, developed by Cheng et al. (2023), is a fine-tuned version of SAM trained on 19.7 million 2D masks from various body parts and imaging techniques. Although SAM-Med2D achieves significant improvements over the pre-trained SAM, treating 3D images such as CT and MRI as independent 2D slices is suboptimal. Haoyu Wang et al. (2023) reformulated SAM into a 3D architecture, called SAM-Med3D, and trained it on 131,000 3D masks across 247 categories. SAM-Med3D generates higher quality masks with significantly fewer point prompts compared to SAM-Med2D. Despite its remarkable performance, SAM-Med3D still faces challenges in processing large inputs (hence the volumetric design) and does not support segmentation using semantic prompts.

Most recently, Du et al. (2024) proposed SegVol, a volumetric model pre-trained on a large collection of CT images from various segmentation datasets. The authors claim that SegVol generalizes remarkably well to unseen data, achieving state-of-the-art zero-shot performance on most MIS tasks. Interestingly, although SegVol was trained explicitly on CT images, it also demonstrates good zero-shot performance in the MRI domain. Additionally, the authors introduced a novel zoom-in-zoom-out method for inference, which significantly reduces the computational cost of volumetric image segmentation while effectively utilizing the 3D structure's information. SegVol also supports semantic prompts, broadening its range of applications.

To develop a truly universal medical image segmentation model, Gao et al. (2024) proposed Hermes, which learns task- and modality-specific priors inspired by the training program of medical radiology residents. Specifically, they trained a pool of "priors," from which Hermes performs context-aware sampling based on the input image's modality (e.g., MRI, CT, PET) and the task description. Segmentation masks are inferred from hidden representations obtained by integrating the input image representation with the learned priors. As a result, Hermes is reported to be competitive with, or even outperform, state-of-the-art task- and modality-specific approaches across a wide range of benchmarks.

## Methodology

### Context-prior learning

Medical segmentation models focus predominantly on fine-tuning to specific modalities. However, this detracts from their generalizability and prevents learning of features useful across modalities. SegVol is a state-of-the-art model for medical image segmentation on the CT modality. Whereas it shows promising zero-shot results on the MRI modality, it underperforms compared to specialized MRI segmentation models. We aim to bring the performance of SegVol to state-of-the-art on the MRI modality, while preserving its performance on the CT modality. To that end we explore learning context-prior tokens for each task and modality (as proposed for medical image segmentation by [Gao et al. 2024](https://arxiv.org/pdf/2306.02416)). The idea is to let the segmentation model learn modality- and task-specific tokens that condition image encoding and mask decoding. These tokens are then fused via a transformer with the image representations from the image encoder, resulting in an updated image representation and updated context-prior tokens. The updated image representations are fed to the mask decoder. An MLP is applied on top of the updated context-prior tokens to generate posterior prototypes. The updated image representation and posterior prototypes are multiplied to obtain the final binary mask predictions. The resulting model is called Hermes. Hermes shows improvements over existing task-specific approaches across tasks and modalities. However, it is inferior to SegVol. Hence, we aim to combine the strengths of SegVol and Hermes to achieve state-of-the-art performance on both the CT and MRI modalities.

### Architecture overview

SegVol | Hermes
:------:|:------:
![SegVol](./assets/SegVol.png) | ![Hermes](./assets/Hermes.png)

Hermes has been shown to be compatible with existing backbones, including ViT, as used in SegVol. SegVol builds on top of Segment Anything (SAM), with image, spatial and semantic embeddings fused and fed to the mask decoder. In case of Hermes, the reference architecture is similar: an image encoder, followed by fusion of image embeddings and context prior tokens, and then followed by a mask decoder.

**Proposed approach to applying SegVol to Hermes:**
We apply the approach taken by Hermes to the pre-trained SegVol model. Following the approach from the Hermes paper, we introduce context priors, a posterior prototype MLP, and add adapters to image encoder, fusion encoder and mask decoder.
Currently, we are still experimenting with the precise architecture that would yield the best results. However, conceptually it looks as follows:
![AdaptMedSeg](./assets/adapt_med_seg.png)

### Datasets

We chose to focus on prostate MRI data, as there are several datasets avaiable amounting to over 400 annotated volumes, and the prostate is a well-defined structure that is easy to segment. We have developed a pre-processing pipeline, and pre-processed MSD, PROMISE12, SAML and T2W datasets in a format compatible with SegVol.

## Results
Based on preliminary evaluation, we have reproduced SegVol performance on CT and MRI data, obtaining results as reporting in the SegVol paper.


## References

<div id="refs" class="references csl-bib-body hanging-indent"
entry-spacing="0">

<div id="ref-sammed2d" class="csl-entry">

Cheng, Junlong, Jin Ye, Zhongying Deng, Jianpin Chen, Tianbin Li, Haoyu
Wang, Yanzhou Su, et al. 2023. “Sam-Med2d.” *arXiv Preprint
arXiv:2308.16184*.

</div>

<div id="ref-ding2022delta" class="csl-entry">

Ding, Ning, Yujia Qin, Guang Yang, Fuchao Wei, Zonghan Yang, Yusheng Su,
Shengding Hu, et al. 2022. “Delta Tuning: A Comprehensive Study of
Parameter Efficient Methods for Pre-Trained Language Models.”
<https://arxiv.org/abs/2203.06904>.

</div>

<div id="ref-segvol" class="csl-entry">

Du, Yuxin, Fan Bai, Tiejun Huang, and Bo Zhao. 2024. “SegVol: Universal
and Interactive Volumetric Medical Image Segmentation.”
<https://arxiv.org/abs/2311.13385>.

</div>

<div id="ref-gao2024training" class="csl-entry">

Gao, Yunhe, Zhuowei Li, Di Liu, Mu Zhou, Shaoting Zhang, and Dimitris N.
Metaxas. 2024. “Training Like a Medical Resident: Context-Prior Learning
Toward Universal Medical Image Segmentation.”
<https://arxiv.org/abs/2306.02416>.

</div>

<div id="ref-hu2021lora" class="csl-entry">

Hu, Edward J., Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi
Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. “LoRA: Low-Rank
Adaptation of Large Language Models.”
<https://arxiv.org/abs/2106.09685>.

</div>

<div id="ref-ji2023sam" class="csl-entry">

Ji, Ge-Peng, Deng-Ping Fan, Peng Xu, Ming-Ming Cheng, Bowen Zhou, and
Luc Van Gool. 2023. “SAM Struggles in Concealed Scenes–Empirical Study
on" Segment Anything".” *arXiv Preprint arXiv:2304.06022*.

</div>

<div id="ref-ji2024segment" class="csl-entry">

Ji, Wei, Jingjing Li, Qi Bi, Tingwei Liu, Wenbo Li, and Li Cheng. 2024.
“Segment Anything Is Not Always Perfect: An Investigation of Sam on
Different Real-World Applications.” Springer.

</div>

<div id="ref-lim2" class="csl-entry">

Ji, Yuanfeng, Haotian Bai, Chongjian Ge, Jie Yang, Ye Zhu, Ruimao Zhang,
Zhen Li, et al. 2022. “Amos: A Large-Scale Abdominal Multi-Organ
Benchmark for Versatile Medical Image Segmentation.” *Advances in Neural
Information Processing Systems* 35: 36722–32.

</div>

<div id="ref-sam" class="csl-entry">

Kirillov, Alexander, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe
Rolland, Laura Gustafson, Tete Xiao, et al. 2023. “Segment Anything.” In
*Proceedings of the IEEE/CVF International Conference on Computer
Vision*, 4015–26.

</div>

<div id="ref-ma2024segment" class="csl-entry">

Ma, Jun, Yuting He, Feifei Li, Lin Han, Chenyu You, and Bo Wang. 2024.
“Segment Anything in Medical Images.” *Nature Communications* 15 (1):
654.

</div>

<div id="ref-peft" class="csl-entry">

Mangrulkar, Sourab, Sylvain Gugger, Lysandre Debut, Younes Belkada,
Sayak Paul, and Benjamin Bossan. 2022. “PEFT: State-of-the-Art
Parameter-Efficient Fine-Tuning Methods.”
<https://github.com/huggingface/peft>.

</div>

<div id="ref-MedicalImageSegmentation2019" class="csl-entry">

Petitjean, Caroline. 2019. “Current Methods in Medical Image
Segmentation.” *Journal of Imaging* 5.

</div>

<div id="ref-lim1" class="csl-entry">

Roy, S, T Wald, G Koehler, MR Rokuss, N Disch, J Holzschuh, D Zimmerer,
KH Maier-Hein, and MD SAM. n.d. “Zero-Shot Medical Image Segmentation
Capabilities of the Segment Anything Model. arXiv 2023.” *arXiv Preprint
arXiv:2304.05396*.

</div>

<div id="ref-he2023accuracy" class="csl-entry">

Sheng, He, Rina Bao, Jingpeng Li, Patricia Grant, and Yangming Ou. 2023.
“Accuracy of Segment-Anything Model (SAM) in Medical Image Segmentation
Tasks,” April.

</div>

<div id="ref-Radiotherapy2023" class="csl-entry">

Valdagni, Riccardo, Marco Montorsi, and Tiziana Rancati. 2023.
“Automatic Segmentation with Deep Learning in Radiotherapy.” *Cancers*
15 (17): 4389.

</div>

<div id="ref-sammed3d" class="csl-entry">

Wang, Haoyu, Sizheng Guo, Jin Ye, Zhongying Deng, Junlong Cheng, Tianbin
Li, Jianpin Chen, et al. 2023. “Sam-Med3d.” *arXiv Preprint
arXiv:2310.15161*.

</div>

<div id="ref-Wang2024" class="csl-entry">

Wang, Huaijun, and Xinhong Hei. 2024. “Automatic Medical Image
Segmentation with Vision Transformer.” *Applied Sciences* 14 (7): 2741.

</div>

<div id="ref-wu2023medical" class="csl-entry">

Wu, Junde, Wei Ji, Yuanpei Liu, Huazhu Fu, Min Xu, Yanwu Xu, and Yueming
Jin. 2023. “Medical SAM Adapter: Adapting Segment Anything Model for
Medical Image Segmentation.” <https://arxiv.org/abs/2304.12620>.

</div>

</div>
