# Introduction

Medical image segmentation (MIS) is a prevailing problem in the field of
computer vision. The problem is non-trivial for several reasons. First,
the data may come from a multitude of sub-domains of medical images such
as Computerized Tomography (CT), Magnetic Resonance Imaging (MRI),
Endoscopy, and Ultrasound (US). All of these are radically different
techniques. Second, medical images depict different parts of the human
body, so the label space and data distribution may vary significantly as
well. Third, contrary to regular images, is it difficult to obtain large
scales of medical data, mainly due to extremely high annotation costs
and privacy concerns. This is particularly true for volumetric (i.e. 3D)
medical images, which are hard to obtain, store and annotate. Even
processing them usually comes with a high computational cost . For all
these reasons, it is challenging to train a “universal” model, with
robust and consistent performance over the complete MIS domain.

In this work, we will focus on SegVol, a foundation model designed for
MIS and pre-trained on CT volumes. Interestingly, it shows promising
zero-shot results on MRI data. We aim to assess the transferability of
SegVol’s pre-training on CT data to MRI data, beyond zero-shot, by
applying fine-tuning techniques. Our approach is twofold: Firstly, we
intend to quantitatively evaluate SegVol’s performance on controlled
input distribution shifts across different modalities (MRI in
particular), transformations, and image complexities. Secondly, we plan
to improve SegVol’s performance specifically in the MRI domain by
incorporating enhanced prompts, parameter-efficient fine-tuning, and
modality-dependent priors, w hile also assessing the impact of
fine-tuning on SegVol’s CT performance. More broadly, we hope that this
work will provide insights into the adaptability of MIS models across
different medical imaging modalities.

## Motivation

Manual segmentation is both costly and time-consuming . Automatic
segmentation, however, can significantly enhance the efficiency and
effectiveness of patient treatments on a variety of tasks. Examples of
the these are monitoring tumor mass, and observing the growth or
shrinkage of organs or sub-organ structures . Ultimately, by improving
medical segmentation areas such as oncology and neurology are benefited.

Building on this foundation, assessing the practical applications and
robustness of SegVol represents a vital research direction that could
benefit the medical community. One limitation of SegVol is that it is
trained exclusively on CT scan data . To potentially improve SegVol, we
suggest incorporating enhanced prompts and extending its training to
include MRI data. As MRI scans capture more expressive structures, we
hope those will be able to aid the model into more accurate general
segmentation.

# Related Work

Recently, several large-scale models were proposed to tackle the
problems of universality and robustness of image segmentation in both
the natural and the medical domains. Most remarkably, the Segment
Anything Model (SAM) is a large pre-trained foundation model designed
specifically for image segmentation and has shown impressive results on
many tasks, including segmentation over out-of-distribution samples.
Subsequent work has shown that, despite SAM’s unparalleled performance
over non-medical images (or *natural*), it has poor performance on most
MIS tasks .

Several new methods were proposed to overcome this limitation, which
tries to fine-tune SAM () to boost its performance over these tasks.
SAM-Med2D by in particular has shown state-of-the-art performance in
two-dimensional MIS, which has later been extended to the
three-dimensional (or *volumetric*) domain by in their proposed model,
SAM-Med3D. Despite its remarkable performance, SAM-Med3D still struggles
to process large inputs and does not support *semantic* segmentation.

Most recently, has proposed SegVol, which is a foundation model,
pre-trained specifically on a large collection of medical images and
then fine-tuned on several different segmentation datasets. The authors
of claim that SegVol generalizes remarkably well to unseen data and has
state-of-the-art zero-shot performance over most MIS tasks. Most
curiously, SegVol was trained explicitly on Computerized Tomography (CT)
images and the authors show that it has good zero-shot performance over
the Magnetic Resonance Imaging (MRI) domain. Finally, the authors claim
to have used a novel zoom-in-zoom-out method for inference which
significantly reduces the computational cost of image segmentation,
while being able to handle volumetric (i.e. 3D) input and output .

Finally, to train a truly universal medical image segmentation model,
proposes Hermes, an MIS model with learned task- and modality-specific
priors. More precisely, they train a pool of "priors", from which Hermes
performs context-aware sampling, given an input image of some modality
(e.g. MRI, CT, PET, etc.) and some task description. Segmentation masks
are inferred from the hidden representations, obtained by fusing the
input image representation with the learned priors. As a result, Hermes
is reported to be competitive or even outperform state-of-the-art task-
and modality-specific approaches on a wide range of benchmarks.

# Methodology

## Context-prior learning
Medical segmentation models focus predominantly on fine-tuning to specific modalities. However, this detracts from their generalizability and prevents learning of features useful across modalities. SegVol is a state-of-the-art model for medical image segmentation on the CT modality. Whereas it shows promising zero-shot results on the MRI modality, it underperforms compared to specialized MRI segmentation models. We aim to bring the performance of SegVol to state-of-the-art on the MRI modality, while preserving its performance on the CT modality. To that end we explore learning context-prior tokens for each task and modality (as proposed for medical image segmentation by [Gao et al. 2024](https://arxiv.org/pdf/2306.02416)). The idea is to let the segmentation model learn modality- and task-specific tokens that condition image encoding and mask decoding. These tokens are then fused via a transformer with the image representations from the image encoder, resulting in an updated image representation and updated context-prior tokens. The updated image representations are fed to the mask decoder. An MLP is applied on top of the updated context-prior tokens to generate posterior prototypes. The updated image representation and posterior prototypes are multiplied to obtain the final binary mask predictions. The resulting model is called Hermes. Hermes shows improvements over existing task-specific approaches across tasks and modalities. However, it is inferior to SegVol. Hence, we aim to combine the strengths of SegVol and Hermes to achieve state-of-the-art performance on both the CT and MRI modalities.

### Datasets
We chose to focus on prostate MRI data, as there are several datasets avaiable amounting to over 400 annotated volumes, and the prostate is a well-defined structure that is easy to segment. We have developed a pre-processing pipeline, and pre-processed MSD, PROMISE12, SAML and T2W datasets in a manner compatible with SegVol.

### Architecture overview

SegVol | Hermes
:------:|:------:
![SegVol](SegVol.png) | ![Hermes](Hermes.png)

Hermes has been shown to be compatible with existing backbones, including ViT, as used in SegVol. SegVol builds on top of Segment Anything (SAM), with image, spatial and semantic embeddings fused and fed to the mask decoder. In case of Hermes, the reference architecture is similar: an image encoder, followed by fusion of image embeddings and context prior tokens, and then followed by a mask decoder.

**Proposed approach to applying SegVol to Hermes:**
We apply the approach taken by Hermes to the pre-trained SegVol model. Following the approach from the Hermes paper, we introduce context priors, a posterior prototype MLP, and add adapters to image encoder, fusion encoder and mask decoder.
Currently, we are still experimenting with the precise architecture that would yield the best results. However, conceptually it looks as follows:
![AdaptMedSeg](adapt_med_seg.png)

# Results
Based on preliminary evaluation, we have reproduced SegVol performance on CT and MRI data, obtaining results as reporting in the SegVol paper.
