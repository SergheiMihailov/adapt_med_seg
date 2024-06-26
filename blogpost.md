# SegEVOLution: Towards Universal Fine-tuning for Medical Image Segmentation with Context-Prior Learning

### Z. Fülöp, S. Mihailov, M. Krastev, M. Hamar, D.A. Toapanta 
> **Supervised by**: Stefanos Achlatis, s.s.achlatis@uva.nl

## Introduction

Medical image segmentation (MIS) is a major direction in computer vision which targets accurate delineation of anatomical structures -- crucial for diagnosis, treatment planning, and disease monitoring. The development of universal models that can perform well across different medical imaging modalities, in particular, is a challenging task. Firstly, because the modalities, notably Computed Tomography (CT), Magnetic Resonance Imaging (MRI), Endoscopy, and Ultrasound (US), each employ fundamentally different techniques. Secondly, each modality targets different parts and/or features of the human anatomy resulting in substantial variations in label space and data distribution. Thirdly, unlike conventional images, acquiring large-scale medical data is challenging due to the high costs of annotation and privacy concerns. This difficulty is exacerbated in the case of volumetric (3D) medical images, which are particularly hard to obtain, store, and annotate, and require significant computational resources for processing [[1]](#ref1). Consequently, developing a universal model that demonstrates robust and consistent performance across the entire MIS domain is exceptionally challenging.

This study focuses on SegVol [[1]](#ref1), a foundation model designed for MIS and pre-trained on CT volumes. Notably, SegVol exhibits promising zero-shot performance on MRI data. Our objective is to evaluate the transferability of SegVol's CT pre-training to MRI data and extend its capabilities through fine-tuning techniques and enhanced prompt augmentation. Our approach comprises two main strategies: First, we quantitatively evaluate SegVol's performance under controlled input distribution shifts within the MRI modality. Second, we enhance SegVol's performance in the MRI domain by employing advanced prompts, parameter-efficient fine-tuning, and modality-, and task-specific priors. Additionally, we assess the impact of fine-tuning on SegVol's performance in the CT domain. Broadly, this work seeks to provide insights into the adaptability of MIS models across different medical imaging modalities.

## Background

Recently, several large-scale models have been proposed to address the challenges of universality and robustness in image segmentation across both natural and medical domains. Notably, the Segment Anything Model (SAM) [[2]](#ref2) is a large pre-trained foundation model specifically designed for image segmentation, demonstrating impressive results on various tasks, including segmentation of out-of-distribution samples. However, subsequent studies have revealed that despite SAM’s exceptional performance on natural images, it significantly underperforms on most medical image segmentation (MIS) tasks, such as organ, tumor, and lesion segmentation across CT, MRI, and Ultrasound modalities [[3]](#ref3) [[4]](#ref4) [[5]](#ref5) [[6]](#ref6) [[7]](#ref7).

To address these limitations, several new models have been proposed to adapt SAM for improved performance on MIS tasks [[8]](#ref8) [[9]](#ref9) [[10]](#ref10). SAM-Med2D [[9]](#ref9) is a fine-tuned version of SAM trained on 19.7 million 2D masks from various body parts and imaging techniques. This version incorporates learnable adapter layers in each Transformer block, allowing the model to acquire domain-specific knowledge crucial for medical image segmentation. Spatial prompts such as point prompts, bounding box prompts, and mask prompts play crucial roles in guiding the model to specific regions of interest within medical images. Despite the adaptation, treating 3D images such as CT and MRI as independent 2D slices is suboptimal.

Haoyu Wang et al. (2023) [[10]](#ref10) reformulated SAM into a 3D architecture, called SAM-Med3D, and trained it on 131,000 3D CT and MRI masks across 247 categories. Unlike SAM-Med2D, which treats volumetric data as individual 2D slices, SAM-Med3D processes the data in its entirety using a 3D decoder. This method allows SAM-Med3D to capture more spatial context and generate higher quality masks with significantly fewer point prompts compared to SAM-Med2D. However, SAM-Med3D still faces challenges in processing large inputs due to its volumetric design and does not support segmentation using semantic prompts.

Most recently, Du et al. (2024) proposed SegVol [[1]](#ref1), a volumetric model pre-trained on 90,000 unlabelled, and fine-tuned on 6,000 labeled CT images from various segmentation datasets of over 200 anatomical structures. The authors claim that SegVol generalizes remarkably well to unseen data (external validation tasks), achieving state-of-the-art zero-shot performance on most MIS tasks. SegVol employs composite-type prompts that combine semantic and spatial information, significantly enhancing its segmentation accuracy. Importantly, SegVol also supports semantic-only prompt which enables a wider range of applications. Additionally, the authors introduced a method for inference called zoom-in-zoom-out, which significantly reduces the computational cost of volumetric image segmentation while effectively utilizing the 3D structure's information. Despite SegVol being explicitly trained on CT images, it demonstrates good zero-shot performance in the MRI domain, underscoring its versatility.

To develop a truly universal medical image segmentation model, Gao et al. (2024) proposed Hermes [[11]](#ref11), which learns task- and modality-specific priors inspired by the training program of medical radiology residents. Hermes integrates these priors through context-aware sampling [[12]](#ref12) based on the input image's modality (e.g., MRI, CT, PET) and the task description (essentially just the name of the anatomical structure that we want to segment). This approach allows Hermes to adapt dynamically to different segmentation challenges, offering a significant improvement over single-task models. Contextual prompts derived from the learned priors are used to adapt the model’s segmentation strategy dynamically. Hermes has been shown to be competitive with, or even outperform, state-of-the-art task- and modality-specific approaches across a wide range of benchmarks.

## Overview of SegVol

In this section, we will briefly introduce the design and architecture of SegVol as well as a description of the M3D-Seg dataset [[33]](#ref), used for fine-tuning the model by the authors. Next, we show some preliminary experiments we conducted to verify the performance of this model in a variety of aspects. Finally, we summarise our findings and further motivate our work.

### Architecture

<table name='fig1' align="center">
  <tr align="center">
    <td>
    	<img src="assets/SegVol.png" alt="SegVol" width=700px />
  	</td>
  </tr>
  <tr align="left">
    <td>
    	<b>Figure 1</b>: Architecture overview of the SegVol model
  	</td>
  </tr>
</table>

The SegVol model takes inspiration from the Segment Anything Model (SAM) [[1]](#ref1),[[2]](#ref2) in its architecture. Concretely, it consists of the following main parts (also see [Figure 1](#fig1)):

- **Vision Transformer (ViT)**: responsible for computing powerful representations of the input image.
- **Prompt Encoder (PE)**: responsible for mapping different types of prompts to the same vector space as the output representations of the ViT. The supported prompt types are the following:
  - **Text prompt**: encodes semantic information about the task at hand. Given a task (e.g. liver segmentation), it uses the pre-trained text encoder of the CLIP model [[34]](#ref34), evaluated using the template 

  		`A Computerized Tomography (CT) of a {ANATOMICAL_STRUCTURE} ` (e.g. liver)
  -  **Point prompt**: specify $n$ points within the organ to help guide the search of the model. Following CLIP, the model computes the positional encoding of these points
  - **Bounding box prompt**: specify a 3D box around the target organ to help guide the search of the model. The positional encodings of the corners of the bounding box are used.

	Overall, the prompt encoder computes representations for each of the provided prompt types and concatenates them.

- **Fusion Encoder**: a lightweight sequential application of two transformer blocks, applying bi-directional self-attention on the concatenated input of the image- and prompt embeddings computed by the earlier modules.
- **Mask Decoder**: Based on the output of the fusion encoder, compute mask predictions using a Multi-Layer Perceptron (MLP) block. These predictions are then used in a standard sliding window inference to find the mask with highest *Intersection over Union (IoU)* score.

**Zoom-out-zoom-in mechanism**:  Given that 3D medical images typically have very high resolution[^1], and naively down-sampling them would cause significant information loss, the authors of SegVol employ a so-called zoom-out-zoom-in mechanism to reduce the memory overhead at inference time. Their method is simple: given an input image and a bbox or point prompt, they produce two inputs to the model; one, which is a downsampled version of the input (zoom-out) and another which is a full resolution but cropped using the provided prompt (zoom-in). This way, instead of having to compute image representations for the whole input, the model can first produce a local representation of the part deemed relevant by the provided prompt, and another, which helps put this representation in the context of the whole image. As a result, the computation overhead significantly decreases[^2].

Despite the obvious benefits of this method, it is important to note that at test time, it is our understanding that the bounding box prompts were generated from the ground truth labels, which makes the zoom-in images always *perfectly aligned* with the target organ. This may indeed leak ground-truth information to the model at inference time and obstruct the reported test results. To investigate this, we performed some preliminary experiments by applying random translations to the generated bounding box prompts, and found that the performance indeed decreases significantly. For more details, please refer to the [Results](#results) section. 

**Training**:  The SegVol model was trained in two phases; *pre-training* and *fine-tuning*. First, the ViT was pre-trained on a large training corpus, consisting of $96\, 000$ unlabelled volumetric CT images. During this phase, the SimMIM algorithm [[22]](#ref22) was used to obtain a weak supervision signal and guide the image encoder to map to a feature-rich embedding space, tailored specifically for the task of image segmentation. Next, once the pre-training of the ViT concluded, the authors employed supervised fine-tuning of the entire model on a set of $6\, 000$ labelled CT images, including $150\,000$ ground truth segmentation masks, from the M3D-Seg dataset [[33]](#ref33)(see the [next section](#datasets)).

Overall, the above architecture is a well-defined extension of the SAM architecture, adapted specifically for the task of volumetric medical image segmentation. While SAM was shown to perform poorly in the medical domain ([4](#ref4), [5](#ref5), [6](#ref6), [7](#ref7)), SegVol consistently out-performs other state-of-the-art methods. By employing zoom-out-zoom-in inference and also through its design, it is not infeasible to perform interactive segmentation in a low-resource environment, paving the way for medical practitioners to use it in their day-to-day activities.

[^1]: e.g. a typical CT volume has dimensions $(256\times 256\times200)$ which amounts to $13\,107\,200$ voxels, in 32 bit floating point representation, this takes $~250\text{MiB}$​.
[^2]: suppose the target organ takes up $50\%$ of the whole image (an over-approximation in our experience), then the zoom-in image size is $1/2$ of the original input and the zoom-out can also be down-sampled by $50\%$. This, combined with the $O(n^2)$ asymptotic running time of the self-attention mechanism, leads to a quadratic increase in performance.

## Datasets

In our work, we consider two *modalities* from the volumetric medical image segmentation domain: Computed Tomography (CT) and Magnetic Resonance Imaging (MRI), and employ different adaptation methods to improve performance on different modalities and tasks. For this reason, we re-used part of the M3D-Seg dataset, released by Bai et. al 2024 [[33]](#ref33), and extended it using 6 public MRI segmentation datasets for training. Please refer to [Table 1](#tab1) for a comprehensive summary.

### Computed Tomography Data

We reused the M3D-Seg dataset, used in the fine-tuning phase of the SegVol model [[1]](#ref1). It was released by [[33]](#ref33), and is currently one of the largest volumetric image segmentation datasets available[^3]. It consists of $5\,771$ CT images, along with $149\,000$ segmentation masks, from a total of $25$ publicly available datasets. The authors made an effort to standardize the data across a wide range of data quality, format and sample sizes. Interestingly, the preprocessing pipeline used is relatively simple compared to common practices in the medical image segmentation domain. 

Concretely, given a raw volumetric image file (typically in [DICOM format](https://en.wikipedia.org/wiki/DICOM) or semi-processed [NIfTI format](https://en.wikipedia.org/wiki/Neuroimaging_Informatics_Technology_Initiative#:~:text=The%20Neuroimaging%20Informatics%20Technology%20Initiative,using%20Magnetic%20Resonance%20Imaging%20methods.)), they first extract the foreground of the image (all voxels of intensity $>$ mean intensity), take the resulting image's $5^\text{th}$ and $95^\text{th}$ percentile, and standardize it using the mean and std of the foreground. Next, for each dataset, they pre-process the ground truth segmentation masks by splitting the different segmentation categories along the first dimension and then concatenating them, to obtain a $K\times H\times W\times D$ tensor of segmentation masks. Finally, they store the resulting image and mask representations as numpy binaries encoded with floating point numbers with 32bit precision.

Here we note, that we discovered that M3D-Seg mistakenly also incorporates in total 32 MRI volumes from the AMOS dataset [[27]](#ref27). We investigated the cause of this and found that the original AMOS dataset has ambiguous metadata files. A more thourough explanation of it can be found in the Appendix A. We estimate that 32 samples from 96,000 should not influence the model much so we consider as if it did not happen. 

### Magnetic Resonance Imaging Data

To successfully train our models to recognise MRI data, we selected $6$ different publicly available datasets, which consist of $1\,682$ MRI volumes and over $12\,000$ ground truth segmentation masks. Please refer to [Table 1](#tab1) for an overview of the datasets we have used. We pre-process each of these samples in the same way as M3D-Seg but also include a per-sample modality information. For example, the CHAOS [[24]](#ref24) dataset contains both CT and MRI data, so it is important to be able to distinguish between them in our dataset representation. 

<center>

| Dataset                                                      | Segmentations                                               | MRI        | CT   |
| ------------------------------------------------------------ | ----------------------------------------------------------- | ---------- | ---- |
| CHAOS [[24]](#ref24)                                         | liver, (left and right) kidneys, spleen                     | 30 (x2)    | 20   |
| M3D-Seg 0001 (HaN-Seg) [[33]](#ref33), [[26]](#ref26)        | 29 different organs and tissues from the head and neck area | -          | 43   |
| M3D-Seg 0008 (Pancreas-CT) [[33]](#ref33), [[23]](#ref23)    | Pancreas                                                    | -          | 83   |
| M3D-Seg 0020 (MSD-Liver) [[33]](#ref33), [[25]](#ref25)      | Liver, tumor                                                | -          | 132  |
| AMOS 2022 [[27]](#ref27)                                     | 15 different abdominal organs                               | 59         | 300  |
| BRaTS 2021 Task 1 [[28]](#ref28)                             | 3 categories of brain tumor tissue                          | 1 251 (x4) | \-   |
| MSD\_Prostate [[25]](#ref25)                                 | prostate core and surrounding tissue                        | 32         | \-   |
| PROMISE-12 [[29]](#ref29)                                    | prostate                                                    | 80         | \-   |
| SAML dataset [[30]](#ref30)                                  | prostate                                                    | 116        | \-   |
| T2 Weigthted MRI Prostate Dataset [[31]](#ref31) [[32]](#ref32) | prostate                                                    | 114        | \-   |

</center>
<table name='tab1'>
<tr>
<td colspan="4"><b>Table 1.</b> Datasets used in our experiments. We re-used four sub-sets of the M3D-Seg dataset and pre-processed an additional $6$ datasets to obtain MRI data of comparable size. The number of CT and MRI samples per dataset can be seen on the last two columns. In the CHAOS datasets, each case corresponds to two volumetric images, one obtained from the In-Phase and one from the Out-Phase of the MRI machine (hence the x2 multiplier). Similarly, in the BRaTS dataset, the raw output of the MRI machines was later processed using four different techniques; namely (i) native T1-weighted, (ii) post contrast T1-Weighted, (iii) T2-Weighted and (iv) T2 Fluid Attenuated Inversion Recovery methods. For each case in the provided dataset, there are four associated volumes available (hence the x4 multiplier).
</tr></table>


[^3]: https://huggingface.co/datasets/GoodBaiBai88/M3D-Seg

## Methodology

### Low Rank Adaptation (LoRA)

LoRA (Low-Rank Adaptation) is a technique used to adapt large pre-trained models to downstream tasks without significantly increasing computational requirements [[13]](#ref13). LoRA reduces the training time and memory footprint of training large models by decomposing the updates of the model's weights into low-rank components.
In this study, we use LoRA to adapt SegVol (initially trained on CT volumes) to directly improve its performance on MRI data by fine-tuning it on a mixture of CT and MRI volumes. An alternative approach, which due to time and computation constraints, we could not test is explained in the Appendix B.


### Context-prior learning
Medical imaging data is characteristically heterogeneous and diverse: different organs have completely different shapes and positions, and image features vary wildly across modalities, such as between CT and MRI. Thus, MIS models are primarily designed for specific tasks and modalities. The downside of such an approach is that models are limited in their domain, and not exploit similarities across tasks and modalities. However, improving model generalizability requires adapting architectures to exploit synergies between tasks and modalities.

Gao et. al 2024 [[11]](#ref11) explore the universal medical image segmentation paradigm. They propose Hermes, an approach for effectively training MIS models across tasks and modalities. Hermes introduces learnable tokens, called **context priors**, into existing MIS architectures. Context priors are used to adapt the image representation input to the mask decoder, as well as the feature map output by the mask decoder, for the modality and task of the given image. 

We apply the approach taken by Hermes to the pre-trained SegVol model, with slight modifications, as described below. You can refer to the [Figure 2](#fig2) above for an overview. We note the code for Hermes was not made available, and our implementation follows the architecture proposed in the paper. 

**Image representation adaptation.** To adapt the output of the image encoder, The context prior $p_i$ tokens and the image representation $X$ through a transformer block, called **prior fusion**. Prior fusion outputs an adapted image representation $X'$ and an updated context prior $p_i'$. The mask decoder uses the adapted image representation $X'$, instead of the original one.

**Mask decoder output adaptation.** The mask decoder produces output tokens, which are then upscaled to produce the masks themselves. To perform mask decoder output adaptation, we use the updated context priors $p_i'$ to generate **posterior prototype** tokens, by applying a multi-layer perceptron (MLP). In our work, we use posterior prototype tokens as additive adaptations to the mask decoder output tokens. This is in contrast to applying posterior prototypes to the generated masks in a multiplicative fashion, as described in Gao et. al 2024 [[11]](#ref11). We achieved subpar results using the latter, whereas adapting over mask decoder output tokens proved to be effective, showing improvements in model performance. Moreover, the norm additive adaptations accounted for 4-7% (varying per task) of the norm of the mask decoder output tokens, showing that mask decoder token adaptation had a considerable impact on the final output. For a deeper explanation of the original context-prior method please see the original paper [[11]](#ref11).

**Fine-tuning the SegVol image encoder and mask decoder.** In Hermes, the task and modality priors are learned together with the backbone. Since SegVol is pre-trained, we fine-tune its image encoder and mask decoder using LoRA for better integration with the Hermes architecture. 

**Modality prediction loss.** Unlike in Hermes, we do not implement the auxiliary modality prediction loss, since it was shown, in the original paper, to only yield a minor improvement.

<table align="center" name="fig2">
  <tr align="center">
      <td><img src="./assets/adapt_med_seg.png"></td>
  </tr>
  <tr align="left">
    <td colspan="2"><b>Figure 2.</b> Proposed architecture combining SegVol model and Hermes context-prior framework. This hybrid model integrates SegVol’s volumetric segmentation with Hermes’s context-prior learning.</td>
  </tr>
</table>

## Experimental Setup

Earlier in this work, we briefly introduced the SegVol model, and proposed two different adaptation methods. In this and the results sections, we will discuss our approach to evaluating their performance and show our findings. Our experimental setup can be decomposed into three separte parts. 

1. We conducted a series of experiments on the SegVol model, using its original weights. This way, on the one hand, we gained valuable insights into the inner workings of the SegVol model, and on the other hand obtained baseline results to which we could compare our adaptation methods.

2. We fine-tuned the SegVol model using LoRA adapters [[13]](#ref13) on our selection of CT and MRI data shown in [Table 1](#tab1). In our expectation, this method would adapt the model to the MRI domain and out-perform the baseline in all our tests. On the other hand, we also expect its performance to decrease over the original (CT) domain of the base model, as is a common phenomenon in fine-tuning methods and also a known phenomenon when using Low-rank adaptation [[35]](#ref35).

3. To overcome the performance drop of the LoRA-adapted model, we introduce Context-prior pooling to the SegVol model, as described in length in the [previous section](#context_prior_learning). We train it on a diverse set of both MRI and CT volumes (as listed in [Table 1](#tab1).

### Evaluation metrics

In all our experiments, we compute the average Dice Similarity Coefficient (dice score) over the different modalities and tasks as well as the average dice score over all samples in the testing data. More precisely, the dice score is computed as follows:
$$\text{dice}(X,Y)=\frac{2\vert X\cap Y\vert}{\vert X\vert + \vert Y\vert}$$
where $\vert X\cup Y\vert$ denotes the cardinality of the intersection of the predicted masks $X$, and the ground truth $Y$. The denominator serves as a normalisation term.

<table align="center" name="fig3">
  <tr align="center">
      <td><img src="./assets/dice_metric.png" width=600px></td>
  </tr>
  <tr align="left">
    <td colspan="2"><b>Figure 3.</b> Illustration of the Dice score metric, showing the overlap between the ground truth mask (<i>Y</i>) and the predicted mask (<i>X</i>).</td>
  </tr>
</table>

## Results
<center>

<!-- | **Method**       | **Training Data**                         | **Expected Outcome**                                    | **Results**                          |
|---------------------------|-------------------------------------------|--------------------------------------------------------|-------------------------------------|
| **SegVol Baseline**       | MRI + CT (all)                            | Baseline performance for comparison                    | 0.729                            |
| **LoRA**                  | MRI + CT (all)                            | Baseline performance for comparison                                  | TBD                                 |
|                           | CT (all)                                  | Baseline performance on CT                             | TBD                                 |
|                           | MRI (all)                                 | Baseline performance on MRI                            | TBD                                 |
|                           | MRI Prostate                              | Performance on MRI prostate                            | TBD                                 |
|                           | MRI Brain                                 | Performance on MRI brain                               | TBD                                 |
| **Context Priors**        | MRI + CT (all)                                   | Baseline performance for comparison            | TBD                                 |
|         | CT (all)                                  | Expected to perform better than LoRA on CT             | TBD                                 |
|                           | MRI (all)                                 | Expected to perform better than LoRA on MRI            | TBD                                 |
|                           | MRI Prostate                              | Potential improvement over LoRA on MRI prostate        | TBD                                 |
|                           | MRI Brain                                 | Potential improvement over LoRA on MRI brain           | TBD                                 | -->

|              | Modality   |   Prostate |   Enhancing tumor |   Non-contrast-enhancing tumor core |      Edema |   Right kidney |    Liver |      Aorta |   Left adrenal gland |   Pancreas |
|:-------------|:----------|-----------:|------------------:|------------------------------------:|-----------:|---------------:|---------:|-----------:|---------------------:|-----------:|
| Baseline     | CT        |            |                   |                                     |            |  | 0.949756 |            |                      |   0.751137 |
| Baseline     | MRI       |   0.382315 |          0.236381 |                            0.154182 |   0.405408 |       0.904191 | 0.725064 |   0.899949 |             0.485659 |   0.671785 |
| LoRA         | CT        |            |                   |                                     |            |                | 0.958793 |            |                      |   0.77236  |
| LoRA         | MRI       |   0.768038 |          0.343241 |                                     |            |       0.904617 | 0.93647  |   0.839851 |             0.618322 |   0.735085 |
| SegVol-Prior | CT        |            |                   |                                     |            |                | 0.958149 |            |                      |   0.765545 |
| SegVol-Prior | MRI       |   0.797154 |          0.473366 |                                     |   0.282421 |       0.903139 | 0.936724 |   0.892555 |             0.58574  |   0.756111 |



|              | Modality   |   Gall bladder |    Stomach |   Duodenum |   Left kidney |   Postcava |   Spleen |   Right adrenal gland |   Colon cancer |     Tumour |
|:-------------|:----------|---------------:|-----------:|-----------:|--------------:|-----------:|---------:|----------------------:|---------------:|-----------:|
| Baseline     | CT        |                |            |            |               |            | 0.964386 |                       |       0.517814 |   0.560809 |
| Baseline     | MRI       |       0.537233 |   0.707523 |   0.615844 |      0.781269 |   0.778236 | 0.775074 |              0.430416 |                |            |
| LoRA         | CT        |                |            |            |               |            | 0.962802 |                       |       0.575204 |   0.567885 |
| LoRA         | MRI       |       0.494653 |   0.817483 |   0.568297 |      0.871564 |   0.763725 | 0.92573  |              0.559679 |                |            |
| SegVol-Prior | CT        |                |            |            |               |            | 0.962918 |                       |       0.678117 |   0.575005 |
| SegVol-Prior | MRI       |       0.484843 |   0.829583 |   0.569049 |      0.876734 |   0.761132 | 0.770932 |              0.553968 |                |            |


</center>

<table name='tab2'>
<tr>
<td colspan="4"><b>Table 2.</b> Internal Validation Results of LoRA and Context Priors on MRI and CT Datasets.
</tr></table>
*Note: The results are presented as mean Dice scores.*




### Performance of SegEVOLution
<table align="center">
  <tr align="center">
    <td>
      <img src="./assets/results_400_ct.jpeg" width=400px>
    </td>
    <td>
      <img src="./assets/results_400_mri.png" width=400px>
    </td>
  </tr>
  <tr align="left">
    <td colspan="2"><b>Figure 4.</b> Combined view of the results.</td>
  </tr>
</table>


## Ablation studies

As we mentioned in the [overview of SegVol](#overview_of_segvol), at test time, the bounding boxes were generated from the ground truth labels. As a result, we wanted to quantify the impact of small perturbations to the bounding boxes would ental on the overall performance of the model. Futhermore, Du et al. (2024) [[1]](#ref1) show ablation studies on different combinations of propmt types, namely text, point and bounding box promts. Most importantly, they show that bounding box prompts combined with textual information typically out-perform other combinations. In our initial experiments, we observed some deviation from test results in the case of text-only prompts and prompts involving bounding boxes. Figure [[5]](#fig5) shows our results.

<table align="center">
  <tr align="center">
    <td>
      <img src="./assets/prompt_type_performance.png" width=400px>
    </td>
    <td>
      <img src="./assets/bbox_performance.png" width=400px>
    </td>
  </tr>
  <tr align="left">
    <td colspan="2"><b>Figure 5.</b> Performance of different models on different prompt types (left) and performance of the same models on different perturbations of the bounding boxes. All models were evaluated on a subset of our multi-modal training data, so the baseline performance is expected to be less than the fine-tuned models. Regardless, we can see that the fine-tuned models are more robust to different prompt types and also over bounding box perturbations.</td>
  </tr>
</table>




## Discussion and Conclusion

We enhance SegVol's performance on MRI data using Low-Rank Adaptation (LoRA), and Context-prior learning. Preliminary experiments establish a robust baseline for the SegVol model across CT scans. The application of LoRA adapters significantly improves the base performance of SegVol on MRI data while maintaining strong performance on CT data, even slightly improving on colon cancer.   

We introduce Context-prior learning to the SegVol model. Beyond semantic prompting, we introduce additional conditioning on task and modality, by fusing the image representation with the context-prior tokens and adapting mask decoder output tokens. The resulting model maintains performance on the CT domain and shows large improvements on the out-of-distribution MRI modality, while fine-tuned on only 400 samples.

These promising results show opportunity for future work on increasing the diversity and size of the training data, to fully exploit the potential of context-priors. A limitation of our approach is that we do not conduct extensive ablations on the architecture, primarily following the Hermes approach. Further architecture search and ablation studies, as well as application to other models, can provide insights into what makes context-prior learning effective in the medical image segmentation setting.

## Invidual Contributions

| **Name**       | **Contribution**                                                                                   |
|----------------|-----------------------------------------------------------------------------------------------------|
| Z. Fülöp       | Literature review, initial experiments with SegVol, training and evaluation, report |
| S. Mihailov    | Literature review, context-prior approach and implementation, backbone and initial evaluation pipeline, training, report |
| M. Krastev     | Literature review, implementation of the backbone of the project and training, visualizations, report |
| M. Hamar       | Literature review, dataset selection, processing and loading, evaluation pipeline, visualizations, report |
| D.A. Toapanta  | Literature review, implementation of LoRA-based approach, training, report |

## References

<a name="ref1">[1]</a>: Du, Yuxin, Fan Bai, Tiejun Huang, and Bo Zhao. 2024. “SegVol: Universal
and Interactive Volumetric Medical Image Segmentation.”
<https://arxiv.org/abs/2311.13385>.

<a name="ref2">[2]</a>: Kirillov, Alexander, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe
Rolland, Laura Gustafson, Tete Xiao, et al. 2023. “Segment Anything.” In
*Proceedings of the IEEE/CVF International Conference on Computer
Vision*, 4015–26.

<a name="ref3">[3]</a>: Ma, Jun, Yuting He, Feifei Li, Lin Han, Chenyu You, and Bo Wang. 2024.
“Segment Anything in Medical Images.” *Nature Communications* 15 (1):
654.

<a name="ref4">[4]</a>: Ji, Ge-Peng, Deng-Ping Fan, Peng Xu, Ming-Ming Cheng, Bowen Zhou, and
Luc Van Gool. 2023. “SAM Struggles in Concealed Scenes–Empirical Study
on" Segment Anything".” *arXiv Preprint arXiv:2304.06022*.

<a name="ref5">[5]</a>: Sheng, He, Rina Bao, Jingpeng Li, Patricia Grant, and Yangming Ou. 2023.
“Accuracy of Segment-Anything Model (SAM) in Medical Image Segmentation
Tasks,” April.

<a name="ref6">[6]</a>: Ji, Wei, Jingjing Li, Qi Bi, Tingwei Liu, Wenbo Li, and Li Cheng. 2024.
“Segment Anything Is Not Always Perfect: An Investigation of Sam on
Different Real-World Applications.” Springer.

<a name="ref7">[7]</a>: Roy, S, T Wald, G Koehler, MR Rokuss, N Disch, J Holzschuh, D Zimmerer,
KH Maier-Hein, and MD SAM. n.d. “Zero-Shot Medical Image Segmentation
Capabilities of the Segment Anything Model. arXiv 2023.” *arXiv Preprint
arXiv:2304.05396*.

<a name="ref8">[8]</a>: Wu, Junde, Wei Ji, Yuanpei Liu, Huazhu Fu, Min Xu, Yanwu Xu, and Yueming
Jin. 2023. “Medical SAM Adapter: Adapting Segment Anything Model for
Medical Image Segmentation.” <https://arxiv.org/abs/2304.12620>.

<a name="ref9">[9]</a>: Cheng, Junlong, Jin Ye, Zhongying Deng, Jianpin Chen, Tianbin Li, Haoyu
Wang, Yanzhou Su, et al. 2023. “Sam-Med2d.” *arXiv Preprint
arXiv:2308.16184*.

<a name="ref10">[10]</a>: Wang, Haoyu, Sizheng Guo, Jin Ye, Zhongying Deng, Junlong Cheng, Tianbin
Li, Jianpin Chen, et al. 2023. “Sam-Med3d.” *arXiv Preprint
arXiv:2310.15161*.

<a name="ref11">[11]</a>: Gao, Yunhe, Zhuowei Li, Di Liu, Mu Zhou, Shaoting Zhang, and Dimitris N.
Metaxas. 2024. “Training Like a Medical Resident: Context-Prior Learning
Toward Universal Medical Image Segmentation.”
<https://arxiv.org/abs/2306.02416>.

<a name="ref12">[12]</a>: Xue Wang and Zhanshan Li and Yongping Huang and Yingying Jiao, 2022, Multimodal medical image segmentation using multi-scale context-aware network

<a name="ref13">[13]</a>: Hu, Edward J., Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. “LoRA: Low-Rank Adaptation of Large Language Models.” *arXiv preprint arXiv:2106.09685*. <https://arxiv.org/abs/2106.09685>.

<a name="ref14">[14]</a>: Shazeer, Noam, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. 2017. “Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.” In *International Conference on Learning Representations*. <https://arxiv.org/abs/1701.06538>.

<a id="ref15">[15]</a>: Geert Litjens, Bram van Ginneken, Henkjan Huisman, Wendy van de Ven, Caroline Hoeks, Dean Barratt, and Anant Madabhushi. “PROMISE12: Data from the MICCAI Grand Challenge: Prostate MR Image Segmentation 2012”. Medical Image Analysis. Zenodo, June 7, 2023. https://doi.org/10.5281/zenodo.8026660

<a id="ref16">[16]</a>: Sebastian Gibala, Rafal Obuchowicz, Julia Lasek, Zofia Schneider, Adam Piorkowski, Elzbieta Pociask, & Karolina Nurzynska. (2023). Prostate MRI T2-weighted images with peripherial and trasition zone segmentations including corresponding PIRADS and PSA values [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7676958


<a id="ref17">[17]</a>: U.Baid, et al., "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification", arXiv:2107.02314, 2021(opens in a new window).

<a id="ref18">[18]</a> B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694 (opens in a new window)

<a id="ref19">[19]</a> S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117

<a id="ref20">[20]</a> Antonelli, M., Reinke, A., Bakas, S. et al. The Medical Segmentation Decathlon. Nat Commun 13, 4128 (2022). https://doi.org/10.1038/s41467-022-30695-9

<a id="ref21">[21]</a> Quande Liu, Qi Dou, Pheng Ann, Heng. Shape-aware Meta-learning for Generalizing Prostate MRI Segmentation to Unseen Domains. International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI). (2020).

<a name="ref22">[22]</a>: Xie, Zhenda, et al. “SimMIM: A Simple Framework for Masked Image Modeling.” 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2022. Crossref, https://doi.org/10.1109/cvpr52688.2022.00943.

<a name="ref23">[23]</a>: Roth, H., Farag, A., Turkbey, E. B., Lu, L., Liu, J., & Summers, R. M. (2016). Data From Pancreas-CT (Version 2) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU

<a name='ref24'>[24]</a>: A.E. Kavur, N.S. Gezer, M. Barış, S. Aslan, P.-H. Conze, et al. "CHAOS Challenge - combined (CT-MR) Healthy Abdominal Organ Segmentation", Medical Image Analysis, Volume 69, 2021. https://doi.org/10.1016/j.media.2020.101950 

<a name='ref25'>[25]</a>: Antonelli, M., Reinke, A., Bakas, S. *et al.* The Medical Segmentation Decathlon. *Nat Commun* **13**, 4128 (2022). https://doi.org/10.1038/s41467-022-30695-9

<a name='ref26'>[26]</a>: Podobnik G, Strojan P, Peterlin P, Ibragimov B, Vrtovec T. HaN-Seg: The head and neck organ-at-risk CT and MR segmentation dataset. *Med Phys*. 2023; 50: 1917–1927. https://doi.org/10.1002/mp.16197

<a name='ref27'>[27]</a>: Ji, Yuanfeng and Bai, Haotian and Yang, Jie and Ge, Chongjian and Zhu, Ye and Zhang, Ruimao and Li, Zhen and Zhang, Lingyan and Ma, Wanling and Wan, Xiang and others. 2022. AMOS: A Large-Scale Abdominal Multi-Organ Benchmark for Versatile Medical Image Segmentation. *arXiv preprint arXiv:2206.08023* 

<a name='ref28'>[28]</a>:  U.Baid, et al., The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification, arXiv:2107.02314, 2021.

<a name='ref29'>[29]</a>: Geert Litjens, Bram van Ginneken, Henkjan Huisman, Wendy van de Ven, Caroline Hoeks, Dean Barratt, & Anant Madabhushi. (2023). PROMISE12: Data from the MICCAI Grand Challenge: Prostate MR Image Segmentation 2012 [Data set]. In Medical Image Analysis Zenodo. https://doi.org/10.1016/j.media.2013.12.002.

<a name='ref30'>[30]</a>: Liu, Quande and Dou, Qi and Heng, Pheng-Ann. 2020. Shape-aware Meta-learning for Generalizing Prostate MRI Segmentation to Unseen Domains. *International Conference on Medical Image Computing and Computer Assisted Intervention*

<a name='ref31'>[31]</a>: Gibala, S.; Obuchowicz, R.; Lasek, J.; Schneider, Z.; Piorkowski, A.; Pociask, E.; Nurzynska, K. Textural Features of MR Images Correlate with an Increased Risk of Clinically Significant Cancer in Patients with High PSA Levels. *J. Clin. Med.* **2023**, *12*, 2836. https://doi.org/10.3390/jcm12082836

<a name='ref32'>[32]</a>:Gibała, S.; Obuchowicz, R.; Lasek, J.; Piórkowski, A.; Nurzynska, K. Textural Analysis Supports Prostate MR Diagnosis in PIRADS Protocol. *Appl. Sci.* **2023**, *13*, 9871. https://doi.org/10.3390/app13179871

<a name='ref33'>[33]</a>: Fan Bai and Yuxin Du and Tiejun Huang and Max Q. -H. Meng and Bo Zhao. 2024. M3D: Advancing 3D Medical Image Analysis with Multi-Modal Large Language Models. *arXiv preprint* [ arXiv:2404.00578](https://arxiv.org/abs/2404.00578)

<a name='ref34'>[34]</a>: Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever, 2021, Learning Transferable Visual Models From Natural Language Supervision, 


<a name='ref35'>[35]</a>: Dan Biderman, Jose Gonzalez Ortiz, Jacob Portes, Mansheej Paul, Philip Greengard, Connor Jennings, Daniel King, Sam Havens, Vitaliy Chiley, Jonathan Frankle, Cody Blakeney, and John P. Cunningham. 2024. "LoRA Learns Less and Forgets Less." [arXiv:2405.09673](https://arxiv.org/abs/2405.09673)

<a name='ref36'>[36]</a>: Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. "Efficient Estimation of Word Representations in Vector Space." [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)


### Appendix A: MRI leakage in M3D-Seg

The AMOS dataset is available at https://zenodo.org/records/7262581. The file `labeled_data_meta_0000_0599.csv` contains the imaging machines used to create the volumes. There are 500 volumes listed with a CT machine and 100 volumes with MRI machines. However, there is also a `dataset.json` which lists all 600 volumes as CT images. In real, the following are MRI volumes: `[541,542,559,563,564,565,566,568,573,576,578,597,543,544,545,546,548,549,551,552,555,558,560,561,562,567,571,572,574,575,580,583,599,550,554,569,585,586,592,594,598,547,553,556,557,577,584,587,589,591,593,595,570,579,581,582,588,590,596,513,505,510,511,512,514,515,516,517,501,518,519,522,523,524,525,526,503,527,528,504,530,532,533,534,535,536,502,500,520,538,506,539,507,508,537,509,521,531,529,540]`
Having this inconsistency in the metadata files, the authors of the M3D-Seg dataset included in total 32 MRI volumes labelled as CT volumes in the training split, namely the following ones: `[554, 558, 589, 571, 595, 584, 517, 507, 530, 557, 587, 586, 538, 583, 585, 540, 597, 591, 580, 548, 593, 570, 518, 551, 514, 599, 596, 508, 588, 522, 541, 510]` and 7 MRI volumes in the test split, namely: `[532, 555, 592, 578, 590, 582, 594]`. Since the SegVol authors fine-tuned on M3D-Seg, the zero-shot performance on MRI modality cannot be strictly considered zero-shot. However, the effect of 32 samples in 96,000 is estimated to be minimal. Nevertheless, we notified the authors of AMOS and M3D-Seg about the mistake, to enhance the academic corectness of future works training on AMOS and M3D-Seg.

### Appendix B: Mixture of Adapters (MoA)

Mixture of Adapters (MoA) is an advanced adaptation technique inspired by the Mixture of Experts (MoE) approach [[14]](#ref13) that utilizes multiple lightweight adapter modules within a model to handle diverse tasks and modalities. Each adapter specializes in a specific task or modality, and the model dynamically selects and combines these adapters during inference. A special case of this is a *top-1* gated mixture, where we only select one adapter at a time, which, combined with existing parameter-efficient fine-tuning (PEFT) methods, has several benefits:

- **Performance guarantees**: adapters can be designed in a way such that the base model's pre-trained weights are never changed, so recovering the original model's performance is equivalent to disabling the adapters during inference. To do this, we can introduce an "identity adapter" into our mixture.
- **Running time benefits**: modern PEFT methods can be applied very efficiently, at almost no additional cost at inference time [[13]](#ref13). Top-1 pooling can also be done in constant time, depending on the selection method we use.

In this work, we only considered a trivial case of this approach, effectively disabling the LoRA adapters to recover the pre-trained weights of the SegVol model. Through this simple trick, we can ensure that fine-tuning does not decrease the model's performance over the original target distribution, while simultaneously enabling better performance over the new target distribution. We leave the development of more sophisticated MoA architectures to future work.

### Appendix C: Interpreting context-prior tokens for tasks 
Context-prior tokens are learnable per task and modality. Whereas we only conduct experiments on 2 modalities, we train our model on 21 tasks, corresponding to different organs from the 400 dataset. To understand how context-priors help leverage similarities between tasks, we performed a t-SNE visualization of the context-prior tokens ([Figure 4](#fig4)):

<table name='fig1' align="center">
  <tr align="center">
    <td>
    	<img src="assets/task-prior-t-SNE.png" alt="task prior t-SNE" width=700px />
  	</td>
  </tr>
  <tr align="left">
    <td>
    	<b>Figure 4</b>: Task Prior Embeddings t-SNE plot
  	</td>
  </tr>
</table>

We observe that the resulting embeddings reflect, to some degree, the anatomical proximity of the respective organs. For instance right/left kidney and adrenal gland are plotted near each other, brain tumors form a cluster in the bottom of the plot. Additionally, inspired by word2vec [[36]](#ref36) we expected a high cosine similarity between `left_adrenal_gland - right_adrenal_gland + right_kidney` and `left_kidney` embeddings. However, the result amounted to -0.11. Thus, the interpretation of the context-prior token semantics remain subject for future work.
