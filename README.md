# SegEVOLution: Enhanced Medical Image Segmentation with Multimodality Learning

### Z. Fülöp, S. Mihailov, M. Krastev, M. Hamar, D.A. Toapanta 

<table align="center" name="fig3">
  <tr align="center">
      <td><img src="./assets/adapt_med_seg.png"></td>
  </tr>
  <tr align="left">
    <!-- <td colspan="2"><b>Figure 3.</b> Proposed architecture combining SegVol model and Hermes context-prior framework. This hybrid model integrates SegVol’s volumetric segmentation with Hermes’s context-prior learning.</td> -->
  </tr>
</table>

**Keywords**: 3D medical SAM, volumetric image segmentation, LoRA, Context-Prior Learning

---

This repository contains a reproduction and extension of ["SegVol: Universal and Interactive Volumetric Medical Image Segmentation"](https://arxiv.org/abs/2311.13385) by Du et al. (2023). 

To read the full report containing detailed information on our  experiments and extension study, please, refer to our [blogpost](blogpost.md).

## Requirements

To get started, clone the repository and install the dependencies using [Poetry](https://python-poetry.org/).

1. Clone the environment 

    ```bash
    git clone https://github.com/SergheiMihailov/adapt_med_seg.git
    ```

2. Activate the existing Poetry environment:

    ```bash
    poetry shell
    ```

3. Install the project dependencies (if not already installed):

    ```bash
    poetry install
    ```

This will activate the primary environment with all necessary dependencies for the main functionalities of this project.

## Datasets Involved 

This project used the [M3D-Seg](https://huggingface.co/datasets/GoodBaiBai88/M3D-Seg) dataset. This dataset was trained  

To donwload the dataset 


## Setting Up the Pipelines

### Training Pipeline

The training pipeline is defined in `adapt_med_seg/train.py`. To run the training script, follow these steps:

1. Run the training script:

    ```bash
    python adapt_med_seg/train.py
    ```

### Evaluation Pipeline

The evaluation pipeline is defined in `adapt_med_seg/eval.py` and `adapt_med_seg/pipelines/evaluate.py`. To evaluate the model, follow these steps:`

1. Run the evaluation script:

    ```bash
    python adapt_med_seg/eval.py
    ```

## Notebooks

Several Jupyter notebooks are provided for various tasks such as preprocessing, inference, and visualization:

- `notebooks/inference_colab.ipynb`: Inference on Colab.
- `notebooks/inference.ipynb`: General inference notebook.
- `notebooks/preprocess.ipynb`: Data preprocessing steps.
- `notebooks/process_results.ipynb`: Processing and analyzing results.
- `notebooks/SegVol_initial_tryouts.ipynb`: Initial segmentation volume tryouts.
- `notebooks/vis.ipynb`: Visualization of results.
- `notebooks/zsombor_balance_amos.py`: Script for balancing AMOS dataset.

To run these notebooks, activate the Poetry environment and start Jupyter Notebook:

```bash
poetry shell
jupyter notebook
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to adjust any of the sections according to your specific requirements and add any additional information that might be relevant.


