import torch
import random
import json
import numpy as np

from adapt_med_seg.pipelines.evaluate import EvaluateArgs, EvaluatePipeline
from adapt_med_seg.utils.cli import parse_arguments

def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

def main():
    args = parse_arguments()
    kwargs = vars(args)
    seed = kwargs.pop("seed")
    seed_everything(seed)

    pipeline = EvaluatePipeline(**kwargs)
    results = pipeline.run()
    print("Evaluation finished. Results:")
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()