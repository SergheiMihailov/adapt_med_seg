from typing import Callable
import numpy as np

SPLIT_NAMES = ['train', 'val', 'test']
MODALITY_MAPPING = {
    '0': 'CT',
    '1': 'MRI'
}

class load_callback:
    def __init__(self, loader: Callable, *args, **kwargs) -> None:
        self.loader = loader
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.loader(*self.args, **self.kwargs)

def three_way_split(*data_lists, test_ratio: float, val_ratio: float):
    """
        Split the data into three sets: train, val, test.
    """
    test_indices = np.random.choice(len(data_lists[0]),
                                    int(test_ratio*len(data_lists[0])), replace=False)
    test_indices = set(test_indices)
    train_indices = set(range(len(data_lists[0]))) - test_indices
    val_indices = np.random.choice(list(train_indices),
                                   int(val_ratio*len(data_lists[0])), replace=False)
    train_indices = train_indices - set(val_indices)
    retlists = [[],[],[]]
    for d_list in data_lists:
        retlists[0].append([d_list[idx] for idx in train_indices])
        retlists[1].append([d_list[idx] for idx in val_indices])
        retlists[2].append([d_list[idx] for idx in test_indices])
    return retlists

def two_way_split(*data_lists, val_ratio: float):
    """
        Split the data into two sets: train, val.
    """
    val_indices = np.random.choice(len(data_lists[0]),
                                   int(val_ratio*len(data_lists[0])), replace=False)
    val_indices = set(val_indices)
    train_indices = set(range(len(data_lists[0]))) - val_indices
    retlists = [[],[]]
    for d_list in data_lists:
        retlists[0].append([d_list[idx] for idx in train_indices])
        retlists[1].append([d_list[idx] for idx in val_indices])
    return retlists