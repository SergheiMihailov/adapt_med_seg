from typing import Callable

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
