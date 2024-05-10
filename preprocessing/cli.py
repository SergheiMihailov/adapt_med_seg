import argparse
# CLI
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes',
                        help='List of classes to segment. This can be overridden by the metadata',
                        type=list,
                        default=['liver', 'right kidney', 'spleen', 'pancreas', 'aorta', 'inferior vena cava', 'right adrenal gland', 'left adrenal gland', 'gallbladder', 'esophagus', 'stomach', 'duodenum', 'left kidney'],
                        action='store')
    parser.add_argument('--image_dir',
                        type=str,
                        required=True,
                        action='store')
    parser.add_argument('--label_dir',
                        help='Path to the directory containing the labels',
                        type=str,
                        required=True,
                        action='store')
    parser.add_argument('--dataset_root',
                        help='Root directory of the dataset',
                        type=str,
                        required=True,
                        action='store')
    parser.add_argument('--dataset_code',
                        help='Custom identifier for the dataset',
                        type=str,
                        required=True,
                        action='store')
    parser.add_argument('--save_root',
                        help='Root directory to save the dataset',
                        type=str,
                        required=True,
                        action='store')
    parser.add_argument('--num_workers',
                        help='Number of worker processes to use',
                        type=int,
                        default=1,
                        action='store')
    parser.add_argument('--seed',
                        help='Random seed, for reproducibility in case any random shuffling is performed',
                        type=int,
                        default=42,
                        action='store')
    random_splits_parser = parser.add_argument_group('random_splits',
                                                    description='Randomly split the dataset into training and testing sets')
    random_splits_parser.add_argument('--test_ratio',
                                      help='Ratio of the dataset to use for testing',
                                      type=float,
                                      required=False,
                                      default=None,
                                      action='store')
    random_splits_parser.add_argument('--val_ratio',
                                      help='Ratio of the dataset to use for validation.'\
                                          'It is the proportion of non-test data, i.e. val_ratio = (1 * test_ratio) * val_ratio',
                                      type=float,
                                      required=False,
                                      default=None,
                                      action='store')
    random_splits_parser.add_argument('--modality',
                                      help='Modality of the dataset',
                                      type=str,
                                      required=False,
                                      default='CT',
                                      action='store')

    meta_parser = parser.add_argument_group('metadata',
                              description='JSON metadata for the dataset')
    meta_parser.add_argument('--dataset_type',
                        type=str,
                        required=True,
                        action='store',
                        help='''Type of dataset (AMOS, CHAOS, ...)
                                Both the splits and the modality of the images will be inferred from the metadata''',
                        choices=['AMOS', 'CHAOS'])

    args = parser.parse_args()
    if args.test_ratio is not None and args.metadata_path is not None:
        raise ValueError('Cannot provide both test_ratio and metadata_path')
    if args.test_ratio is None and args.metadata_path is None:
        raise ValueError('Must provide either test_ratio or metadata_path')
    return args
