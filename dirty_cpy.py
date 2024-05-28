import json
import os
import sys
import numpy as np
from copy import deepcopy

try:
    source = sys.argv[1]
    dest = sys.argv[2]
    num_train_samples = int(sys.argv[3])  # number of training samples
    num_validation_samples = int(sys.argv[4])
    num_test_samples = int(sys.argv[5])
except:
    print("Usage: dirty_cpy.py source dest #train_samples #validation_samples #test_samples")
    sys.exit(1)

# discover all json files
data_jsons = [
    (dirname, json_file)
    for dirname, _, files in os.walk(source)
    for json_file in files if json_file.endswith('.json')
]
data_dicts = [
    (dirname, json_file, json.load(open(os.path.join(dirname, json_file))))
    for dirname, json_file in data_jsons
]
print('dicts loaded')

new_data_dicts = []
for path, json_file, data_dict in data_dicts:
    new_data_dicts.append((path, json_file, deepcopy(data_dict)))
    new_data_dicts[-1][2][
        "training" if "training" in data_dict else "train"
    ].clear()
    if "validation" in data_dict:
        new_data_dicts[-1][2]["validation"].clear()
    new_data_dicts[-1][2]["test"].clear()

for split_t, required_samples in zip([("training", "train"), ("validation", None), ("test",)], [num_train_samples, num_validation_samples, num_test_samples]):
    while True:
        for idx, dict_t in enumerate(data_dicts):
            _, _, data_dict = dict_t
            key = split_t[0] if split_t[0] in data_dict else split_t[1]

            if key is None: # skip if no validation data is present
                continue

            if len(data_dict[key]) == 0:
                continue

            # pick a single index from the list
            sample = data_dict[key].pop()
            new_data_dicts[idx][2][key].append(sample)
            required_samples -= 1

            if required_samples == 0:
                break
        if required_samples == 0:
            break
    print(f"done {split_t[0]}")

# now we have our choice of samples. We need to recreate the directory structure
# at the destination
for dirname, json_name, data_dict in new_data_dicts:
    data_rel_path = os.path.relpath(dirname, source)
    if not os.path.exists(os.path.join(dest, data_rel_path)):
        os.makedirs(os.path.join(dest, data_rel_path))
    splits = ["training", "validation", "test"] if "training" in data_dict else [
        "train", "test"]
    for split in splits:
        for sample in data_dict[split]:
            source_sample = os.path.dirname(sample["image"])
            source_path = os.path.join(dirname, source_sample)
            if not os.path.exists(source_path):
                source_sample = os.path.basename(source_sample)
                source_path = os.path.join(dirname, source_sample)
                if not os.path.exists(source_path):
                    print(f"Could not find {source_path}")
                    continue
            dest_path = os.path.join(dest, data_rel_path, source_sample)
            #if not os.path.exists(dest_path):
             #   os.makedirs(os.dest_path)
            #input(f'will copy {source_path}/ {dest_path}. exists: {os.path.exists(os.path.dirname(dest_path))}. cont?')
            rc = os.system(f"cp -R {source_path}/ {dest_path}")
            if rc != 0:
                print(f"Could not copy {source_path} --> {dest_path}")
                raise Exception("copy failed")
            print(f"split {split}: copied {source_path} --> {dest_path}")
            assert os.path.exists(os.path.join(dest, data_rel_path, sample["image"])) or \
                os.path.exists(os.path.join(dest, os.path.dirname(data_rel_path), sample["image"])), \
                f"Could not find {sample['image']} in {dest_path}"
    print(f"copied {data_rel_path}")
    # save the dictionary
    with open(os.path.join(dest, data_rel_path, json_name), "w") as f:
        json.dump(data_dict, f, indent=4)
    print(f"saved {json_name}")
print("done")
