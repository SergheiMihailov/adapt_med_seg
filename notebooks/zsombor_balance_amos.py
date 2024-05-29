import json
import random

with open("/scratch-shared/scur0402/AMOS_2022/amos_mrct_42/dataset.json_notest", "r") as f:
    data = json.load(f)


# Separate samples by modality
modality_0 = [sample for sample in data['training']+data["validation"] if sample['modality'] == "0"]
modality_1 = [sample for sample in data['training']+data["validation"] if sample['modality'] == "1"]

# Determine the target number of samples for each modality
target_num_samples = min(len(modality_0), len(modality_1))
print(f"Target number of samples: {target_num_samples}")

# Randomly select the target number of samples from each modality
random.shuffle(modality_0)
random.shuffle(modality_1)

modality_0 = modality_0[:target_num_samples]
modality_1 = modality_1[:target_num_samples]

# Combine the balanced samples
balanced_samples = modality_0 + modality_1

# Shuffle the balanced samples
random.shuffle(balanced_samples)

# Determine the number of training and validation samples
num_samples = len(balanced_samples)
num_training_samples = int(num_samples * 0.9)
num_validation_samples = num_samples - num_training_samples

# Split into training and validation sets
training_samples = balanced_samples[:num_training_samples]
validation_samples = balanced_samples[num_training_samples:]

# Update the data dictionary
data['training'] = training_samples
data['validation'] = validation_samples
data['test'] = []

data['numTraining'] = len(training_samples)
data['numValidation'] = len(validation_samples)
data['numTest'] = 0

# Print or save the updated JSON data
updated_json_data = json.dumps(data, indent=2)
with open("/scratch-shared/scur0402/AMOS_2022/amos_mrct_42/balanced_dataset.json", "w") as f:
    f.write(updated_json_data)