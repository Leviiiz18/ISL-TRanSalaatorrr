import os
import json

train_dir = 'Indian'
class_names = sorted(os.listdir(train_dir))  # get class folder names sorted
class_indices = {class_name: i for i, class_name in enumerate(class_names)}

print(class_indices)

# Save as JSON file
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f)
