import scipy.io as sio
import numpy as np
import json
import os



data_root = '/home/COCOSearch18/TP/fixations'
train_fixation_file = os.path.join(data_root, 'coco_search18_fixations_TP_train.json')
validation_fixation_file = os.path.join(data_root, 'coco_search18_fixations_TP_validation.json')
test_fixation_file = os.path.join(data_root, 'coco_search18_fixations_TP_test.json')

with open(train_fixation_file) as f:
    train_fixations = json.load(f)

with open(validation_fixation_file) as f:
    validation_fixations = json.load(f)

with open(test_fixation_file) as f:
    test_fixations = json.load(f)

fixations = train_fixations + validation_fixations + test_fixations

save_json_file = '/home/COCOSearch18/TP/processed/fixations.json'
with open(save_json_file, 'w') as f:
    json.dump(fixations, f, indent=2)