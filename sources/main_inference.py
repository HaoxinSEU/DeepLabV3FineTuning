import torch
import numpy as np
from tqdm import tqdm

# local import
import custom_model
from dataloader import DataLoaderSegmentation
from iou import iou

# file path
data_dir = "../example_forest"
weights_dir = "../saved_model_weights/best_DeepLabV3_weights.pth"
num_classes = 3

# detect if we have GPU or not
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# import our trained model
model = custom_model.initialize_model(num_classes, keep_feature_extract=True)
state_dict = torch.load(weights_dir, map_location=device)
model = model.to(device)
model.load_state_dict(state_dict)

# set the model in evaluation mode
model.eval()

# load the test set
test_dataset = DataLoaderSegmentation(data_dir, 'test')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

print("Starting to do inference...")

running_iou_means = []

# Iterate over test set
for inputs, labels in tqdm(test_dataloader):
    inputs = inputs.to(device)
    labels = labels.to(device)

    # do the inference
    outputs = model(inputs)["out"]
    _, preds = torch.max(outputs, 1)

    # statistics
    iou_mean = iou(preds, labels, num_classes).mean()
    running_iou_means.append(iou_mean)

test_iou = np.array(running_iou_means).mean()
print('Test mIoU is: ', test_iou)
