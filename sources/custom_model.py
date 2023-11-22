import torchvision
from torchvision import models
import torch

class DeepLabV3Wrapper(torch.nn.Module):
    def __init__(self, model):
        super(DeepLabV3Wrapper, self).__init__()
        self.model = model

    def forward(self, input):
        output = self.model(input)['out']
        return output

def initialize_model(num_classes, keep_feature_extract=False):
    """ DeepLabV3 pretrained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
    """
    model_deeplabv3 = models.segmentation.deeplabv3_resnet50(weights='DEFAULT', progress=True)
    model_deeplabv3.aux_classifier = None
    if keep_feature_extract:
        for param in model_deeplabv3.parameters():
            param.requires_grad = False

    model_deeplabv3.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, num_classes)

    return model_deeplabv3

