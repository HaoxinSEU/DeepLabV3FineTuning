import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import os
import glob
from PIL import Image

# custom Dataset class to load our dataset
# used for train, val, and test
class DataLoaderSegmentation(torch.utils.data.dataset.Dataset):
    def __init__(self, folder_path, mode):
        super(DataLoaderSegmentation, self).__init__()
        # get all image filenames
        self.img_files = glob.glob(os.path.join(folder_path, 'input', mode, '*.*'))

        # get all targets (GT)
        self.label_files = []
        for img_path in self.img_files:
            image_filename, _ = os.path.splitext(os.path.basename(img_path))
            image_filename = image_filename.split('_', 1)  # get image ID
            label_filename_with_ext = f"target_{image_filename[1]}.tif"
            self.label_files.append(os.path.join(folder_path, 'target', mode, label_filename_with_ext))

        if "val" == mode or "test" == mode:
            # just normalize and resize (512x512) for validation and test
            self.transforms = transforms.Compose([
                transforms.Resize(size=(512, 512), interpolation=InterpolationMode.NEAREST),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406, 0], [0.229, 0.224, 0.225, 1])
            ])
        else:
            # data augmentation, normalize and resize (512x512) for training
            self.transforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.Resize(size=(512, 512), interpolation=InterpolationMode.NEAREST),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406, 0], [0.229, 0.224, 0.225, 1])
                ])

    def __getitem__(self, index):
            img_path = self.img_files[index]
            label_path = self.label_files[index]

            image = Image.open(img_path)
            label = Image.open(label_path)

            # Concatenate image and label, to apply same transformation on both
            image_np = np.asarray(image)
            label_np = np.array(label)
            label_np[label_np == 255] = 2  # replace 255 by 2 in the label, so that crossEntropy can work
            new_shape = (image_np.shape[0], image_np.shape[1], image_np.shape[2] + 1)
            image_and_label_np = np.zeros(new_shape, image_np.dtype)
            image_and_label_np[:, :, 0:3] = image_np
            image_and_label_np[:, :, 3] = label_np

            # Convert to PIL
            image_and_label = Image.fromarray(image_and_label_np)

            # Apply Transforms
            image_and_label = self.transforms(image_and_label)

            # Extract image and label
            image = image_and_label[0:3, :, :]
            label = image_and_label[3, :, :].unsqueeze(0)

            # Normalize back from [0, 1] to [0, 255]
            label = label * 255
            # Convert to int64 and remove second dimension
            label = label.long().squeeze()

            return image, label

    def __len__(self):
        return len(self.img_files)
