import torch
import pandas as pd
from PIL import Image
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from cloud_classifier.cloud_classifier.common.constants import IMAGE_PATH

def load_metadata(limit=None):

    metadata_df = pd.read_csv(
        os.path.join(IMAGE_PATH, "to_download.csv"),
        converters={"tags": eval}
    )

    if limit:
        metadata_df = metadata_df.sample(limit)

    classes = list(sorted(metadata_df.explode("tags")["tags"].unique().tolist()))
    return metadata_df, classes

class ImageDataset(Dataset):
    """
    Use as follows:

    train_data = ImageDataset(
    metadata_df, train=True, test=False
    )
    valid_data = ImageDataset(
    metadata_df, train=False, test=False
    )

    train_loader = DataLoader(
    train_data,
    batch_size=8,
    shuffle=True
    )
    """
    def __init__(self,
                 csv,
                 train=True,
                 test=False,
                 train_ratio=0.75,
                 standard_size=(224, 224)
                 ):

        self.metadata_csv = csv
        nrecords = len(csv)
        self.train = train
        self.test = test

        self.all_image_names = self.metadata_csv[:]['id']
        self.all_labels = list(self.metadata_csv["tags"].tolist())
        self.train_ratio = int(train_ratio * nrecords)
        self.valid_ratio = nrecords - self.train_ratio
        self.resize_size = standard_size
        self.classes = list(sorted(self.metadata_csv.explode("tags")["tags"].unique().tolist()))

        # set the training data images and labels
        if self.train == True:
            print(f"Number of training images: {self.train_ratio}")
            self.image_names = list(self.all_image_names[:self.train_ratio])
            self.labels = list(self.all_labels[:self.train_ratio])

            # define the training transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(mode="RGB"),
                transforms.ToTensor()
            ])
        # set the validation data images and labels
        elif self.train == False and self.test == False:
            print(f"Number of validation images: {self.valid_ratio}")
            self.image_names = list(self.all_image_names[-self.valid_ratio:-10])
            self.labels = list(self.all_labels[-self.valid_ratio:])
            # define the validation transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(mode="RGB"),
                transforms.ToTensor()
            ])
        # set the test data images and labels, only last 10 images
        # this, we will use in a separate inference script
        elif self.test == True and self.train == False:
            self.image_names = list(self.all_image_names[-10:])
            self.labels = list(self.all_labels[-10:])
            # define the test transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(mode="RGB"),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image = Image.open(
            os.path.join(IMAGE_PATH, "{}_image.jpg".format(self.image_names[index]))
        ).resize(self.resize_size)
        image = np.asarray(image)
        image = self.transform(image)
        targets = self.encode_label(self.labels[index])

        return {
            'image': torch.tensor(image, dtype=torch.float32),
            'label': torch.tensor(targets, dtype=torch.float32)
        }

    def encode_label(self, label):
        """
        encoding the classes into a tensor of shape (nclasses) with 0 and 1s.
        :param label:
        :param classes_list:
        :return:
        """
        target = torch.zeros(len(self.classes))
        for l in label:
            idx = self.classes.index(l)
            target[idx] = 1
        return target

    def decode_target(self, target, threshold=0.5):
        """
        decoding the prediction tensors of 0s and 1s into text form
        :param target:
        :param threshold:
        :return:
        """
        result = []
        for i, x in enumerate(target):
            if (x >= threshold):
                result.append(self.classes[i])
        return ','.join(result)


def show_batch(dataloader, nmax=16):
    for op in dataloader:
        images = op["image"]
        labels = op["label"]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]);
        ax.set_yticks([])
        ax.imshow(make_grid(images[:nmax], nrow=4, pad_value=50).permute(1, 2, 0))
        for i, lab in enumerate(labels):
            print("{}, {}".format(i, dataloader.decode_target(lab.numpy())))
        break
