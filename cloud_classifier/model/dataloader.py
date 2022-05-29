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
from cloud_classifier.cloud_classifier.model.utils import (
    set_transform,
    set_denormalize
)


def load_metadata(limit=None, replicate=2, image_path=IMAGE_PATH):
    """
    Load the metadata of the downloaded frames with option to replicate for use
    with augmentation
    :param limit:
    :param replicate:
    :param image_path:
    :return:
    """
    metadata_df = pd.read_csv(
        os.path.join(image_path, "to_download.csv"),
        converters={"tags": eval}
    )

    if limit:
        metadata_df = metadata_df.sample(limit)

    classes = list(
        sorted(
            metadata_df.explode("tags")["tags"].dropna().unique().tolist()
        )
    )

    dfs = []
    if replicate > 0:
        for i in range(replicate):
            dfs.append(metadata_df)
    metadata_df = pd.concat(dfs).sample(frac=1)

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
                     standard_size=(224, 224),
                     use_augmentation=True,
                     image_path=IMAGE_PATH
                     ):

            self.metadata_csv = csv
            nrecords = len(csv)
            self.train = train
            self.test = test

            self.all_image_names = self.metadata_csv[:]['id']
            self.all_labels = list(self.metadata_csv["tags"].tolist())
            self.train_records = int(train_ratio * nrecords)
            self.valid_records = nrecords - self.train_ratio
            self.resize_size = standard_size
            self.image_path = image_path

            self.classes = list(
                sorted(
                    self.metadata_csv.explode("tags")["tags"].dropna().unique().tolist()
                )
            )

            # set the training data images and labels
            if self.train == True:
                print(f"Number of training images: {self.train_records}")
                self.image_names = list(self.all_image_names[:self.train_records])
                self.labels = list(self.all_labels[:self.train_records])

                # define the training transforms
                if use_augmentation:
                    self.transform = set_transform()
                else:
                    self.transform = transforms.Compose([
                        transforms.ToPILImage(mode="RGB"),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        )
                    ])

            # set the validation data images and labels
            elif self.train == False and self.test == False:
                print(f"Number of validation images: {self.valid_records}")
                self.image_names = list(self.all_image_names[-self.valid_records:-10])
                self.labels = list(self.all_labels[-self.valid_records:])
                # define the validation transforms

                if use_augmentation:
                    self.transform = set_transform()
                else:
                    self.transform = transforms.Compose([
                        transforms.ToPILImage(mode="RGB"),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        )
                    ])

            # set the test data images and labels, only last 10 images
            # this, we will use in a separate inference script
            elif self.test == True and self.train == False:
                self.image_names = list(self.all_image_names[-10:])
                self.labels = list(self.all_labels[-10:])
                # define the test transforms
                self.transform = transforms.Compose([
                    transforms.ToPILImage(mode="RGB"),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])

        def __len__(self):
            return len(self.image_names)

        def __getitem__(self, index):
            image = Image.open(
                os.path.join(
                    self.image_path,
                    "{}_image.jpg".format(self.image_names[index])
                )
            ).resize(self.resize_size)

            # Convert to array
            image = np.asarray(image)
            # Transform (and augment, if desired)
            image = self.transform(image)
            # Encode labels
            targets = self.encode_label(self.labels[index])

            return {
                'image': image.clone().detach(),
                'label': targets.clone().detach()
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

            # Case when the image is of something that has not been classified
            if len(result) == 0:
                result = ["other"]
            return ','.join(result)

def show_batch(dataset, dataloader, nmax=16):
    """
    Show batch
    :param dataset:
    :param dataloader:
    :param nmax:
    :return:
    """

    denormalizer = set_denormalize()
    for op in dataloader:
        images = op["image"]
        labels = op["label"]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]);
        ax.set_yticks([])

        # denormalize the images
        images = denormalizer(images)

        ax.imshow(make_grid(images[:nmax], nrow=4, pad_value=50).permute(1, 2, 0))
        for i, lab in enumerate(labels):
            print("{}, {}".format(i, dataset.decode_target(lab.numpy())))
        break
