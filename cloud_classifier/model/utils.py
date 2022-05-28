from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import numpy as np
import torchvision.transforms as transforms


def set_transform():
    transform_pipe = transforms.Compose([
        transforms.ToPILImage(mode="RGB"),
        transforms.RandomChoice([
            transforms.RandomRotation(degrees=(-30, 30)),
            transforms.RandomHorizontalFlip(p=0.5)
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform_pipe


def evaluation_metrics(model, inputs, labels, classes, threshold=0.5):
    """

    :param model:
    :param inputs:
    :param labels:
    :param classes:
    :param threshold:
    :return:
    """
    outputs = model(inputs)
    outputs = torch.sigmoid(outputs).detach().cpu().numpy()
    predictions = np.where(outputs > threshold, 1, 0)
    target = labels.detach().cpu().numpy()

    precision_array = precision_score(predictions, target, average=None)
    recall_array = recall_score(predictions, target, average=None)
    f1_array = f1_score(predictions, target, average=None)

    print("-------------------------------------")
    for class_name, p, r, f1 in zip(classes, precision_array, recall_array, f1_array):
        print("{}: Pr: {} Re: {} F1: {}".format(class_name, p, r, f1))

    overall_f1 = f1_score(predictions, target, average="micro")

    return overall_f1
