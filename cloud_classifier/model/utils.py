from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import numpy as np
import torchvision.transforms as transforms


def set_transform():
    transform_pipe = transforms.Compose([
        transforms.ToPILImage(mode="RGB"),
        transforms.RandomChoice([
            transforms.RandomRotation(degrees=(-30, 30)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ]),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ])

    return transform_pipe


def evaluation_metrics(model ,inputs ,labels ,threshold=0.5):

    outputs = model(inputs)
    outputs = torch.sigmoid(outputs).detach().numpy()
    predictions = np.where(outputs>threshold, 1, 0)
    target = labels.detach().numpy()

    # print these out
    # precision_score(predicted_op,true_op,average=None)
    # recall_score(predicted_op,true_op,average=None)
    # f1_score(predicted_op,true_op,average=None)
    overall_f1 = f1_score(predictions,
                          target,
                          average="micro")

    return overall_f1
