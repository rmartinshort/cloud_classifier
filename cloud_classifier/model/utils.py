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
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform_pipe


def evaluation_metrics(model ,inputs ,labels , threshold=0.5):

    outputs = model(inputs)
    outputs = torch.sigmoid(outputs).detach().cpu().numpy()
    predictions = np.where(outputs>threshold, 1, 0)
    target = labels.detach().cpu().numpy()

    # print these out
    # precision_score(predicted_op,true_op,average=None)
    # recall_score(predicted_op,true_op,average=None)
    # f1_score(predicted_op,true_op,average=None)
    overall_f1 = f1_score(predictions,
                          target,
                          average="micro")

    return overall_f1
