from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import torchvision.transforms as transforms


def set_denormalize():

    de_normalize = transforms.Compose(
        [
            transforms.Normalize(mean=[0., 0., 0.],
                                 std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                                 ),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                 std=[1., 1., 1.]
                                 ),
        ]
    )

    return de_normalize

def set_transform():

    transform_pipe = transforms.Compose([
        transforms.ToPILImage(mode="RGB"),
        transforms.RandomChoice([
            transforms.RandomRotation(degrees=(-30, 30)),
            transforms.RandomHorizontalFlip(p=0.2)
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform_pipe


def calculate_number_correct(outputs, labels):

    corrects = 0
    for predictions, trues in zip(outputs, labels):
        corrects += np.array_equal(predictions, trues)
    perc_corrects = corrects / len(labels)

    return perc_corrects, corrects


def evaluation_metrics(predictions, target, classes):

    precision_array = precision_score(predictions, target, average=None)
    recall_array = recall_score(predictions, target, average=None)
    f1_array = f1_score(predictions, target, average=None)

    print("-------------------------------------")
    for class_name, p, r, f1 in zip(classes, precision_array, recall_array, f1_array):
        print("{}: Pr: {} Re: {} F1: {}".format(class_name, p, r, f1))

    overall_f1 = f1_score(predictions, target, average="micro")

    return overall_f1


def calculate_validation_accuracy_metrics(epoch_validation_list):

    preds = np.vstack([x[0] for x in epoch_validation_list])
    targets = np.vstack([x[1] for x in epoch_validation_list])

    overall_f1 = evaluation_metrics(preds, targets)

    return overall_f1
