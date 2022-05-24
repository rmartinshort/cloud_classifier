from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import numpy as np

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
