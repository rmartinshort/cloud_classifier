from cloud_classifier.cloud_classifier.model.utils import (
    calculate_number_correct,
    calculate_validation_accuracy_metrics
)

import torch
import time
import copy
import numpy as np
from tqdm import tqdm
from torchvision import models as models
import torch.nn as nn

def set_parameter_requires_grad(model, feature_extracting):
    """

    :param model:
    :param feature_extracting:
    :return:
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name,
                     num_classes,
                     feature_extract,
                     use_pretrained=True,
                     final_dropout=0.2):
    """
    Note that feature extract is bool and set to
    true of we don't want to fine-tune the hidden layers
    """

    model_ft = None
    input_size = 0

    if "resnet" in model_name:
        """ Resnet models
        See https://pytorch.org/hub/pytorch_vision_resnet/
        """
        model_size = model_name[-2:]
        if model_size == "18":
            model_ft = models.resnet18(pretrained=use_pretrained)
        elif model_size == "34":
            model_ft = models.resnet34(pretrained=use_pretrained)
        elif model_size == "50":
            model_ft = models.resnet50(pretrained=use_pretrained)
        else:
            # Default to resnet 101
            model_ft = models.resnet101(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features

        # Set the final classification layer to include droput
        model_ft.fc = nn.Sequential(
            nn.Dropout(final_dropout),
            nn.Linear(num_ftrs, num_classes)
        )

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def train_model(model,
                dataloaders,
                criterion,
                optimizer,
                num_epochs=25,
                is_inception=False,
                batch_lim=2,
                threshold=0.5):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    since = time.time()

    train_epoch_loss_history = []
    valid_epoch_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)

        # Each epoch has a training and validation phase
        epoch_valid_predictions = []
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            # Get the current dataloader
            dataloader = dataloaders[phase]
            Ndata = len(dataloader.dataset)

            # Iterate over batches
            for i, loaded_data in tqdm(enumerate(dataloader), total=int(Ndata / dataloader.batch_size)):
                inputs = loaded_data["image"]
                labels = loaded_data["label"]

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)

                        # Need all OP between 0 and 1
                        outputs = torch.sigmoid(outputs)
                        aux_outputs = torch.sigmoid(aux_outputs)

                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)

                        # Need all OP between 0 and 1
                        outputs = torch.sigmoid(outputs)
                        loss = criterion(outputs, labels)

                    # Get the proportion of the batch that was correctly labelled
                    # Does array comprision for multilabel problem
                    op_tmp = outputs.detach().cpu().numpy()
                    predicted_op = np.where(op_tmp > threshold, 1, 0)
                    targets = labels.detach().cpu().numpy()
                    p_correct, n_correct = calculate_number_correct(
                        predicted_op,
                        targets
                    )
                    running_corrects += n_correct

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        # Append the batch predictions if we're in validation mode
                        epoch_valid_predictions.append((predicted_op, targets))

                # statistics
                running_loss += loss.item() * inputs.size(0)

                if batch_lim:
                    if i >= batch_lim:
                        print("At batch lim = {}".format(batch_lim))
                        break

            # Loss for the epoch
            epoch_loss = running_loss / Ndata

            # proportion of predictions that were fully correct
            epoch_acc = running_corrects / Ndata

            if phase == "train":
                train_epoch_loss_history.append(epoch_loss)
            else:
                # Do final validation calculaton here
                # Here we calculate accuarcy metrics on the results of the
                # entire validation set, which has been passed though the model and is
                # contained in the list epoch_valid_predictions
                validation_f1 = calculate_validation_accuracy_metrics(epoch_valid_predictions, dataloader.dataset.classes)
                valid_epoch_loss_history.append(epoch_loss)
                print("--------------------------------------------")
                print('Validation overall F1 score: {:.2f}'.format(validation_f1))
                print("--------------------------------------------")

            print('{} Loss: {:.2f}, Accuracy: {:.2f}, # corrects: {:.1f}, # inputs: {:.1f}'.format(
                phase,
                epoch_loss,
                epoch_acc,
                running_corrects,
                Ndata
            ))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_epoch_loss_history, valid_epoch_loss_history
