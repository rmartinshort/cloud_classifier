from cloud_classifier.cloud_classifier.model.dataloader import (
    load_metadata,
    ImageDataset
)

from cloud_classifier.cloud_classifier.model.torchvision_model import (
    initialize_model,
    train_model
)

import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch

def baseline_main(params):
    """
    params = {
        "epochs":30,
        "data_limit":None,
        "batch_size":8,
        "lr":1e-4,
        "model_name":"resnet"
    }
    :param params:
    :return:
    """

    epochs = params.get("epochs", 30)
    data_limit = params.get("data_limit", None)
    batch_size = params.get("batch_size", 8)
    learning_rate = params.get("learning_rate", 1e-4)
    torchvision_model = params.get("model_name", "resnet")

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    metadata_df, classes = load_metadata(limit=data_limit)

    train_data = ImageDataset(
        metadata_df, train=True, test=False
    )

    valid_data = ImageDataset(
        metadata_df, train=False, test=False
    )

    # train data loader
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )
    # validation data loader
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False
    )

    dataloaders = {
        "train": train_loader,
        "valid": valid_loader
    }

    # initialize the model
    model, input_size = initialize_model(
        torchvision_model,
        num_classes=len(classes),
        feature_extract=False,
        use_pretrained=True
    )

    model.to(device)

    # learning parameters
    lr = learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Training
    trained_model, train_loss, valid_loss = train_model(model,
                                                        dataloaders,
                                                        criterion,
                                                        optimizer,
                                                        N=len(train_data),
                                                        num_epochs=epochs,
                                                        is_inception=False,
                                                        batch_lim=None
                                                        )

    return trained_model, train_loss, valid_loss
