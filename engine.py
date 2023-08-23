# Modified by Colin Wang, Weitang Liu

import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model import get_model
from tqdm import tqdm, trange


def prepare_model(device: torch.device, args=None):
    """
        Creates a model, criterion (loss function), optimizer, and scheduler
        based on arguments pass in when running. 
    """
    def get_criterion():
        """
            Returns a criterion for the argument passed in. 
        """
        criterion = args["criterion"]

        if criterion == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif criterion == "MSE":
            return nn.MSELoss()
        else:
            raise NotImplementedError

    def get_optimizer(model: nn.Module):
        """
            Returns an optimizer for the argument passed in.
        """
        optimizer = args["optimizer"]

        if optimizer == "adam":
            return optim.Adam(
                model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
            )
        elif optimizer == "sgd":
            return optim.SGD(
                model.parameters(),
                lr=args["lr"],
                momentum=args["momentum"],
                dampening=args["dampening"],
                nesterov=args["nesterov"],
                weight_decay=args["weight_decay"],
            )
        else:
            raise NotImplementedError

    def get_scheduler(optimizer: optim.Optimizer):
        """
            Returns a scheduler for the argument passed in. 
        """
        scheduler = args["lr_scheduler"]

        if scheduler == "steplr":
            return optim.lr_scheduler.StepLR(
                optimizer, step_size=args["step_size"], gamma=args["gamma"]
            )
        else:
            raise NotImplementedError

    # load model, criterion, optimizer, and learning rate scheduler
    if not os.path.exists(args["path"]):
        os.mkdir(args["path"])

    # creates a new model or uses a pretrained model if path is specified
    model = get_model(args).to(device)
    model_path = os.path.join(args["model_dir"] or 'null', f"{args['model']}.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print('Loading Existing Model From: ', model_path)

    # get the criterion, optimizer, and scheduler for the model
    criterion = get_criterion()
    optimizer = get_optimizer(model)
    lr_scheduler = get_scheduler(optimizer)

    return model, criterion, optimizer, lr_scheduler


def train_model(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
    dataloaders: list[DataLoader],
    args: dict = None,
):
    """
        Trains the model using a certain amount of epochs and testing on the
        validation set after every epoch. 
    """
    # Get the dataloaders for the training and validation set 
    train_loader, val_loader, test_loader = dataloaders
    epochs = args["epoch"]

    # initialize the writer and output for the training
    writer = SummaryWriter(args["path"], comment=args["model"])
    checkpoint_path = os.path.join(args["path"], f"{args['model']}.pt")

    # keep track of validation loss and patience
    best_val_loss = math.inf
    patience = 0

    # train over a number of epochs 
    for epoch in trange(epochs, desc="Epoch"):
        model.train()

        # keep track of loss, correct, and accuracy for training 
        epoch_train_correct = 0
        epoch_train_loss = 0
        epoch_train_acc = 0
        iterations = 0  # iterations; number of batches

        # train over a number of batches
        for i, (data, target) in enumerate(
            tqdm(train_loader, desc="Train", leave=False)
        ):
            # forward pass
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(axis=1)
            loss = criterion(output, target)

            # keep track of loss and # of correct
            iterations += 1
            epoch_train_loss += loss
            epoch_train_correct += (pred == target).sum()

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Average out the accuracy and loss to make them size invariant
        epoch_train_acc = epoch_train_correct / len(train_loader.dataset)
        epoch_train_loss = epoch_train_loss / iterations
        print(f'Train Loss: {epoch_train_loss:.3f}')
        print(f'Train Acc:  {epoch_train_acc:.5f}')
        writer.add_scalar("Acc/train", epoch_train_acc, epoch)
        writer.add_scalar("Loss/train", epoch_train_loss, epoch)

        # test on validation set by doing a forward pass
        iterations = 0
        model.eval()
        with torch.no_grad():
            epoch_val_loss: torch.Tensor = 0
            epoch_val_correct: torch.Tensor = 0
            for i, (data, target) in enumerate(
                tqdm(val_loader, desc="Val", leave=False)
            ):
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(axis=1)
                loss = criterion(output, target)

                # accumulate loss and correct values
                iterations += 1
                epoch_val_loss += loss
                epoch_val_correct += (pred == target).sum()

        # Average out the accuracy and loss to make them size invariant
        epoch_val_acc = epoch_val_correct / len(val_loader.dataset)
        epoch_val_loss = epoch_val_loss / iterations
        print(f'Val Loss: {epoch_val_loss:.3f}')
        print(f'Val Acc:  {epoch_val_acc:.5f}')
        writer.add_scalar("Acc/val", epoch_val_acc, epoch)
        writer.add_scalar("Loss/val", epoch_val_loss, epoch)

        # patience and validation increase detection
        if epoch_val_loss.item() < best_val_loss:
            torch.save(model.state_dict(), checkpoint_path)
            patience = 0
        else:
            patience += 1

        # early stop detection
        if args["early_stop"] and patience > args["patience"]:
            break

        scheduler.step()

    model.load_state_dict(torch.load(checkpoint_path)) # return the model with weight selected by best performance
    writer.flush()
    return model 


def test_model(
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    dataloaders: list[DataLoader],
    args: dict = None,
):
    """
        Tests the model by feeding in unseen test data into the network and calculating
        accuracy and loss for that unseen set of data. 
    """
    # load test data and set model to evaluate
    test_loader = dataloaders[2]
    model.eval()
    
    # keep track of loss, correct, and iterations for accuracy and loss calculations
    test_loss: torch.Tensor = 0
    test_correct: torch.Tensor = 0
    iterations = 0

    # do a forward pass with the test data 
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(test_loader, desc="Test")):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(axis=1)
            loss = criterion(output, target)

            # accumulate loss and correct values
            test_loss += loss
            iterations += 1
            test_correct += (pred == target).sum()

    # average out the test loss and accuracy to make them size invariant
    test_loss = test_loss / iterations
    acc = test_correct / len(test_loader.dataset)
    return test_loss, acc
