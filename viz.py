import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import model as models
import math

def plot_conv_weights(title: str, weight: torch.Tensor):
    """
    Plots the weights for a Conv2D weights
    """
    n_out = weight.shape[0]
    plot_width = int(n_out ** .5)
    fig, ax = plt.subplots(plot_width, plot_width, figsize=((plot_width * 1.5, plot_width * 1.5)))
    for i, out_channel in enumerate(weight):
        row = i % plot_width
        col = i // plot_width

        # Original: (in channels [RGB], width, height)
        # Needed: (width, height, in channels [RGB])
        out_channel = out_channel.clone().detach().permute(1, 2, 0)
        out_channel -= out_channel.min()
        out_channel /= out_channel.max()

        ax[row, col].imshow(out_channel)
    fig.suptitle(title)
    fig.savefig(f'viz/{title}.png')

def plot_model_weights(model: nn.Module):
    """
    Plots the weights for a particular model
    """
    model = model.to('cpu')

    if isinstance(model, models.custom):
        return plot_conv_weights('Custom Weight Map', model.conv1.conv.weight)
    elif isinstance(model, models.vgg):
        return plot_conv_weights('VGG Weight Map', model.pretrained.features[0].weight)
    elif isinstance(model, models.resnet):
        return plot_conv_weights('ResNet Weight Map', model.pretrained.conv1.weight)
    else:
        raise NotImplementedError


def plot_activation_weights(title: str, X: torch.Tensor):
    """
    Plot activation weights
    """
    n_out = X.shape[0]
    nrows = math.ceil(n_out ** .5)
    ncols = math.ceil(n_out / nrows)
    fig, ax = plt.subplots(nrows, ncols, figsize=((nrows * 1.5, ncols * 1.5)))
    for i, out_channel in enumerate(X):
        col = i % ncols
        row = i // ncols

        out_channel = out_channel.clone().detach()
        out_channel -= out_channel.min()
        out_channel /= out_channel.max()

        ax[row, col].imshow(out_channel)
    fig.suptitle(title)
    fig.savefig(f'viz/{title}.png')

def plot_forward_hook(title: str):
    """
    Wrapper to plot activation weights as a forward hook
    """
    def hook(module, input, output):
        plot_activation_weights(title, output[0])
    return hook

def plot_model_activations(test_loader: DataLoader, model: nn.Module):
    """
    Plots the activations for three different layers in the model
    """
    model = model.to('cpu')

    X, y = next(iter(test_loader))
    X = X[:1] # take first only

    activations: dict[str, nn.Conv2d]
    if isinstance(model, models.custom):
        activations = {
            'Custom First Conv Activation': model.conv1.conv,
            'Custom Middle Conv Activation': model.conv4.conv,
            'Custom Last Conv Activation': model.conv6.conv,
        }
    elif isinstance(model, models.vgg):
        activations = {
            'VGG First Conv Activation': model.pretrained.features[0],
            'VGG Middle Conv Activation': model.pretrained.features[20],
            'VGG Last Conv Activation': model.pretrained.features[40],
        }
    elif isinstance(model, models.resnet):
        activations = {
            'ResNet First Conv Activation': model.pretrained.conv1,
            'ResNet Middle Conv Activation': model.pretrained.layer2[0].conv2,
            'ResNet Last Conv Activation': model.pretrained.layer4[1].conv2,
        }
    else:
        raise NotImplementedError

    hooks = []
    for title, layer in activations.items():
        hook = layer.register_forward_hook(plot_forward_hook(title))
        hooks.append(hook)
    
    model(X)
    for hook in hooks:
        hook.remove()

