import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn, resnet18


class ConvLayer(nn.Module):
    """
        Class for creating a Convolutional layer that has BatchNorm and ReLU activation.
    """
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size=3, stride=1
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, X: torch.Tensor):
        out = self.conv(X)
        out = self.norm(out)
        out = self.activation(out)
        return out


class FCLayer(nn.Module):
    """
        Class for a fully connected layer that has dropout and ReLU activation.
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout()
        self.activation = nn.ReLU()

    def forward(self, X: torch.Tensor):
        out = self.linear(X)
        out = self.dropout(out)
        out = self.activation(out)
        return out


class baseline(nn.Module):
    """
        Creates a baseline model based on the specifications in the writeup. 
    """
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = ConvLayer(3, 64, kernel_size=3)
        self.conv2 = ConvLayer(64, 128, kernel_size=3)
        self.conv3 = ConvLayer(128, 128, kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3)
        self.conv4 = ConvLayer(128, 128, kernel_size=3, stride=2)
        self.adaptiveavgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = FCLayer(128, 128)
        self.fc2 = nn.Linear(128, num_classes)

        def xavier_init(m: nn.Module):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)

        self.apply(xavier_init)

    def forward(self, X: torch.Tensor):
        out = self.conv1(X)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.maxpool1(out)
        out = self.conv4(out)
        out = self.adaptiveavgpool(out)
        out = out.squeeze()
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
class custom(nn.Module):
    """
        An improvement upon the baseline model with 2 extra convolutional layers and 
        an extra max pooling layer.
    """
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = ConvLayer(3, 64, kernel_size=3)
        self.conv2 = ConvLayer(64, 128, kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3)     # added one more max pooling layer
        self.conv3 = ConvLayer(128, 128, 3, 1) # add one more conv layers (2 added conv layers per requirement)
        self.conv4 = ConvLayer(128, 128, kernel_size=3, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3)
        self.conv5 = ConvLayer(128, 128, 3, 1) # add one more conv layers (2 added conv layers per requirement)
        self.conv6 = ConvLayer(128, 128, kernel_size=3, stride=2)
        self.adaptiveavgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = FCLayer(128, 128)
        self.fc2 = nn.Linear(128, num_classes)

        def xavier_init(m: nn.Module):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)

        self.apply(xavier_init)

    def forward(self, X: torch.Tensor):
        out = self.conv1(X)
        out = self.conv2(out)
        out = self.maxpool1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.maxpool2(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.adaptiveavgpool(out)
        out = out.squeeze()
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class resnet(nn.Module):
    """
        Transfer learning using resnet 18. 
    """
    def __init__(self, finetune: bool, num_classes: int, selective: bool) -> None:
        super().__init__()
        self.pretrained = resnet18(True)

        if not finetune:
            # Freeze all pretrained parameters
            for param in self.pretrained.parameters():
                param.requires_grad = False

        if selective:
            # selectively freeze the first convolutional layer of resnet
            self.pretrained.conv1.requires_grad = False

        # replace last fully connected layer with a layer fit for our number of outputs
        pretrained_features = self.pretrained.fc.in_features
        self.pretrained.fc = nn.Linear(pretrained_features, num_classes)

    def forward(self, X: torch.Tensor):
        return self.pretrained(X)


class vgg(nn.Module):
    """
        Transfer learning using vgg 16 with batch norm. 
    """
    def __init__(self, finetune: bool, num_classes: int, selective: bool) -> None:
        super().__init__()
        self.pretrained = vgg16_bn(True)

        if not finetune:
            # Freeze all pretrained parameters
            for param in self.pretrained.parameters():
                param.requires_grad = False
        elif selective: 
            # selectively freeze the first convolutional layer of vgg
            self.pretrained.features[0].requires_grad = False

        # replace last fully connected layer with a layer fit for our number of outputs
        pretrained_features = self.pretrained.classifier[-1].in_features
        self.pretrained.classifier[-1] = nn.Linear(pretrained_features, num_classes)

    def forward(self, X: torch.Tensor):
        return self.pretrained(X)


def get_model(args):
    """
        Creates a model with the specified arguments from the command line. 
    """
    model_type = args['model']
    num_classes = args['num_classes']
    finetune = args['pt_ft'] == 1
    selective = args['selective']

    if model_type == 'baseline':
        return baseline(num_classes)
    elif model_type == 'custom':
        return custom(num_classes)
    elif model_type == 'resnet':
        return resnet(finetune, num_classes, selective)
    elif model_type == 'vgg':
        return vgg(finetune, num_classes, selective)
    else:
        raise NotImplementedError
