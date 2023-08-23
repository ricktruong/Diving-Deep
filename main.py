import torch
import argparse
import os, sys, json
from datetime import datetime
from data import get_dataloaders
from engine import *

parser = argparse.ArgumentParser()

parser.add_argument('--log', default=1, type=int,
                    help='Determine if we log the outputs and experiment configurations to local disk')
parser.add_argument('--path', default=datetime.now().strftime('%Y-%m-%d-%H%M%S'), type=str,
                    help='Default log/model output path if not specified')
parser.add_argument('--device_id', default=0, type=int,
                    help='the id of the gpu to use')    
# Model Related
parser.add_argument('--model', default='vgg', type=str,
                    help='Model being used')
parser.add_argument('--pt_ft', default=1, type=int,
                    help='Determine if the model is for partial fine-tune mode (pretrained weights not frozen)')
parser.add_argument('--model_dir', default=None, type=str,
                    help='Load some saved parameters for the current model')
parser.add_argument('--num_classes', default=20, type=int,
                    help='Number of classes for classification')
parser.add_argument('--selective', default=0, type=int,
                    help='Selective freezing of first layer')

# Data Related
parser.add_argument('--bz', default=32, type=int,
                    help='batch size')
parser.add_argument('--shuffle_data', default=True, type=bool,
                    help='Shuffle the data')
parser.add_argument('--normalization_mean', default=(0.485, 0.456, 0.406), type=tuple,
                    help='Mean value of z-scoring normalization for each channel in image')
parser.add_argument('--normalization_std', default=(0.229, 0.224, 0.225), type=tuple,
                    help='Mean value of z-scoring standard deviation for each channel in image')
parser.add_argument('--augmentation', default=0, type=int)

# feel free to add more augmentation/regularization related arguments

# Other Choices & hyperparameters
parser.add_argument('--epoch', default=25, type=int,
                    help='number of epochs')
    # for loss
parser.add_argument('--criterion', default='cross_entropy', type=str,
                    help='which loss function to use')
    # for optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    help='which optimizer to use')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--dampening', default=0, type=float,
                    help='dampening for momentum')
parser.add_argument('--nesterov', default=False, type=bool,
                    help='enables Nesterov momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay')
# for scheduler
parser.add_argument('--lr_scheduler', default='steplr', type=str,
                    help='learning rate scheduler')
parser.add_argument('--step_size', default=7, type=int,
                    help='Period of learning rate decay')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Multiplicative factor of learning rate decay.')

# for generalization / stopping
parser.add_argument('--early_stop', default=False, type=bool,
                    help='Stop early if validation acc goes up for patience epochs.')
parser.add_argument('--patience', default=3, type=int,
                    help='Number of epochs to wait for val acc to improve before early stopping.')

# testing
parser.add_argument('--test', default=1, type=int,
                    help='Test pretrained model only.')

# feel free to add more arguments if necessary
args = vars(parser.parse_args())

def main(args):

    # Sets the device to run on 
    device = torch.device("cuda:{}".format(args['device_id']) if torch.cuda.is_available() else "cpu")
    
    # create dataloaders for the model 
    dataloaders = get_dataloaders('food-101/train.csv', 'food-101/test.csv', args)

    # get the model and its criterion, optimizer, and scheduler
    model, criterion, optimizer, lr_scheduler = prepare_model(device, args)

    # if test, we train then test
    if args['test']:
        model = train_model(model, criterion, optimizer, lr_scheduler, device, dataloaders, args)
        loss, acc = test_model(model, criterion, device, dataloaders, args)
        print(f'Test Loss: {loss.item():.3f}')
        print(f'Test Acc:  {acc.item():.5f}')

    # otherwise just train
    else: 
        model = train_model(model, criterion, optimizer, lr_scheduler, device, dataloaders, args)

if __name__ == '__main__':
    main(args)
