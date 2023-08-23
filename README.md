[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-f059dc9a6f8d3a56e377f745f24479a46679e63a5d9fe6f495e02850cd0d8118.svg)](https://classroom.github.com/online_ide?assignment_repo_id=6958498&assignment_repo_type=AssignmentRepo)
# Diving Deep, A Convolutional Neural Network 

To run the baseline model without any argument changes:
```
python3 main.py --model baseline
```

To run the custom model without any argument changes:
```
python3 main.py --model custom
```

To run the vgg model without any arguments changes:
```
python3 main.py --model vgg
```

To run the resnet model without any arguments changes:
```
python3 main.py --model resnet
```

To run with different arguments, they will be listed below: 

- "--log", default=1, Determines if we log the outputs and experiment configurations to local disk
- "--path", default=current date and time, Specifies the log/model output path
- "--device_id", default=0, Specifies the id of the GPU to use
- "--model", default=vgg, Specifies the model to be used to train/test
- "--pt_ft", default=1, If 1 is set, then partial finetune is on an all pretrained model weights are unfrozen, if 0, then all weights except last layer are frozen (aka not finetuned)
- "--model_dir", default=None, Loads saved parameters for the model
- "--num_classes", default=20, Specifies the number of classes for classification 
- "--selective", default=0, If 1, then selective weight freezing of the first layer, else if 0, then weights are left as is
- "--bz", default=32, Specifies the batch size
- "--shuffle_data", default=True, Specifies whether to shuffle the data or not
- "--normalization_mean", default=(0.485, 0.456, 0.406), Specifies a tuple of 3 normalization means to normalize dataset
- "--normalization_std", default=(0.229, 0.224, 0.225), Specifies a tuple of 3 normalization standard deviation to normalize dataset
- "--augmentation", default=0, If 0, then the augmentation for test model is used on train/val set data, if 1, then the provided resnet augmentation settings are used
- "--epoch", default=25, an integer to specify the amount of epochs to train the model
- "--criterion", default='cross_entropy', a string for which loss function to use
- "--optimizer", default='adam', a string for which optimizer to use when training
- "--lr", default=1e-3, a float to specify the learning rate
- "--momentum", default=0.9, a float to determine the gamma for momentum
- "--dampening", default=0, 1 to dampen momentum, 0 to not dampen
- "--nesterov", default=False, specifies whether to use Nesterov momentum or not
- "--weight_decay", default=1e-4, specifies the weight decay
- "--lr_scheduler", default='steplr', specifies the type of learning rate scheduler in the model
- "--step_size", default=7, an integer to specify the period of learning rate decay
- "--gamma", default=0.1, a float to specify the factor of learning rate decay
- "--early_stop", default=False, Stop early if validation accuracy goes up for patience amount of epochs
- "--patience", default=3, Specifies number of epochs to wait for validation accuracy to improve before early stopping
- "--test", default=1, 1 to test on the model, 0 to not

Examples: 
1) Typically, you'd want to specify a model, learning rate, and testing, you would enter into the command line,
```
python3 main.py --model custom --lr 1e-4 --test 1
```
2) Another example is if you want to train a custom model, but want to modify the amount of epochs to run, but still keep early stop on in case of overfitting, you can enter,
```
python3 main.py --model custom --epoch 30 --early_stop True
```
3) If you want to do transfer learning on the vgg16 or resnet18 pretrained models from PyTorch with finetuning, enter,
```
python3 main.py --model vgg --pt_ft 1
```
4) If you want to do transfer learning on the pretrained PyTorch models but with all the weights frozen, enter,
```
python3 main.py --model vgg --pt_ft 0
```
5) If you want to do transfer learning with selective weight freezing (our code freezes the first convolutional layer only), enter,
```
python3 main.py --model resnet --selective 1
```
