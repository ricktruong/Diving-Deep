from numpy import genfromtxt
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch

########## DO NOT change this function ##########
# If you change it to achieve better results, we will deduct points. 
def train_val_split(train_dataset):
    train_size = int(len(train_dataset) * 0.8)
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size], 
                                            generator=torch.Generator().manual_seed(42))
    return train_subset, val_subset
#################################################

########## DO NOT change this variable ##########
# If you change it to achieve better results, we will deduct points. 
transform_test = transforms.Compose([
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)
#################################################

class FoodDataset(Dataset):
    def __init__(self, data_csv, transforms=None):
        self.data = genfromtxt(data_csv, delimiter=',', dtype=str)
        self.transforms = transforms
        
    def __getitem__(self, index):
        fp, _, idx = self.data[index]
        idx = int(idx)
        img = Image.open(fp)
        if self.transforms is not None:
            img = self.transforms(img)
        return (img, idx)

    def __len__(self):
        return len(self.data)

def get_dataset(csv_path, transform):
    return FoodDataset(csv_path, transform)

def create_dataloaders(train_set: FoodDataset, val_set: FoodDataset, test_set: FoodDataset, args=None):
    """
        Creates dataloaders based on the arguments passed in for the train, validation, and test set
        specified. 
    """
    return tuple(
        DataLoader(
            x,
            batch_size=args['bz'],
            shuffle=args['shuffle_data'],
            pin_memory=True,
            num_workers=4
        )
        for x in (train_set, val_set, test_set)
    )

def get_dataloaders(train_csv, test_csv, args=None):
    """
        Transforms the datasets based on specified augmentation and returns the dataloaders
        using create_dataloaders().
    """ 

    # if we want to augment, then apply these transforms
    if args['augmentation']:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.RandomAffine(45),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(mean=args['normalization_mean'], std=args['normalization_std'])]
        )
    
    # otherwise, just apply the same transforms as test set
    else: 
        transform_train = transform_test

    train_dataset = get_dataset(train_csv, transform_train)

    ########## DO NOT change the following two lines ##########
    # If you change it to achieve better results, we will deduct points. 
    test_dataset = get_dataset(test_csv, transform_test)
    train_set, val_set = train_val_split(train_dataset)
    ###########################################################

    dataloaders = create_dataloaders(train_set, val_set, test_dataset, args)
    return dataloaders
