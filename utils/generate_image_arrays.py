"""
Save images as pickle files for faster loading time in the PyTorch dataset
"""

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dset
import pickle

from config import ensure_dir


class CustomImageFolder(dset.ImageFolder):
    r"""
    Extend CustomImageFolder class to return the filename (in particular, the id of the image)
    """
    def __getitem__(self, index):
        """
        Overrides __getitem__ operator in partent class
        """
        return super(CustomImageFolder, self).__getitem__(index)[0], \
               self.imgs[index][0].split("/")[-1].split(".")[0]  # return image path


def extract_split(dataloader):
    r"""
    Generate arrays for the current split saved in dataloader
    :param dataloader: dataloader for all the images belonging
    :return:
    """
    img_dict = dict()
    for i, data in enumerate(dataloader):
        # Logging
        if i % (5000 // 16) == 0:
            print(f"{i}/{len(dataloader)}")
        
        # get image content and id
        img, image_ids = data
        img = img * 2 - 1  # normalize image values in [-1,+1]
        
        # store numpy arrays in the img_dict for every image in the batch
        for i, image_id in enumerate(image_ids):
            img_dict[image_id] = img[i].numpy()
    
    return img_dict

def main_generate_image_arrays():
    r"""
    Run main script to generate image numpy arrays out of raw images.
    :return:
    """
    # Import image as grayscale, and convert to tensors
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    split_names = ['valid', 'test', 'train']
    path_dataset = f"dataset/0.001/"
    ensure_dir(path_dataset)
    path_files = {
        0: path_dataset + "{}_images.pickle".format(split_names[0]),
        1: path_dataset + "{}_images.pickle".format(split_names[1]),
        2: path_dataset + "{}_images.pickle".format(split_names[2]),
    }
    
    for i, split in enumerate(split_names):
        dataset = CustomImageFolder(root=f'dataset/0.001/{split}', transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)  # increase batchsize for faster processing
        print("Dataset split -> {}: {}".format(split, len(dataset)))
        print(f"Starting visual features generation for {split}...\n")
        
        images = extract_split(dataloader)
        f = open(path_files[i], 'wb')
        pickle.dump(images, f)
        

if __name__ == '__main__':
    main_generate_image_arrays()

