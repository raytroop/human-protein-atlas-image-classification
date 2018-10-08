import random
import os
import pandas as pd
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

label_names = {
    0:  "Nucleoplasm",
    1:  "Nuclear membrane",
    2:  "Nucleoli",
    3:  "Nucleoli fibrillar center",
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",
    6:  "Endoplasmic reticulum",
    7:  "Golgi apparatus",
    8:  "Peroxisomes",
    9:  "Endosomes",
    10:  "Lysosomes",
    11:  "Intermediate filaments",
    12:  "Actin filaments",
    13:  "Focal adhesion sites",
    14:  "Microtubules",
    15:  "Microtubule ends",
    16:  "Cytokinetic bridge",
    17:  "Mitotic spindle",
    18:  "Microtubule organizing center",
    19:  "Centrosome",
    20:  "Lipid droplets",
    21:  "Plasma membrane",
    22:  "Cell junctions",
    23:  "Mitochondria",
    24:  "Aggresome",
    25:  "Cytosol",
    26:  "Cytoplasmic bodies",
    27:  "Rods & rings"
}


def imgRead(in_path):
    r_img = Image.open(in_path.replace('.rgb', '_red.png'))
    g_img = Image.open(in_path.replace('.rgb', '_green.png'))
    b_img = Image.open(in_path.replace('.rgb', '_blue.png'))
    rgb_arr = np.stack([r_img, g_img, b_img], -1)
    return Image.fromarray(rgb_arr)


class ProteinDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """

    def __init__(self, df, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.imgnames = df.path.tolist()
        self.labels = None if 'target_vec' not in df.columns else np.array(df.target_vec.values.tolist(), dtype=np.float32)
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.imgnames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        """
        img_path = self.imgnames[idx]
        image = imgRead(img_path)  # PIL image

        if self.labels is not None:
            label = self.labels[idx]
            return self.transform(image), torch.from_numpy(label)

        return self.transform(image)


def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    assert set(types) <= set(
        ['train', 'val', 'test']), "data types have to be among {'train', 'val', 'test'}"

    train_transformer = transforms.Compose([
        # resize the image to 64x64 (remove if images are already 64x64)
        # transforms.Resize(256),
        transforms.RandomRotation(20.0),
        transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
        transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.Resize((params.image_size, params.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=params.mean,
                             std=params.std)])  # transform it into a torch tensor

    # loader for evaluation, no horizontal flip
    eval_transformer = transforms.Compose([
        # resize the image to 64x64 (remove if images are already 64x64)
        transforms.Resize((params.image_size, params.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=params.mean,
                             std=params.std)])  # transform it into a torch tensor

    if 'test' in types:
        test_image_dir = os.path.join(data_dir, 'test')
        test_df = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
        test_df['path'] = test_df['Id'].map(lambda x: os.path.join(test_image_dir, '{}.rgb'.format(x)))

    if 'train' in types:
        train_image_dir = os.path.join(data_dir, 'train')
        # https://www.kaggle.com/kmader/rgb-transfer-learning-with-vgg16-for-protein-atlas
        train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        train_df['path'] = train_df['Id'].map(
            lambda x: os.path.join(train_image_dir, '{}.rgb'.format(x)))
        train_df['target_list'] = train_df['Target'].map(
            lambda x: [int(a) for a in x.split(' ')])

        # create a categorical vector
        train_df['target_vec'] = train_df['target_list'].map(
            lambda ck: [int(i in ck) for i in range(28)])

        raw_train_df, valid_df = train_test_split(train_df,
                                                  test_size=0.3,
                                                  # hack to make stratification work
                                                  stratify=train_df['target_list'].map(lambda x: np.random.choice(x) if 27 not in x else 27))

        # keep labels with more than 500 objects
        out_df_list = []
        for k in range(28):
            keep_rows = raw_train_df['target_list'].map(lambda x: k in x)
            out_df_list += [raw_train_df[keep_rows].sample(params.imgs_per_cat,
                                                           replace=True)]
        train_df = pd.concat(out_df_list, ignore_index=True)

        # _, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
        # train_sum_vec = np.sum(np.stack(train_df['target_vec'].values, 0), 0)
        # valid_sum_vec = np.sum(np.stack(valid_df['target_vec'].values, 0), 0)
        # ax1.barh(list(label_names.keys()), train_sum_vec)
        # ax1.set_title('Training Distribution')
        # ax2.barh(list(label_names.keys()), valid_sum_vec)
        # ax2.set_title('Validation Distribution')
        # plt.waitforbuttonpress()

    dataloaders = {}
    for split in set(types):
        # use the train_transformer if training data, else use eval_transformer without random flip
        if split == 'train':
            dl = DataLoader(ProteinDataset(train_df, train_transformer),
                            batch_size=params.batch_size,
                            shuffle=True,
                            num_workers=params.num_workers,
                            pin_memory=params.cuda)
            dataloaders[split] = dl
        elif split == 'val':
            dl = DataLoader(ProteinDataset(valid_df, eval_transformer),
                            batch_size=params.batch_size,
                            shuffle=False,
                            num_workers=params.num_workers,
                            pin_memory=params.cuda)
            dataloaders[split] = dl
        else:
            dl = DataLoader(ProteinDataset(test_df, eval_transformer),
                            batch_size=params.batch_size,
                            shuffle=False,
                            num_workers=params.num_workers,
                            pin_memory=params.cuda)
            dataloaders[split] = dl

    return dataloaders
