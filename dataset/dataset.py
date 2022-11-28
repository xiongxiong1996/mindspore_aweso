import os
import pandas as pd
from PIL import Image


def pil_loader(path):
    """
    读取图像，将图像转为RGB
    @param path:(str): Image path
    @return:(image, target) where target is class_index of the target class.
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class GetCUBDataset:
    def __init__(self, root, transform=None, train=False, loader=pil_loader):
        """
        获取CUB数据集
        @param root:
        @param transform:
        @param train:
        @param loader:
        """
        img_folder = os.path.join(root, "images")
        img_paths = pd.read_csv(os.path.join(root, "images.txt"), sep=" ", header=None, names=['idx', 'path'])
        img_labels = pd.read_csv(os.path.join(root, "image_class_labels.txt"), sep=" ", header=None,
                                 names=['idx', 'label'])
        train_test_split = pd.read_csv(os.path.join(root, "train_test_split.txt"), sep=" ", header=None,
                                       names=['idx', 'train_flag'])
        data = pd.concat([img_paths, img_labels, train_test_split], axis=1)
        data = data[data['train_flag'] == train]
        data['label'] = data['label'] - 1  # 0-199

        imgs = data.reset_index(drop=True)

        if len(imgs) == 0:
            raise (RuntimeError("no csv file"))
        self.root = img_folder
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.train = train

    def __getitem__(self, index):
        """
        获取item
        @param index: (int)Index
        @return: (image, target) where target is class_index of the target class.
        """
        item = self.imgs.iloc[index]
        file_path = item['path']
        target = item['label']
        img = self.loader(os.path.join(self.root, file_path))
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)
