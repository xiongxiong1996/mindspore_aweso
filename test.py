# -*-coding:utf-8-*-
import numpy as np
import mindspore.dataset as ds
# from torch.utils.data import Dataset
# import torch
# In MindSpore, GeneratorDataset generates data from Python by invoking Python data source each epoch.
# The column names and column types of generated dataset depend on Python data defined by users.
class GetDatasetGenerator:

    def __init__(self):
        np.random.seed(58)
        self.__data = np.random.sample((5, 2))
        self.__label = np.random.sample((5, 1))

    def __getitem__(self, index):
        return (self.__data[index], self.__label[index])

    def __len__(self):
        return len(self.__data)



dataset_generator = GetDatasetGenerator()
dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)

for data in dataset.create_dict_iterator():
    print(data["data"], data["label"])


# class GetDatasetGenerator1(Dataset):
#
#     def __init__(self):
#         np.random.seed(58)
#         self.__data = np.random.sample((5, 2))
#         self.__label = np.random.sample((5, 1))
#
#     def __getitem__(self, index):
#         return (self.__data[index], self.__label[index])
#
#     def __len__(self):
#         return len(self.__data)
#
#
# dataset = GetDatasetGenerator1()
# for item in dataset:
#     print("item:", item)