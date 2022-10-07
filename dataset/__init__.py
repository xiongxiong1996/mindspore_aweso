import mindspore.dataset.transforms.py_transforms as transforms
import mindspore.dataset.vision.py_transforms as py_vision
from dataset.dataset import GetCUBDataset
import mindspore.dataset as ds

# Use Pytorch   set transform
# ----------------------------------------------------------------------
# import torch
# from torchvision import transforms
# normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                                  std=[0.5, 0.5, 0.5])
#
# normalize_v2 = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#
# train_transform_v1 = transforms.Compose([
#                       transforms.RandomResizedCrop(448),
#                       transforms.RandomHorizontalFlip(),
#                       transforms.ToTensor(),
#                       normalize,
#                       ])
#
# train_transform_v2 = transforms.Compose([
#                       transforms.Resize(512),
#                       transforms.RandomCrop(448),
#                       transforms.RandomHorizontalFlip(),
#                       transforms.ToTensor(),
#                       normalize,
#                       ])
#
# test_transform = transforms.Compose([
#                      transforms.Resize(512),
#                      transforms.CenterCrop(448),
#                      transforms.ToTensor(),
#                      normalize
#                      ])
#
# test_transform_v2 = transforms.Compose([
#                      transforms.Resize((448, 448)),
#                      transforms.ToTensor(),
#                      normalize
#                      ])
# ----------------------------------------------------------------------

# Use MindSpore    set transform
# ----------------------------------------------------------------------
normalize = py_vision.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
normalize_v2 = py_vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_transform_v1 = transforms.Compose([
    py_vision.RandomResizedCrop(448),
    py_vision.RandomHorizontalFlip(),
    py_vision.ToTensor(),
    normalize,
])

train_transform_v2 = transforms.Compose([
    py_vision.Resize(512),
    py_vision.RandomCrop(448),
    py_vision.RandomHorizontalFlip(),
    py_vision.ToTensor(),
    normalize,
])

test_transform = transforms.Compose([
    py_vision.Resize(512),
    py_vision.CenterCrop(448),
    py_vision.ToTensor(),
    normalize
])

test_transform_v2 = transforms.Compose([
    py_vision.Resize((448, 448)),
    py_vision.ToTensor(),
    normalize
])


# ----------------------------------------------------------------------

def get_loader(args, shuffle=True, train=True):
    data_transform = train_transform_v2 if train else test_transform

    # load dataset
    if args.data == "CUB":
        print("load CUB dataset")
        dataset_generator = GetCUBDataset(root=args.data_root, transform=data_transform, train=train)
        dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=shuffle)
    elif args.data == "AIR":
        print("load AIR dataset")
    else:
        print("error:unknow dataset")
        return

    # dataloader
    dataset = dataset.batch(batch_size=args.batch_size)
    iterator = dataset.create_dict_iterator()
    # print(next(iter(iterator)))

    return iterator
