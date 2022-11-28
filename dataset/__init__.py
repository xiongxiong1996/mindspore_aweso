import mindspore.dataset.transforms.py_transforms as transforms
import mindspore.dataset.vision.py_transforms as py_vision
from dataset.dataset import GetCUBDataset
import mindspore.dataset as ds

# 数据处理定义------------------------------------------------------------------------------------------------------------
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


# 读取数据---------------------------------------------------------------------------------------------------------------
def get_loader(args, shuffle=True, train=True):
    '''
    读取数据
    @param args:基础参数，可得到数据集名字及batch_size
    @param shuffle:是否进行shuffle处理，默认为True
    @param train:是否读取训练数据集，默认为True。 注：当选择False时会读取测试数据集
    @return:
    '''

    data_transform = train_transform_v2 if train else test_transform # 进行数据增强。 归一化、resize、裁剪
    # 判断是那个数据集，进行数据集读取---------------------------------------------------------------------------------------
    if args.data == "CUB":
        print("load CUB dataset")
        dataset_generator = GetCUBDataset(root=args.data_root, transform=data_transform, train=train)
        dataset = ds.GeneratorDataset(source=dataset_generator, column_names=["data", "label"], shuffle=shuffle)
    elif args.data == "AIR":
        print("Undefined. Coming soon")
    elif args.data == "CAR":
        print("Undefined. Coming soon")
    elif args.data == "DOG":
        print("Undefined. Coming soon")
    else:
        print("error:unknow dataset")
        return
    # 定义batch_size, mindspore的特有写法---------------------------------------------------------------------------------
    dataset = dataset.batch(batch_size=args.batch_size)
    return dataset
