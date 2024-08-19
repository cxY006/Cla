import logging
import os
# from utils.augmentation import FixedRotation, Cutout, AddGaussianNoise
import re

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# from config import config
from dataset.randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

# cifar10_mean = (0.4914, 0.4822, 0.4465)
# cifar10_std = (0.2471, 0.2435, 0.2616)
# cifar100_mean = (0.5071, 0.4867, 0.4408)
# cifar100_std = (0.2675, 0.2565, 0.2761)
UCM_mean = (0.485, 0.456, 0.406)
UCM_std = (0.229, 0.224, 0.225)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_breast():
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.CenterCrop(size=256),
        # transforms.RandomRotation(degrees=45),
        # transforms.RandomCrop(size=32,
        #                       padding=int(32 * 0.125),
        #                       padding_mode='reflect'),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(0.15, 0.15, 0.15, 0.075),  # 照明度
        transforms.RandomAffine(degrees=30),  # 随机仿射变换
        transforms.RandomPerspective(),  # 透视
        transforms.CenterCrop(size=256),
        transforms.ToTensor(),
        transforms.Normalize(mean=UCM_mean, std=UCM_std)
    ])  # 有标签数据集的标准化
    transfortm_val = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.CenterCrop(size=256),
        # transforms.RandomRotation(degrees=45),
        # transforms.RandomCrop(size=32,
        #                       padding=int(32 * 0.125),
        #                       padding_mode='reflect'),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(0.15, 0.15, 0.15, 0.075),  # 照明度
        transforms.RandomAffine(degrees=30),  # 随机仿射变换
        transforms.RandomPerspective(),  # 透视
        transforms.CenterCrop(size=256),
        transforms.ToTensor(),
        transforms.Normalize(mean=UCM_mean, std=UCM_std)
    ])  # 验证集的标准化

    # labeled_path = config.breast_labeled_path
    # labeled_path = './data/NWPU/10'
    labeled_path = '/data/code/aid/build/2'
    train_labeled_dataset = breastSet(labeled_path, transform=transform_labeled)

    # unlabeled_path = config.breast_Unlabeled_path
    # unlabeled_path = './data/NWPU/8ulabled'
    unlabeled_path = '/data/code/aid/build/2unlabeled'
    train_unlabeled_dataset = breastSet(unlabeled_path,
                                        transform=TransformFixMatch(mean=UCM_mean, std=UCM_std),label_dict=get_label_dict(train_labeled_dataset.images))

    # test_path = config.breast_test
    # test_path = './data/NWPU/test'
    test_path = '/data/code/ycx/aid/build/test'
    test_dataset = breastSet(test_path, transform=transfortm_val,label_dict=get_label_dict(train_labeled_dataset.images))

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_images_and_labels(dir_path):
    images_list = []  # 文件名列表
    labels_list = []  # 标签列表

    for i in os.listdir(dir_path):
        images_list.append(i)
    print("images_list_len:", len(images_list))
    for i in images_list:
        # if 'airplane' in i:
        #     labels_list.append(0)
        # elif 'airport' in i:
        #     labels_list.append(1)
        # elif 'baseball_diamond' in i:
        #     labels_list.append(2)
        # elif 'basketball_court' in i:
        #     labels_list.append(3)
        # elif 'beach' in i:
        #     labels_list.append(4)
        # elif 'bridge' in i:
        #     labels_list.append(5)
        # elif 'chaparral' in i:
        #     labels_list.append(6)
        # elif 'church' in i:
        #     labels_list.append(7)
        # elif 'circular_farmland' in i:
        #     labels_list.append(8)
        # elif 'cloud' in i:
        #     labels_list.append(9)
        # elif 'commercial_area' in i:
        #     labels_list.append(10)
        # elif 'dense_residential' in i:
        #     labels_list.append(11)
        # elif 'desert' in i:
        #     labels_list.append(12)
        # elif 'forest' in i:
        #     labels_list.append(13)
        # elif 'freeway' in i:
        #     labels_list.append(14)
        # elif 'golf_course' in i:
        #     labels_list.append(15)
        # elif 'ground_track_field' in i:
        #     labels_list.append(16)
        # elif 'harbor' in i:
        #     labels_list.append(17)
        # elif 'industrial_area' in i:
        #     labels_list.append(18)
        # elif 'intersection' in i:
        #     labels_list.append(19)
        # elif 'island' in i:
        #     labels_list.append(20)
        # elif 'lake' in i:
        #     labels_list.append(21)
        # elif 'meadow' in i:
        #     labels_list.append(22)
        # elif 'medium_residential' in i:
        #     labels_list.append(23)
        # elif 'mobile_home_park' in i:
        #     labels_list.append(24)
        # elif 'mountain' in i:
        #     labels_list.append(25)
        # elif 'overpass' in i:
        #     labels_list.append(26)
        # elif 'palace' in i:
        #     labels_list.append(27)
        # elif 'parking_lot' in i:
        #     labels_list.append(28)
        # elif 'railway' in i:
        #     labels_list.append(29)
        # elif 'railway_station' in i:
        #     labels_list.append(30)
        # elif 'rectangular_farmland' in i:
        #     labels_list.append(31)
        # elif 'river' in i:
        #     labels_list.append(32)
        # elif 'roundabout' in i:
        #     labels_list.append(33)
        # elif 'runway' in i:
        #     labels_list.append(34)
        # elif 'sea_ice' in i:
        #     labels_list.append(35)
        # elif 'ship' in i:
        #     labels_list.append(36)
        # elif 'snowberg' in i:
        #     labels_list.append(37)
        # elif 'sparse_residential' in i:
        #     labels_list.append(38)
        # elif 'stadium' in i:
        #     labels_list.append(39)
        # elif 'storage_tank' in i:
        #     labels_list.append(40)
        # elif 'tennis_court' in i:
        #     labels_list.append(41)
        # elif 'terrace' in i:
        #     labels_list.append(42)
        # elif 'thermal_power_station' in i:
        #     labels_list.append(43)
        # elif 'wetland' in i:
        #     labels_list.append(44)
        # elif 'unlabeled' in i:
        #     labels_list.append(45)
        if 'agricultural' in i:
            labels_list.append(0)
        elif 'airplane' in i:
            labels_list.append(1)
        elif 'baseballdiamond' in i:
            labels_list.append(2)
        elif 'beach' in i:
            labels_list.append(3)
        elif 'buildings' in i:
            labels_list.append(4)
        elif 'chaparral' in i:
            labels_list.append(5)
        elif 'denseresidential' in i:
            labels_list.append(6)
        elif 'forest' in i:
            labels_list.append(7)
        elif 'freeway' in i:
            labels_list.append(8)
        elif 'golfcourse' in i:
            labels_list.append(9)
        elif 'harbor' in i:
            labels_list.append(10)
        elif 'intersection' in i:
            labels_list.append(11)
        elif 'mediumresidential' in i:
            labels_list.append(12)
        elif 'mobilehomepark' in i:
            labels_list.append(13)
        elif 'overpass' in i:
            labels_list.append(14)
        elif 'parkinglot' in i:
            labels_list.append(15)
        elif 'river' in i:
            labels_list.append(16)
        elif 'runway' in i:
            labels_list.append(17)
        elif 'sparseresidential' in i:
            labels_list.append(18)
        elif 'storagetanks' in i:
            labels_list.append(19)
        elif 'tenniscourt' in i:
            labels_list.append(20)
        elif 'unlabelled' in i:
            labels_list.append(21)
    print("labels_list_len:", len(labels_list))

    return images_list, labels_list

    # malignant,benign,normal

def get_label_dict(images_list):
    label_set = set()
    for i in images_list:
        label_name = getLabel(i)
        if label_name:
            label_set.add(label_name)
    tar_list = list(label_set)
    tar_list.sort()
    label_map = {}
    for i, tar in enumerate(tar_list):
        label_map[tar] = i
    return label_map


def get_images_and_labels_auto(dir_path,label_dict=None):
    images_list = []  # 文件名列表
    labels_list = []  # 标签列表

    for i in os.listdir(dir_path):
        images_list.append(i)
    print("images_list_len:", len(images_list))

    if not label_dict:
        label_dict = get_label_dict(images_list)

    for i in images_list:
        label_name = getLabel(i)
        if label_name in label_dict:
            labels_list.append(label_dict[label_name])
        else:
            labels_list.append(len(label_dict))

    print("labels_list_len:", len(labels_list))

    assert len(images_list) == len(labels_list)
    return images_list, labels_list

def getLabel(imgName):
    pattern = r'^[a-zA-Z]+'
    match = re.search(pattern,imgName)
    if match:
        return match.group()
    else:
        return None


class breastSet(Dataset):
    def __init__(self, dir_path, transform=None,label_dict=None):
        self.dir_path = dir_path  # 数据集根目录
        print("dir_path:", self.dir_path)
        self.transform = transform
        self.images, self.labels = get_images_and_labels_auto(self.dir_path,label_dict)
        # self.images, self.labels = get_images_and_labels(self.dir_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.dir_path, img_name)

        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        # 读取图片，np.fromfile解决路径中含有中文的问题

        # img = torch.from_numpy(img)  # Numpy需要转成torch之后才可以使用transform
        # img = img.permute(2, 0, 1)
        img = Image.fromarray(img)  # 实现array到image的转换，Image可以直接用transform
        img = self.transform(img)  # 重点！！！如果为无标签的一致性正则化，那么此处会返回两个图   img即为一个list
        label = self.labels[index]
        return img, label
    # 获取标签数量
    def getLabelNum(self):
        return len(set(self.labels))


class TransformFixMatch(object):

    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.CenterCrop(size=256),
            # transforms.RandomRotation(degrees=45),
            # transforms.RandomCrop(size=32,
            #                       padding=int(32 * 0.125),
            #                       padding_mode='reflect'),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(0.15, 0.15, 0.15, 0.075),  # 照明度
            transforms.RandomAffine(degrees=30),  # 随机仿射变换
            transforms.RandomPerspective(),
            transforms.CenterCrop(size=256)])  # 弱增强

        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=256),
            # transforms.RandomRotation(degrees=45),
            # transforms.RandomCrop(size=32,
            #                       padding=int(32 * 0.125),
            #                       padding_mode='reflect'),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(0.15, 0.15, 0.15, 0.075),  # 照明度
            transforms.RandomAffine(degrees=30),  # 随机仿射变换
            transforms.RandomPerspective(),  # 透视
            transforms.CenterCrop(size=256),
            # transforms.Resize(256),
            RandAugmentMC(n=2, m=10)])  # 强增强，比弱增强多了两种图像失真处理

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        # 将弱增强后的图  强增强的图  分别进行标准化
        return self.normalize(weak), self.normalize(strong)  # 返回一对弱增强、强增强


if __name__ == '__main__':
    labeled_dataset, unlabeled_dataset, test_dataset = get_breast()
    labeled_trainloader = DataLoader(
        labeled_dataset,
        batch_size=4,
        drop_last=True)
    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        batch_size=4,
        drop_last=True)
    test_trainloader = DataLoader(
        test_dataset,
        batch_size=1,
        drop_last=True)
    for i, (img, label) in enumerate(labeled_trainloader):
        print("load labeled_datatset!")
        print(img.shape, label)
    for (img1, img2), y in unlabeled_trainloader:
        print("load unlabeled_datatset!")
        print(img1.shape, img2.shape, y)
    for (img, label) in test_trainloader:
        print("load test_datatset!")
        print(img.shape, label)

DATASET_GETTERS = {'ucm': get_breast}
