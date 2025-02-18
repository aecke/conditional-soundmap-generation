#from mlxtend.data import loadlocal_mnist
from torch.utils import data
import numpy as np
from torchvision import transforms
from PIL import Image

from helper import label_to_tensor


class MnistDataset(data.Dataset):
    def __init__(self, data_folder, img_size):
        """
        :param data_folder:
        :param img_size: is of type list.
        """
        # self.data_folder = data_folder
        self.img_size = img_size
        self.imgs, self.labels = read_mnist(data_folder)
        self.transforms = transforms.Compose([transforms.Resize(img_size),
                                              transforms.ToTensor()])

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, item):
        img = self.imgs[item].reshape(28, 28) / 255  # normalize to 0-1 interval
        label = self.labels[item]

        # returning the torch tensors
        img_tensor = self.transforms(Image.fromarray(img))
        # return torch.from_numpy(img).unsqueeze(dim=0)  # adding the channel dimension
        return {'image': img_tensor,
                'label': label_to_tensor(label, self.img_size[0], self.img_size[1]),
                'label2': label}


def init_mnist_loader(mnist_folder, img_size, loader_params):
    dataset = MnistDataset(mnist_folder, img_size)
    data_loader = data.DataLoader(dataset, **loader_params)
    return data_loader


def read_mnist(mnist_folder, verbose=False):
    imgs_path = mnist_folder + '/train-images-idx3-ubyte'
    labels_path = mnist_folder + '/train-labels-idx1-ubyte'

    imgs, labels = loadlocal_mnist(imgs_path, labels_path)
    if verbose:
        print(f'In [read_mnist]: read images and labels of shape: {imgs.shape}, {labels.shape}')
        print('distribution:', np.bincount(labels))

    return imgs.astype(np.float32), labels


