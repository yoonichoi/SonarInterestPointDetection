# Dataset
import numpy as np
import torch
import cv
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from utils import add_keypoint_map


# Dataset classes
class SonarDataset(torch.utils.data.Dataset):
    def __init__(self, root, partition= "train"): 
        '''
        Initializes the dataset.
        '''

        # Load the directory and all files in them
        self.train_samples = []

        if partition == "train":
          self.img_dir = root + "/train/img/"
          self.pts_dir = root + "/train/pts/"
        elif partition == 'val':
          self.img_dir = root + "/valid/img/"
          self.pts_dir = root + "/valid/pts/"
        else:
          self.img_dir = root + "/test/img/"
          self.pts_dir = root + "/test/pts/"

        
        '''
        You may decide to do this in __getitem__ if you wish.
        However, doing this here will make the __init__ function take the load of
        loading the data, and shift it away from training.
        '''

        img_files = glob.glob(self.img_dir+'/*.png')
        pt_files = glob.glob(self.pts_dir+'/*.npy')

        for idx in tqdm(range(len(img_files))):
            sample = (img_files[idx], pt_files[idx])
            self.train_samples.append(sample)


    def __len__(self):
        return len(self.train_samples)


    def __getitem__(self, index):
        def _read_image(filename):
            image = cv2.imread(filename, 0)
            image = image.astype('float32')
            return image

        def _read_points(filename):
            return np.load(filename).astype(np.float32)
        
        sample = self.train_samples[index]
        image = _read_image(sample[0])
        pts = np.reshape(_read_points(sample[1]), [-1, 2])

        data = {'image': image, 'keypoints': pts}
        data = add_keypoint_map(data)

        # Convert to Tensors
        image = torch.from_numpy(data['image'])
        keypoint_map = torch.from_numpy(data['keypoint_map'])

        image = torch.unsqueeze(image, 0)

        return image, keypoint_map