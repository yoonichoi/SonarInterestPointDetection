import torch
import torch.nn as nn
from utils import SpaceToDepth


# Cross Entropy Loss
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        return self.loss_fn(logits, target)

class DetectorLoss(nn.Module):
    '''
    Calculates the Error using the CrossEntropyLoss
    Typically the grid_size is 8 so that can do that mapping in the comments
    '''

    def __init__(self, grid_size):
        super(DetectorLoss, self).__init__()

        self.grid_size = grid_size
        self.s2d = SpaceToDepth(self.grid_size)

    def forward(self, logits, keypoint_map):
        '''
        :param logits: Keypoint output from network is format B,C=65,H/8,W/8
        :param keypoint_map: Ground truth keypoint map is of format 1,H,W
        :param valid_mask:
        :return:
        '''
        # Model outputs to size C=65,H/8,W/8 . The 65 channels represent the 8x8 grid in the full scale version, +1
        # for the no keypoint bin

        # Modify keypoint map to correct size
        labels = keypoint_map
        labels = labels[:, None, :, :]
        # Convert from 1xHxW to 64xH/8xW/8
        labels = self.s2d.forward(labels)

        new_shape = labels.shape
        new_shape = torch.Size((new_shape[0], 1, new_shape[2], new_shape[3]))

        # Add an extra channel for the no_keypoint_bin i.e channel 65 with all 1(ones)
        # And ensure all the keypoint locations have value 2 not 1
        labels = torch.cat((2 * labels, torch.ones(new_shape, device=labels.device)), dim=1)
        # labels is now size B,C=65,H,W
        labels = labels.squeeze(0)

        # Take the argmax of the channels dimension. If there was a keypoint at a channel location then it has the
        # value 2 so its index is the max. If instead though as in most cases there is no keypoint then it will return
        # the 65 channel so index 64
        labels = torch.argmax(labels, dim=1)

        loss_fn = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fn(logits, labels)
        
        return loss.mean()
