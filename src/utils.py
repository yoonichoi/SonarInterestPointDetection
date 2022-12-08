import torch
import torch.nn as nn

def add_keypoint_map(data):
  image_shape = np.shape(data['image'])
  kp = np.floor(data['keypoints']).astype(np.int32)
  kmap = np.zeros(image_shape).astype(np.float32)
  for p in kp:
    kmap[p[0],p[1]] = 1.0
  return {**data, 'keypoint_map': kmap}

  
class SpaceToDepth(nn.Module):
    '''
    TF Op: tf.nn.space_to_depth
    '''
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.contiguous().view(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output
