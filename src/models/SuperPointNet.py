import torch
import torch.nn as nn
from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
from src.models.unet_parts import *
import numpy as np
from src.models.model_utils import SuperPointNet_process
class SuperPointNet(torch.nn.Module):
  """ Pytorch definition of SuperPoint Network. """
  def __init__(self, config):
    super(SuperPointNet, self).__init__()
    self.relu = torch.nn.ReLU(inplace=True)
    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
    # Shared Encoder.
    self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
    self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
    self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
    self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
    self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
    self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
    self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
    self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
    # Detector Head.
    self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
    # Descriptor Head.
    self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
    self.output =None

  def forward(self, x):
    """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """
    # Shared Encoder.
    x = self.relu(self.conv1a(x))
    x = self.relu(self.conv1b(x))
    x = self.pool(x)
    x = self.relu(self.conv2a(x))
    x = self.relu(self.conv2b(x))
    x = self.pool(x)
    x = self.relu(self.conv3a(x))
    x = self.relu(self.conv3b(x))
    x = self.pool(x)
    x = self.relu(self.conv4a(x))
    x = self.relu(self.conv4b(x))
    # Detector Head.
    cPa = self.relu(self.convPa(x))
    semi = self.convPb(cPa)

    # Descriptor Head.
    cDa = self.relu(self.convDa(x))
    desc = self.convDb(cDa)
    dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
    desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
    output = {'semi': semi, 'desc': desc}
    self.output = output
    return output

  def process_output(self, sp_processer, output, choice=True):
    """
    input:
      N: number of points
    return: -- type: tensorFloat
      pts: tensor [batch, N, 2] (no grad)  (x, y)
      pts_offset: tensor [batch, N, 2] (grad) (x, y)
      pts_desc: tensor [batch, N, 256] (grad)
    """
    from src.utils.utils import flattenDetection

    # from models.model_utils import pred_soft_argmax, sample_desc_from_points
    semi = output['semi']
    desc = output['desc']  # batch_size 256 H W
    # flatten
    heatmap = flattenDetection(semi)  # N x 65 x H/8 x W/8 -> [batch_size, 1, H, W]
    if choice:
      # nms
      heatmap_nms_batch = sp_processer.heatmap_to_nms(heatmap, tensor=True)
      # extract offsets
      outs = sp_processer.pred_soft_argmax(heatmap_nms_batch, heatmap)
      residual = outs['pred']
      # extract points
      # TODO :use softargmax to select top k
      outs = sp_processer.batch_extract_features(desc, heatmap_nms_batch, residual)

      # output.update({'heatmap': heatmap, 'heatmap_nms': heatmap_nms, 'descriptors': descriptors})
      output.update(outs)
      # output :{'pts_int': pts_int, 'pts_offset': pts_offset, 'pts_desc': pts_desc}
      # self.output = output
    else:
      pts_int, pts_offset, pts_desc = [], [], []
      batch_size = heatmap.shape[0]
      cell_size = int(8)
      H, W = desc.shape[2] * cell_size, desc.shape[3] * cell_size
      grid = np.mgrid[:H, 0:W]
      grid = grid.reshape((2, -1))
      grid = grid.transpose(1, 0)
      grid = grid[:, [0, 1]]
      pts_int_b = torch.tensor(grid).to(desc.device).float()
      res_b = torch.zeros((H * W, 2)).to(desc.device)  # tensor [N, 2(x,y)]
      for i in range(batch_size):
        pts_int_b = pts_int_b[:, [1, 0]]  # tensor [N, 2(x,y)]
        pts_b = pts_int_b + res_b  # important
        pts_desc_b = SuperPointNet_process.sample_desc_from_points(desc[i].unsqueeze(0), pts_b).squeeze(0)
        pts_int.append(pts_int_b)
        pts_offset.append(res_b)
        pts_desc.append(pts_desc_b)
      pts_int = torch.stack((pts_int), dim=0)
      pts_offset = torch.stack((pts_offset), dim=0)
      pts_desc = torch.stack((pts_desc), dim=0)
      pts_offset = pts_offset.fill_(0)
      outs = {'pts_int': pts_int, 'pts_offset': pts_offset, 'pts_desc': pts_desc}
      output.update(outs)

      pass
    return output

  def draw_keypoint(self, outs_post, data):
    from src.utils.print_tool import print_dict_attr
    # print_dict_attr(outs_post, 'shape')
    from src.utils.draw import draw_keypoints
    from src.utils.utils import toNumpy
    pts_int = outs_post['pts_int']
    pts_offset = outs_post['pts_offset']
    pts_desc = outs_post['pts_desc']
    for i in range(2):
      img = draw_keypoints(toNumpy(data["image" + f"{i}"].squeeze()),
                           toNumpy((pts_int[i] + pts_offset[i]).squeeze()).transpose())
      # print("img: ", img_0)
      plt.imshow(img)
      plt.show()
    from src.models.model_wrap import PointTracker

    tracker = PointTracker(max_length=2, nn_thresh=0.7)

    for i in range(2):
      f = lambda x: toNumpy(x.squeeze())
      tracker.update(f(pts_int[i]).transpose(), f(pts_desc[i]).transpose())

    matches = tracker.get_matches().T
    print("matches: ", matches.transpose().shape)

    from src.utils.draw import draw_matches
    # filename = path_match + '/' + f_num + 'm.png'
    draw_matches(f(data["image0"]), f(data["image1"]), matches, filename='', show=True)

