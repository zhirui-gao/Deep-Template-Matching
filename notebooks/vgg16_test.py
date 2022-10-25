import torch
import math
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(in_channels=121, out_channels=128, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(128, eps=1e-05)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05)

        self.fc1 = nn.Linear(121*64, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels != m.out_channels or m.out_channels!=m.groups or m.bias is not None:
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    print('Not initializing')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x size is [bs,C,H,W]
        # x is corr
        # L2 l2_normalize
        x = nn.functional.normalize(x, p=2, dim=1)
        print(x.requires_grad)
        n, c, w, h = x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = torch.flatten(x,start_dim=1)

        out = self.fc1(x)

        return out




a = torch.randn(252,121,256)
b = torch.randn(252,121,256)

M, WW, C = a.shape
W = int(math.sqrt(WW))
sim_matrix = torch.einsum('mlc,mrc->mlr', a, b)  # 252,121,121
sim_matrix = sim_matrix.view(M,W,W,-1).permute(0,3,1,2)

my_net = Net()
out = my_net(sim_matrix)
print(out.shape)
# softmax_temp = 1. / C ** .5

# heatmap = torch.softmax(softmax_temp * sim_matrix, dim=2).view(M, W, W,-1)
# print(heatmap.shape)


