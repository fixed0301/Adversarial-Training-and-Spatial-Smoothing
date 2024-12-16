import torch
from torchviz import make_dot


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func

    def forward(self, input):
        return self.func(input)


class Mymodel(nn.Module):
    def __init__(self, i_c=3, n_c=2):
        super(Mymodel, self).__init__()

        self.conv1 = nn.Conv2d(i_c, 32, 5, stride=1, padding=2, bias=True)
        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2), padding=0)

        self.conv2 = nn.Conv2d(32, 64, 5, stride=1, padding=2, bias=True)
        self.pool2 = nn.MaxPool2d((2, 2), stride=(2, 2), padding=0)

        self.flatten = Expression(lambda tensor: tensor.view(tensor.shape[0], -1))
        self.fc1 = nn.Linear(8 * 8 * 64, 1024, bias=True) # 28/ 4 = 7. (28*28 이미지?)
        self.fc2 = nn.Linear(1024, n_c)

    def forward(self, x_i, _eval=False):

        if _eval:
            # switch to eval mode
            self.eval()
        else:
            self.train()

        x_o = self.conv1(x_i)
        x_o = torch.relu(x_o)
        x_o = self.pool1(x_o)

        x_o = self.conv2(x_o)
        x_o = torch.relu(x_o)
        x_o = self.pool2(x_o)

        x_o = self.flatten(x_o)

        x_o = torch.relu(self.fc1(x_o))

        self.train()

        return self.fc2(x_o)

model = Mymodel()
dummy_input = torch.randn(1, 3, 32, 32)
output = model(dummy_input)

# Create visualization
make_dot(output, params=dict(model.named_parameters())).render("src", format="png")