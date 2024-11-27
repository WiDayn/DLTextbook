import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.layer = nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)
