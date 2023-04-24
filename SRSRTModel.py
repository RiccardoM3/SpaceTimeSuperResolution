import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# define the model architecture
class SRSRTModel(nn.Module):
    def __init__(self):
        super(SRSRTModel, self).__init__()

        # CUDA
        if not torch.cuda.is_available():
            print("Please use a device which supports CUDA")
            sys.exit(0)
        
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        # define the layers
        # self.__input_size = 1000
        # self.__hidden_size = 800
        self.hidden = nn.Linear(64, 256)

    def forward(self, x):
        B, D, C, H, W = x.size()  # D input video frames
        
        print(B, D, C, H, W)

        # x = x.permute(0, 2, 1, 3, 4)
        # upsample_x = F.interpolate(x, (2*D-1, H*4, W*4), mode='trilinear', align_corners=False)
        # x = x.permute(0, 2, 1, 3, 4)

        x = F.relu(self.hidden(x))

        # merge x (trained residual) with trillinear interpolation (upsample_x)
        # x = upsample_x + x

        return x
        
    def load_model(self, path):
        if os.path.exists(f"{path}.pth"):
            self.load_state_dict(torch.load(f"{path}.pth"))
        else:
            print(f"{path}.pth doesn't exist.")

    def save_model(self, path):
        if not os.path.exists("models"):
            os.makedirs("models")

        torch.save(self.state_dict(), f"{path}.pth")
    
    def evaluate(self, evaluation_path):
        # create a sample input
        x = torch.randn(1, self.__input_size)

        # pass the input through the model and print the output
        output = self(x)
        print(output)
