import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from Layers import (make_layer, ResidualBlock_noBN, EncoderLayer, DecoderLayer, 
                     InputProj, Downsample, Upsample)

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
        self.upsample1 = Upsample(3, 3)
        self.upsample2 = Upsample(3, 3)

    def forward(self, x):
        B, D, C, H, W = x.size()  # D input video frames
        
        # print(B, D, C, H, W)

        # x = x.permute(0, 2, 1, 3, 4)
        # upsample_x = F.interpolate(x, (2*D-1, H*4, W*4), mode='trilinear', align_corners=False)
        # x = x.permute(0, 2, 1, 3, 4)

        # x = F.relu(self.hidden(x))

        # merge x (trained residual) with trillinear interpolation (upsample_x)
        # x = upsample_x + x


        # upsample twice
        x = self.upsample1(x)
        x = self.upsample2(x)

        # copy some frames over
        duplicated_frames = x[:, :3]
        x = torch.cat((x, duplicated_frames), dim=1)

        return x
        
    def load_model(self, model_name):
        model_path = f"models/{model_name}_model"

        if os.path.exists(f"{model_path}.pth"):
            self.load_state_dict(torch.load(f"{model_path}.pth"))
            print("Loaded model")
        else:
            print(f"{model_path}.pth doesn't exist.")

    def save_model(self, model_name):
        if not os.path.exists("models"):
            os.makedirs("models")

        model_path = f"models/{model_name}_model"

        torch.save(self.state_dict(), f"{model_path}.pth")
        print("Saved model")
    
    def evaluate(self, evaluation_path):
        # create a sample input
        x = torch.randn(1, self.__input_size)

        # pass the input through the model and print the output
        output = self(x)
        print(output)
