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

        # Define options
        E = 4 # num of encoder frames
        C = 3 # num of channels
        FC = 64 # num of feature channels
        D = 3 # num of decoder frames
        
        drop_rate=0.
        attn_drop_rate=0.
        mlp_ratio=2.
        drop_path_rate=0.1
        norm_layer = nn.LayerNorm
        qkv_bias=True
        qk_scale=None
        window_size = (4, 4)

        encoder_layer_settings = [
            {
                "heads": 2,
                "windows": 8
            }, {
                "heads": 4,
                "windows": 8
            }, {
                "heads": 8,
                "windows": 8
            }, {
                "heads": 16,
                "windows": 8
            }
        ]
        decoder_layer_settings = [
            {
                "heads": 16,
                "windows": 8
            }, {
                "heads": 8,
                "windows": 8
            }, {
                "heads": 4,
                "windows": 8
            }, {
                "heads": 2,
                "windows": 8
            }
        ]

        # Stochastic depth
        num_windows_per_encoder_layer = [sum(setting["windows"] for setting in encoder_layer_settings)]
        num_windows_per_decoder_layer = [sum(setting["windows"] for setting in decoder_layer_settings)]
        enc_dpr= [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_windows_per_encoder_layer))] 
        dec_dpr = enc_dpr[::-1]

        # Define Layers

        # Feature Extraction
        self.feature_extraction_1 = nn.Conv2d(C, FC, kernel_size=3, stride=1, padding=1)
        self.feature_extraction_2 = nn.Conv2d(C, FC, kernel_size=3, stride=1, padding=1)

        # Encoder
        self.encoder_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        for i in range(len(encoder_layer_settings)):
            encoder_layer_setting = encoder_layer_settings[i]
            encoder_layer = EncoderLayer(
                    dim=FC, 
                    depth=encoder_layer_setting["windows"], 
                    num_heads=encoder_layer_setting["heads"], 
                    num_frames=E, window_size=window_size, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, 
                    attn_drop=attn_drop_rate,
                    # drop_path=enc_dpr[sum(num_windows_per_encoder_layer[:i]):sum(num_windows_per_encoder_layer[:i + 1])], #TODO
                    norm_layer=norm_layer
            )
            self.encoder_layers.append(encoder_layer)
            if i != len(encoder_layer_settings) - 1:
                downsample = Downsample(FC, FC)
                self.downsample_layers.append(downsample)


        # Decoder
        self.decoder_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        for i in range(len(decoder_layer_settings)):
            decoder_layer_setting = decoder_layer_settings[i]
            decoder_layer = DecoderLayer(
                    dim=FC, 
                    depth=decoder_layer_setting["windows"], 
                    num_heads=decoder_layer_setting["heads"], 
                    num_kv_frames=E, num_out_frames=D, 
                    window_size=window_size, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, 
                    attn_drop=attn_drop_rate,
                    # drop_path=dec_dpr[sum(num_windows_per_decoder_layer[:i]):sum(num_windows_per_decoder_layer[:i + 1])], #TODO
                    norm_layer=norm_layer
            )
            self.decoder_layers.append(decoder_layer)
            if i != len(decoder_layer_settings) - 1:
                upsample = Upsample(FC, FC)
                self.upsample_layers.append(upsample)

        # Final Super Resolution
        self.upconv1 = nn.Conv2d(FC, FC * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(FC, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        # Activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, x, y, pos=(0,1)):
        assert(pos[1] - pos[0] == 1) #ensure they are consecutive numbers

        B, N, C, H, W = x.size()  # [5 4 3 64 96] B batch size. D num input video frames. C num colour channels.
        D = 3
        FC = 64
        OD = 2*N-1 # num output video frames
        OH = H*4 # output H
        OW = W*4 # output W
        QH = H//2**(len(self.encoder_layers)-1) # query H
        QW = W//2**(len(self.encoder_layers)-1) # query W

        # Trilinear interpolation
        x = x.permute(0, 2, 1, 3, 4)
        upsample_x = F.interpolate(x, (OD, OH, OW), mode='trilinear', align_corners=False)
        upsample_x = upsample_x.permute(0, 2, 1, 3, 4)
        x = x.permute(0, 2, 1, 3, 4)

        # Feature Extraction
        x = x.view(B * N, C, H, W) # Reshape the input tensor to merge batches with the sequence of images
        x = self.feature_extraction_1(x)
        x = x.view(B, N, FC, H, W) # Reshape the output tensor back to its original shape

        # Generate Queries for Decoder
        # TODO: posblock?
        # TODO: add pos
        # TODO: 1 = 1 - avg(2,3)
        # TODO: 2 = 2 - avg(1,3)
        # TODO: 3 = 3 - avg(1,2)
        y = y.reshape(B * 2, C, H, W) # Reshape the input tensor to merge batches with the sequence of images
        y = F.interpolate(y, scale_factor=1/2**(len(self.encoder_layers)-1), mode='bilinear', align_corners=False) # downscale the images to match the decoder query input size
        y = y.view(B, 2, C, QH, QW) # Reshape the output tensor back to its original shape
        y = y.permute(0, 2, 1, 3, 4)
        y = F.interpolate(y, (D, QH, QW), mode='trilinear', align_corners=False) # trilinear interpolation to get the middle image
        y = y.permute(0, 2, 1, 3, 4)
        y = y.reshape(B * D, C, QH, QW) # Reshape the input tensor to merge batches with the sequence of images
        y = self.feature_extraction_2(y) # get FC features
        y = y.view(B, D, FC, QH, QW) # Reshape the output tensor back to its original shape
        
        # Obtain encoder features
        encoder_features = []
        for i in range(len(self.encoder_layers)):
            x = self.encoder_layers[i](x)
            encoder_features.append(x)
            if i != len(self.encoder_layers) - 1:
                x = self.downsample_layers[i](x)

        # Get decoder output 
        for i in range(len(self.decoder_layers)):
            y = self.decoder_layers[i](y, encoder_features[-i - 1])
            if i != len(self.decoder_layers) - 1:
                y = self.upsample_layers[i](y)

        # Final Super Resolution
        y = y.view(B*D, FC, H, W)
        y = self.lrelu(self.pixel_shuffle(self.upconv1(y)))
        y = self.lrelu(self.pixel_shuffle(self.upconv2(y)))
        y = self.lrelu(self.HRconv(y))
        y = self.conv_last(y)
        y = y.view(B, D, C, OH, OW)

        return y + upsample_x[:, 2*pos[0]:2*pos[1]+1, :, :, :]
        
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
