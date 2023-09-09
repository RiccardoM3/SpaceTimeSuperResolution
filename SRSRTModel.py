import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from Layers import (EncoderLayer, DecoderLayer, InputProj, Downsample, Upsample, PositionalEncoding)

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
        enc_windows = [setting["windows"] for setting in encoder_layer_settings]
        dec_windows = [setting["windows"] for setting in decoder_layer_settings]
        enc_dpr= [x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_windows))] 
        dec_dpr = enc_dpr[::-1]

        # Define Layers

        # Feature Extraction
        self.feature_extraction = InputProj(in_channels=C, embed_dim=FC, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)

        # Positional Encoding
        self.positional_encoding_large = PositionalEncoding()
        self.positional_encoding_small = PositionalEncoding()

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
                    drop_path=enc_dpr[sum(enc_windows[:i]):sum(enc_windows[:i + 1])],
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
                    drop_path=dec_dpr[sum(dec_windows[:i]):sum(dec_windows[:i + 1])],
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


    # Note: the pos is supposed to give an idea about the position that the input frames y can be found in the context x
    def forward(self, x, y, pos, skip_encoder=False):
        B, E, C, H, W = x.size()  # [5 4 3 64 96] B batch size. D num output video frames. C num colour channels.
        D = 3
        FC = 64
        OH = H*4 # output H
        OW = W*4 # output W
        QH = H//2**(len(self.encoder_layers)-1) # query H
        QW = W//2**(len(self.encoder_layers)-1) # query W

        # Trilinear interpolation
        y = y.permute(0, 2, 1, 3, 4)
        y_upsampled = F.interpolate(y, (D, OH, OW), mode='trilinear', align_corners=False)
        y_upsampled = y_upsampled.permute(0, 2, 1, 3, 4)
        y = y.permute(0, 2, 1, 3, 4)

        # Generate Queries for Decoder
        y = y.reshape(B * 2, C, H, W) # Reshape the input tensor to merge batches with the sequence of images
        y = F.interpolate(y, scale_factor=1/2**(len(self.encoder_layers)-1), mode='area') # downscale the images to match the decoder query input size
        y = y.reshape(B, 2, C, QH, QW) # Reshape the output tensor back to its original shape
        y = y.permute(0, 2, 1, 3, 4)
        y = F.interpolate(y, (D, QH, QW), mode='trilinear', align_corners=False) # trilinear interpolation to get the middle image
        y = y.permute(0, 2, 1, 3, 4)

        y = self.feature_extraction(y) # get FC features from the query

        y = self.positional_encoding_small(y, E, [2*pos[0], 2*pos[0]+1, 2*pos[1]]) # add the positional encoding

        q = torch.zeros(B, D, FC, QH, QW, device=y.device)
        # q[:, 0, :, :, :] = y[:, 0, :, :, :] - ((y[:, 1, :, :, :] + y[:, 2, :, :, :]) / 2)
        q[:, 1, :, :, :] = y[:, 1, :, :, :] - ((y[:, 0, :, :, :] + y[:, 2, :, :, :]) / 2) # take the difference average of the interpolated part
        # q[:, 2, :, :, :] = y[:, 2, :, :, :] - ((y[:, 0, :, :, :] + y[:, 1, :, :, :]) / 2)
        y = q

        # Encoder
        if not skip_encoder:
            # Extract FC features
            x = self.feature_extraction(x)

            # Add positional encoding to each feature
            x = self.positional_encoding_large(x, E, [0,2,4,6])

            # Obtain encoder features
            self.encoder_features = []
            for i in range(len(self.encoder_layers)):
                x = self.encoder_layers[i](x)
                self.encoder_features.append(x)
                if i != len(self.encoder_layers) - 1:
                    x = self.downsample_layers[i](x)

        # Get decoder output 
        for i in range(len(self.decoder_layers)):
            y = self.decoder_layers[i](y, self.encoder_features[-i - 1])
            if i != len(self.decoder_layers) - 1:
                y = self.upsample_layers[i](y)

        # Final Super Resolution
        y = y.view(B*D, FC, H, W)
        y = self.lrelu(self.pixel_shuffle(self.upconv1(y)))
        y = self.lrelu(self.pixel_shuffle(self.upconv2(y)))
        y = self.lrelu(self.HRconv(y))
        y = self.conv_last(y)
        y = y.view(B, D, C, OH, OW)

        return y + y_upsampled


    def debug_show_images(self, images):
        fig, axs = plt.subplots(len(images), 1, figsize=(12,8), sharex=True, sharey=True)
        fig.subplots_adjust(wspace=0, hspace=0)
        for i in range(len(images)):
            current_image = images[i].permute(1,2,0).cpu().detach().clone().numpy()
            axs[i].axis('off')
            axs[i].imshow(current_image)
        plt.show()

    def load_model(self, model_name):
        model_path = f"models/{model_name}_model"

        if os.path.exists(f"{model_path}.pth"):
            self.load_state_dict(torch.load(f"{model_path}.pth"))
            print(f"Loaded model. Params: {self.num_params()}M")

    

        else:
            print(f"{model_path}.pth doesn't exist.")

    def save_model(self, model_name):
        if not os.path.exists("models"):
            os.makedirs("models")

        model_path = f"models/{model_name}_model"

        torch.save(self.state_dict(), f"{model_path}.pth")
        print("Saved model")

    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return (1.0*params/(1000*1000))

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        r"""Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself.

        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            (string, Parameter): Tuple containing the name and parameter

        Example::

            >>> for name, param in self.named_parameters():
            >>>    if name in ['bias']:
            >>>        print(param.size())

        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem
