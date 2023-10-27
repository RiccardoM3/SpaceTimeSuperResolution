import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from Layers import (EncoderLayer, DecoderLayer, InputProj, Downsample, Upsample, PositionalEncoding)

# define the model architecture
class LVSRFITModel(nn.Module):
    def __init__(self):
        super(LVSRFITModel, self).__init__()

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
        self.positional_encoding = PositionalEncoding()

        # Encoder
        self.encoder_features = []
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

    def calc_encoder(self, x):
        B, E, C, H, W = x.size()  # [5 4 3 64 96] B batch size. E num encoder video frames. C num colour channels.

        # Extract FC features
        x = self.feature_extraction(x)

        # Add positional encoding to each feature
        positions = torch.tensor([[0,2,4,6]] * B)
        x = self.positional_encoding(x, E, positions)

        # Obtain encoder features
        self.encoder_features = []
        for i in range(len(self.encoder_layers)):
            x = self.encoder_layers[i](x)
            self.encoder_features.append(x)
            if i != len(self.encoder_layers) - 1:
                x = self.downsample_layers[i](x)

    # Note: the pos is supposed to give an idea about the position that the input frames y can be found in the context x
    def forward(self, y, positions):
        B, _, C, H, W = y.size()  # [5 2 3 64 96] B batch size. . C num colour channels.
        E = 4 # E num encoder frames
        D = 3 # D num output video frames
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
        y = y.permute(0, 2, 1, 3, 4)
        y = F.interpolate(y, (D, H, W), mode='trilinear', align_corners=False) # trilinear interpolation to get the middle image
        y = y.permute(0, 2, 1, 3, 4)
        decoder_queries = []
        for i in range(len(self.decoder_layers)):
            current_H = int(H/2**(i))
            current_W = int(W/2**(i))
            
            query = y.reshape(B * D, C, H, W) # Reshape the input tensor to merge batches with the sequence of images
            query = F.interpolate(query, scale_factor=1/2**(i), mode='area') # downscale the images to match the decoder query input size
            query = query.reshape(B, D, C, current_H, current_W) # Reshape the output tensor back to its original shape
       
            query = self.feature_extraction(query) # get FC features from the query
            
            final_query = torch.zeros(B, D, FC, current_H, current_W, device=y.device)
            # final_query[:, 0, :, :, :] = query[:, 0, :, :, :] - ((query[:, 1, :, :, :] + query[:, 2, :, :, :]) / 2)
            final_query[:, 1, :, :, :] = query[:, 1, :, :, :] - ((query[:, 0, :, :, :] + query[:, 2, :, :, :]) / 2) # take the difference average of the interpolated part
            # final_query[:, 2, :, :, :] = query[:, 2, :, :, :] - ((query[:, 0, :, :, :] + query[:, 1, :, :, :]) / 2)
            
            positions_tensor = torch.zeros(B, D, dtype=torch.int64)
            for i in range(len(positions)):
                positions_tensor[i, 0] = 2*positions[i][0]
                positions_tensor[i, 1] = 2*positions[i][0] + 1
                positions_tensor[i, 2] = 2*positions[i][1]
            final_query = self.positional_encoding(final_query, D, positions_tensor) # add the positional encoding

            decoder_queries.append(final_query) # save the query for later

        # Encoder
        # Pre-calculated in self.encoder_features
        assert len(self.encoder_features) > 0

        # Get decoder output
        y = torch.zeros(B, D, FC, QH, QW, device=y.device)
        for i in range(len(self.decoder_layers)):
            y = self.decoder_layers[i](y + decoder_queries[-i - 1], self.encoder_features[-i - 1])
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
        modules_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))
        model_path = f"{modules_path}/{model_name}_model.pth"

        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path))
            print(f"Loaded model. Params: {self.num_params()}M")

        else:
            print(f"{model_path} doesn't exist.")

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
