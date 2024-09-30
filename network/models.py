from __future__ import annotations
import warnings
from collections.abc import Sequence
import torch
import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, export
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextConfig
import random
from utils.monai_inferers_utils import select_points, generate_box
from utils.loss import BCELoss, BinaryDiceLoss
from torch.cuda.amp import autocast

#%% set up model
class lifelong(nn.Module):
    def __init__(self, 
                image_encoder, 
                mask_decoder,
                prompt_encoder,
                clip_ckpt,
                roi_size,
                patch_size,
                test_mode=False,
                ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.text_encoder = TextEncoder(clip_ckpt)
        self.feat_shape = np.array(roi_size)/np.array(patch_size)
        self.test_mode = test_mode
        self.dice_loss = BinaryDiceLoss().cuda()
        self.bce_loss = BCELoss().cuda()
        self.decoder_iter = 6
        self.fusion_layer = nn.Linear(512, 512)

    def forward(self, image, text=None, boxes=None, points=None, **kwargs):
        bs = image.shape[0]
        img_shape = (image.shape[2], image.shape[3], image.shape[4])
        image_embedding, _ = self.image_encoder(image)
        image_embedding = image_embedding.transpose(1, 2).view(bs, -1, 
            int(self.feat_shape[0]), int(self.feat_shape[1]), int(self.feat_shape[2]))
        # test mode
        if self.test_mode:
            return self.forward_decoder(image_embedding, img_shape, text, boxes, points)
        # train mode
        ## sl
        sl_loss = self.supervised_forward(image, image_embedding, img_shape, text, kwargs['train_organs'], kwargs['train_labels'])
        return sl_loss

    def forward_decoder(self, image_embedding, img_shape, text=None, boxes=None, points=None):
        with torch.no_grad():
            if text is not None:
                text_embedding = self.text_encoder(text)  # (B, 768)
            else:
                text_embedding = None

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            masks=None,
            text_embedding=text_embedding,
        )

        dense_pe = self.prompt_encoder.get_dense_pe()
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            text_embedding = text_embedding,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
          )
        logits = F.interpolate(low_res_masks, size=img_shape, mode='trilinear', align_corners=False)
        return logits

    def supervised_forward(self, image, image_embedding, img_shape, training_organs, train_labels, text):
        iter_text = self.build_prompt_label(image.shape[0], training_organs, train_labels)
        sl_loss = 0
        logits = self.forward_decoder(image_embedding, img_shape, text=iter_text)
        # cal loss
        sl_loss_dice = self.dice_loss.forward(logits.squeeze().float(), train_labels.squeeze().float())
        sl_loss_bce = self.bce_loss.forward(logits.squeeze().float(), train_labels.squeeze().float())
        sl_loss += sl_loss_dice + sl_loss_bce
        return sl_loss
    
    def build_prompt_label(self, bs, training_organs, train_labels):
        # generate prompt & label
        iter_organs = []
        iter_text = []

        for sample_idx in range(bs):
            # organ prompt
            iter_organs.append(training_organs)

        iter_text = torch.stack(iter_organs, dim=0).cuda()
        return iter_organs, iter_text
    
class TextEncoder(nn.Module):
    def __init__(self, clip_ckpt):
        super().__init__()
        config = CLIPTextConfig()
        self.clip_text_model = CLIPTextModel(config)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_ckpt)
        self.dim_align = nn.Linear(512, 768)
        # freeze text encoder
        for param in self.clip_text_model.parameters():
            param.requires_grad = False

    def tokenize(self, organ_names):
        text_list = ['A computerized tomography of a {}.'.format(organ_name) for organ_name in organ_names]
        tokens = self.tokenizer(text_list, padding=True, return_tensors="pt")
        for key in tokens.keys():
            tokens[key] = tokens[key].cuda()
        return tokens
    
    def forward(self, text):
        if text is None:
            return None
        if type(text) is str:
            text = [text]
        tokens = self.tokenize(text)
        clip_outputs = self.clip_text_model(**tokens)
        text_embedding = clip_outputs.pooler_output
        text_embedding = self.dim_align(text_embedding)
        return text_embedding

__all__ = ["UNet", "Unet"]
@export("monai.networks.nets")
@alias("Unet")
class UNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        text_embedding_dim: int,
        kernel_size: Sequence[int] | int = 3,
        up_kernel_size: Sequence[int] | int = 3,
        num_res_units: int = 0,
        act: tuple | str = Act.PRELU,
        norm: tuple | str = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ) -> None:
        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if isinstance(kernel_size, Sequence) and len(kernel_size) != spatial_dims:
            raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence) and len(up_kernel_size) != spatial_dims:
            raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        self.text_fc - nn.Linear(text_embedding_dim, in_channels)

        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ) -> nn.Module:
            c = channels[0]
            s = strides[0]
            subblock: nn.Module
            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides[1:], False)  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]
            down = self._get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
            up = self._get_up_layer(upc, outc, s, is_top)  # create layer in upsampling path
            return self._get_connection_block(down, up, subblock)

        self.model = _create_block(in_channels, out_channels, self.channels, self.strides, True)

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        mod: nn.Module
        if self.num_res_units > 0:
            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        conv: Convolution | nn.Sequential
        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)
        return conv

    def forward(self, x: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        text_features = self.text_fc(text_embedding)
        x = torch.cat((x, text_features), dim=1)
        x = self.model(x)
        return x
