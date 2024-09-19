

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
            points=points,
            boxes=boxes,
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
    
    # def unsupervised_forward(self, image, image_embedding, pseudo_seg_cleaned, img_shape):
    #     sll_loss = 0
    #     for iter in range(self.decoder_iter):
    #         if iter % 2 == 0:
    #             pseudo_labels, pseudo_points_prompt = self.build_pseudo_point_prompt_label(image.shape, pseudo_seg_cleaned)
    #             logits = self.forward_decoder(image_embedding, img_shape, text=None, boxes=None, points=pseudo_points_prompt)
    #         else:
    #             pseudo_labels, pseudo_bboxes_prompt = self.build_pseudo_box_prompt_label(image.shape, pseudo_seg_cleaned)
    #             logits = self.forward_decoder(image_embedding, img_shape, text=None, boxes=pseudo_bboxes_prompt, points=None)
    #         # cal loss
    #         sll_loss_dice = self.dice_loss.forward(logits.squeeze().float(), pseudo_labels.squeeze().float())
    #         sll_loss_bce = self.bce_loss.forward(logits.squeeze().float(), pseudo_labels.squeeze().float())
    #         sll_loss += sll_loss_dice + sll_loss_bce
    #     return sll_loss

    def build_prompt_label(self, bs, training_organs, train_labels):
        # generate prompt & label
        iter_organs = []
        iter_bboxes = []
        iter_points_ax = []
        iter_point_labels = []
        iter_text = []

        for sample_idx in range(bs):
            # organ prompt
            iter_organs.append(training_organs)
            # box prompt
            box = generate_box(train_labels[sample_idx])
            iter_bboxes.append(box)
            # point prompt
            num_positive_extra_max, num_negative_extra_max = 10, 10
            num_positive_extra = random.randint(0, num_positive_extra_max)
            num_negative_extra = random.randint(0, num_negative_extra_max)
            point, point_label = select_points(
                train_labels[sample_idx],
                num_positive_extra=num_positive_extra,
                num_negative_extra=num_negative_extra,
                fix_extra_point_num=num_positive_extra_max + num_negative_extra_max)
            iter_points_ax.append(point)
            iter_point_labels.append(point_label)

        # batched prompt
        iter_points_ax = torch.stack(iter_points_ax, dim=0).cuda()
        iter_point_labels = torch.stack(iter_point_labels, dim=0).cuda()
        iter_points = (iter_points_ax, iter_point_labels)
        iter_bboxes = torch.stack(iter_bboxes, dim=0).float().cuda()

        iter_text = torch.stack(iter_organs, dim=0).cuda()
        return iter_points, iter_bboxes, iter_organs, iter_text
    
    # def build_pseudo_point_prompt_label(self, input_shape, seg_labels):
    #     pseudo_labels = torch.zeros(input_shape).cuda()
    #     # generate points
    #     points = []
    #     point_labels = []
    #     for batch_idx in range(input_shape[0]):
    #         # generate pseudo label
    #         unique_ids = torch.unique(seg_labels[batch_idx])
    #         unique_ids = unique_ids[unique_ids != -1]
    #         region_id = random.choice(unique_ids).item()
    #         pseudo_labels[batch_idx][seg_labels[batch_idx]==region_id] = 1
    #         # generate point prompt
    #         num_positive_extra_max, num_negative_extra_max = 10, 10
    #         num_positive_extra = random.randint(4, num_positive_extra_max)
    #         num_negative_extra = random.randint(0, num_negative_extra_max)
    #         assert len(pseudo_labels[batch_idx][0].shape) == 3
    #         point, point_label = select_points(
    #             pseudo_labels[batch_idx][0],
    #             num_positive_extra=num_positive_extra,
    #             num_negative_extra=num_negative_extra,
    #             fix_extra_point_num=num_positive_extra_max + num_negative_extra_max)
    #         points.append(point)
    #         point_labels.append(point_label)
    #     points = torch.stack(points, dim=0).cuda()
    #     point_labels = torch.stack(point_labels, dim=0).cuda()
    #     pseudo_points_prompt = (points, point_labels)
    #     return pseudo_labels, pseudo_points_prompt

    # def build_pseudo_box_prompt_label(self, input_shape, seg_labels_cleaned):
    #     pseudo_labels = torch.zeros(input_shape).cuda()
    #     iter_bboxes = []
    #     # generate boxes
    #     for batch_idx in range(input_shape[0]):
    #         # generate ori pseudo label
    #         unique_ids = torch.unique(seg_labels_cleaned[batch_idx])
    #         unique_ids = unique_ids[unique_ids != -1]
    #         region_id = random.choice(unique_ids).item()
    #         pseudo_labels[batch_idx][seg_labels_cleaned[batch_idx]==region_id] = 1
    #         # generate box prompt
    #         box = generate_box(pseudo_labels[batch_idx][0])
    #         iter_bboxes.append(box)
    #         # refine pseudo label
    #         x_min, y_min, z_min, x_max, y_max, z_max = box
    #         binary_cube = torch.zeros_like(pseudo_labels[batch_idx][0]).int()
    #         binary_cube[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = 1
    #         # cal iou
    #         mask_label = seg_labels_cleaned[batch_idx][0]
    #         assert binary_cube.shape == mask_label.shape, str(binary_cube.shape) + ' ' + str(mask_label.shape)
    #         mask_values_in_binary_cube = mask_label[binary_cube == 1]
    #         unique_mask_values = torch.unique(mask_values_in_binary_cube)
    #         # print('unique_mask_values ', unique_mask_values)
    #         for value in unique_mask_values:
    #             if value == -1: continue
    #             mask_area = (mask_label == value)
    #             intersection = (binary_cube & mask_area)
    #             iou = intersection.float().sum() / mask_area.float().sum()
    #             if iou > 0.90:
    #                 # print(f"Mask value {value} has IOU > 0.90 in binary cube.")
    #                 pseudo_labels[batch_idx][seg_labels_cleaned[batch_idx]==value] = 1

    #     bboxes = torch.stack(iter_bboxes, dim=0).float().cuda()
    #     return pseudo_labels, bboxes
    
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
    """
    Enhanced version of UNet which has residual units implemented with the ResidualUnit class.
    The residual part uses a convolution to change the input dimensions to match the output dimensions
    if this is necessary but will use nn.Identity if not.
    Refer to: https://link.springer.com/chapter/10.1007/978-3-030-12029-0_40.

    Each layer of the network has a encode and decode path with a skip connection between them. Data in the encode path
    is downsampled using strided convolutions (if `strides` is given values greater than 1) and in the decode path
    upsampled using strided transpose convolutions. These down or up sampling operations occur at the beginning of each
    block rather than afterwards as is typical in UNet implementations.

    To further explain this consider the first example network given below. This network has 3 layers with strides
    of 2 for each of the middle layers (the last layer is the bottom connection which does not down/up sample). Input
    data to this network is immediately reduced in the spatial dimensions by a factor of 2 by the first convolution of
    the residual unit defining the first layer of the encode part. The last layer of the decode part will upsample its
    input (data from the previous layer concatenated with data from the skip connection) in the first convolution. this
    ensures the final output of the network has the same shape as the input.

    Padding values for the convolutions are chosen to ensure output sizes are even divisors/multiples of the input
    sizes if the `strides` value for a layer is a factor of the input sizes. A typical case is to use `strides` values
    of 2 and inputs that are multiples of powers of 2. An input can thus be downsampled evenly however many times its
    dimensions can be divided by 2, so for the example network inputs would have to have dimensions that are multiples
    of 4. In the second example network given below the input to the bottom layer will have shape (1, 64, 15, 15) for
    an input of shape (1, 1, 240, 240) demonstrating the input being reduced in size spatially by 2**4.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        channels: sequence of channels. Top block first. The length of `channels` should be no less than 2.
        strides: sequence of convolution strides. The length of `stride` should equal to `len(channels) - 1`.
        kernel_size: convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        up_kernel_size: upsampling convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        num_res_units: number of residual units. Defaults to 0.
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        bias: whether to have a bias term in convolution blocks. Defaults to True.
            According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
            if a conv layer is directly followed by a batch norm layer, bias should be False.
        adn_ordering: a string representing the ordering of activation (A), normalization (N), and dropout (D).
            Defaults to "NDA". See also: :py:class:`monai.networks.blocks.ADN`.

    Examples::

        from monai.networks.nets import UNet

        # 3 layer network with down/upsampling by a factor of 2 at each layer with 2-convolution residual units
        net = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(4, 8, 16),
            strides=(2, 2),
            num_res_units=2
        )

        # 5 layer network with simple convolution/normalization/dropout/activation blocks defining the layers
        net=UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(4, 8, 16, 32, 64),
            strides=(2, 2, 2, 2),
        )

    Note: The acceptable spatial size of input data depends on the parameters of the network,
        to set appropriate spatial size, please check the tutorial for more details:
        https://github.com/Project-MONAI/tutorials/blob/master/modules/UNet_input_size_constrains.ipynb.
        Typically, when using a stride of 2 in down / up sampling, the output dimensions are either half of the
        input when downsampling, or twice when upsampling. In this case with N numbers of layers in the network,
        the inputs must have spatial dimensions that are all multiples of 2^N.
        Usually, applying `resize`, `pad` or `crop` transforms can help adjust the spatial size of input data.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
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

        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ) -> nn.Module:
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.

            Args:
                inc: number of input channels.
                outc: number of output channels.
                channels: sequence of channels. Top block first.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
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
        """
        Returns the block object defining a layer of the UNet structure including the implementation of the skip
        between encoding (down) and decoding (up) sides of the network.

        Args:
            down_path: encoding half of the layer
            up_path: decoding half of the layer
            subblock: block defining the next layer in the network.
        Returns: block for this layer: `nn.Sequential(down_path, SkipConnection(subblock), up_path)`
        """
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the encoding (down) part of a layer of the network. This typically will downsample data at some point
        in its structure. Its output is used as input to the next layer down and is concatenated with output from the
        next layer to form the input for the decode (up) part of the layer.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
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
        """
        Returns the bottom or bottleneck layer at the bottom of the network linking encode to decode halves.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the decoding (up) part of a layer of the network. This typically will upsample data at some point
        in its structure. Its output is used as input to the next layer up.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x
