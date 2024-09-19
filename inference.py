import argparse
import os
import torch
import torch.nn.functional as F
import json
from segment_anything_volumetric import sam_model_registry
from network.models import SegVol
from data_process.demo_data_process import process_ct_gt
import monai.transforms as transforms
from utils.monai_inferers_utils import sliding_window_inference, generate_box, select_points, build_binary_cube, build_binary_points, logits2roi_coor
from utils.visualize import draw_result

def set_parse():
    # %% set up parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_mode", default=True, type=bool)
    parser.add_argument("--resume", type = str, default = '')
    parser.add_argument("-infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
    parser.add_argument("-spatial_size", default=(32, 256, 256), type=tuple)
    parser.add_argument("-patch_size", default=(4, 16, 16), type=tuple)
    parser.add_argument('-work_dir', type=str, default='./work_dir')
    ### demo
    parser.add_argument('--demo_config', type=str, required=True)
    parser.add_argument("--clip_ckpt", type = str, default = './config/clip')
    parser.add_argument("--organ_index", type=int, default=0, help="Index of the organ to segment")
    args = parser.parse_args()
    return args

def dice_score(preds, labels):  # on GPU
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match\n" + str(preds.shape) + str(labels.shape)
    predict = preds.view(1, -1)
    target = labels.view(1, -1)
    if target.shape[1] < 1e8:
        predict = predict.cuda()
        target = target.cuda()
    predict = torch.sigmoid(predict)
    predict = torch.where(predict > 0.5, 1., 0.)
    
    tp = torch.sum(torch.mul(predict, target))
    den = torch.sum(predict) + torch.sum(target) + 1
    dice = 2 * tp / den

    if target.shape[1] < 1e8:
        predict = predict.cpu()
        target = target.cpu()
    return dice

def zoom_in_only(args, unet_model, image, gt3D, categories=None):
    logits_labels_record = {}
    image_single = image[0,0]
    ori_shape = image_single.shape

    item_idx = args.organ_index  # 只分割指定的器官
    label_single = gt3D[0][item_idx]

    # skip meaningless categories
    if torch.sum(label_single) == 0:
        print('No object, skip')
        return None

    # generate prompts
    text_single = categories[item_idx] if args.use_text_prompt else None
    print(f'inference |{categories[item_idx]}| target...')

    # zoom-in inference:
    min_d, min_h, min_w, max_d, max_h, max_w = logits2roi_coor(args.spatial_size, label_single.float())
    if min_d is None:
        print('Fail to detect foreground!')
        return None

    # Crop the image
    image_single_cropped = image_single[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1].unsqueeze(0).unsqueeze(0)
    global_preds = label_single[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1]

    # Prompt reflection
    prompt_reflection = (
        global_preds.unsqueeze(0).unsqueeze(0),
        global_preds.unsqueeze(0).unsqueeze(0)
    )

    # inference
    with torch.no_grad():
        logits_single_cropped = sliding_window_inference(
                image_single_cropped.cuda(), prompt_reflection,
                args.spatial_size, 1, unet_model, args.infer_overlap,
                text=text_single
            )
        logits_single_cropped = logits_single_cropped.cpu().squeeze()

    logits_global_single = torch.zeros_like(label_single)
    logits_global_single[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1] = logits_single_cropped
    zoom_in_dice = dice_score(logits_global_single.squeeze(), label_single.squeeze())
    
    logits_labels_record[categories[item_idx]] = (
        zoom_in_dice,
        image_single, 
        None,
        None,
        logits_global_single, 
        label_single
    )
    print(f'zoom-in dice {zoom_in_dice:.4f}')
    
    return logits_labels_record

def inference_single_ct(args, unet_model, data_item, categories):
    unet_model.eval()
    image, gt3D = data_item["image"].float(), data_item["label"]

    logits_labels_record = zoom_in_only(
        args, unet_model, 
        image.unsqueeze(0), 
        gt3D.unsqueeze(0),
        categories=categories)
    
    # visualize
    if args.visualize and logits_labels_record is not None:
        for target, values in logits_labels_record.items():
            dice_score, image, _, _, logits, labels = values
            print(f'{target} result with Dice score {dice_score:.4f} visualizing')
            draw_result(target + f"-Dice {dice_score:.4f}", image, None, None, logits, labels, args.spatial_size, args.work_dir)

def main(args):
    gpu = 0
    torch.cuda.set_device(gpu)
    # build model
    sam_model = sam_model_registry['vit'](args=args)
    unet_model = SegVol(
                        image_encoder=sam_model.image_encoder, 
                        mask_decoder=sam_model.mask_decoder,
                        prompt_encoder=sam_model.prompt_encoder,
                        clip_ckpt=args.clip_ckpt,
                        roi_size=args.spatial_size,
                        patch_size=args.patch_size,
                        test_mode=args.test_mode,
                        ).cuda()
    unet_model = torch.nn.DataParallel(unet_model, device_ids=[gpu])

    # load param
    if os.path.isfile(args.resume):
        ## Map model to be loaded to specified single GPU
        loc = 'cuda:{}'.format(gpu)
        checkpoint = torch.load(args.resume, map_location=loc)
        unet_model.load_state_dict(checkpoint['model'], strict=True)
        print("loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    # load demo config
    with open(args.demo_config, 'r') as file:
        config_dict = json.load(file)
    ct_path, gt_path, categories = config_dict['demo_case']['ct_path'], config_dict['demo_case']['gt_path'], config_dict['categories']

    # preprocess for data
    data_item = process_ct_gt(ct_path, gt_path, categories, args.spatial_size)

    # seg config for prompt & zoom-in
    args.use_zoom_in = True
    args.use_text_prompt = True
    args.visualize = True

    inference_single_ct(args, unet_model, data_item, categories)

if __name__ == "__main__":
    args = set_parse()
    main(args)
