import argparse
import datetime
import time
import torch
import os
from detect import detect
from models import Darknet
from pruner.prunedirector.pytorch.dependency import save_tracker, PruneTracker, load_tracker
from pruner.util.pytorch.module_traverse import get_submodule
from utils import torch_utils
from utils.datasets import LoadImages
from utils.parse_config import parse_data_cfg
from utils.utils import non_max_suppression, load_classes
from yolov3_prune_utils import prune_segment_from_layer

pruned_pt_name = 'current.pt'
pruned_tracker_name = 'tracker.obj'

def initial_model_gen(cfg, img_size,weights):
    device = torch_utils.select_device()
    model = Darknet(cfg, img_size)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
    else:  # darknet format
        assert False, 'initial point needs to be pytorch format'
    model.to(device).eval()
    tracker = PruneTracker()
    model(torch.rand( 1, 3, img_size, img_size).to(device), tracker=tracker)
    return model, tracker

def restore_pruned_checkpt(cfg, img_size, resume_prune_point):
    device = torch_utils.select_device()
    model = Darknet(cfg, img_size)
    pruned_weight = resume_prune_point+os.sep+pruned_pt_name
    pruned_tracker = resume_prune_point+os.sep+pruned_tracker_name
    load_tracker(pruned_tracker,model)
    checkpoint = torch.load(pruned_weight, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    return model, pruned_tracker

def save_prune_checkpt(model, tracker, prune_point, epoch=0, best_loss=float('inf'), optimizer=None):
    # we always save to cpu
    model.to(torch.device('cpu'))
    checkpoint = {'epoch': epoch,
                  'best_loss': best_loss,
                  'model': model.state_dict(),
                  'optimizer': optimizer.state_dict() if optimizer is not None else None}
    full_pruned_weight_name = prune_point+os.sep+pruned_pt_name
    full_pruned_tracker_name = prune_point+os.sep+pruned_tracker_name

    torch.save(checkpoint, full_pruned_weight_name)
    save_tracker(tracker,full_pruned_tracker_name)

def run_single_detect(model,images,img_size,conf_thres=0.3,nms_thres=0.45):
    device = torch_utils.select_device()
    dataloader = LoadImages(images, img_size=img_size)
    classes = load_classes(parse_data_cfg('cfg/coco.data')['names'])
    for i, (path, img, im0) in enumerate(dataloader):
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        pred = model(img)
        pred = pred[pred[:, :, 4] > conf_thres]  # remove boxes < threshold

        if len(pred) > 0:
            # Run NMS on predictions
            detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0]
            # Print results to screen
            unique_classes = detections[:, -1].cpu().unique()
            for c in unique_classes:
                n = (detections[:, -1].cpu() == c).sum()
                print('%g %ss' % (n, classes[int(c)]), end=', ')


def run_pruner(
        cfg,
        weights,
        images,
        img_size=416,
        conf_thres=0.3,
        nms_thres=0.45,
        resume_prune_point=None
):
    # we get the initial model, and tracker by invoking the detector
    if resume_prune_point is None:
        model,tracker = initial_model_gen(cfg, img_size,weights)
    else:
        model,tracker = restore_pruned_checkpt(cfg, img_size, resume_prune_point)
    print("model created")
    # we prune and save dependency
    prune_segment_from_layer(tracker,'module_list:0:conv_0',(1,20),model)
    run_single_detect(model,images,img_size)
    print("tested current pruning section state")

    # save dependency then restore inside detect
    initial_ts = str(time.mktime(datetime.datetime.now().timetuple()))
    os.makedirs('prune_tracker'+os.sep+initial_ts)
    save_prune_checkpt(model,tracker,'prune_tracker'+os.sep+initial_ts)


    # we load the model and then run detect
    restored_model, restored_tracker = restore_pruned_checkpt(cfg,img_size,'prune_tracker'+os.sep+initial_ts)
    run_single_detect(restored_model,images, img_size)

    #save_tracker(tracker,'prune_tracker.obj')
    # we run train (which also test the pruned model)
    '''
    model, tracker = detect(
        cfg,
        weights,
        images,
        img_size=img_size,
        conf_thres=conf_thres,
        nms_thres=nms_thres,
        tracker_file_restore='prune_tracker.obj'
    )
    '''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3.pt', help='path to weights file')
    parser.add_argument('--images', type=str, default='data/samples', help='path to images')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.50, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()
    with torch.no_grad():
        run_pruner(
            opt.cfg,
            opt.weights,
            opt.images,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres
        )

