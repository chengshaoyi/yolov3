from collections import defaultdict, OrderedDict
import torch.nn as nn
from pruner.prunedirector.pytorch.dependency import save_tracker
from utils.parse_config import *
from utils.utils import *

from yolov3_prune_utils import in_out_record, create_entries_for_conv_group, link_prev_io_entry_to_current_module, \
    link_shortcut_siblings


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        padding=pad,
                                                        bias=not bn))
            if bn:
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))

        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1:
                modules.add_module('_debug_padding_%d' % i, nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module('maxpool_%d' % i, maxpool)

        elif module_def['type'] == 'upsample':
            # upsample = nn.Upsample(scale_factor=int(module_def['stride']), mode='nearest')  # WARNING: deprecated
            upsample = Upsample(scale_factor=int(module_def['stride']), mode='nearest')
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module('shortcut_%d' % i, EmptyLayer())

        elif module_def['type'] == 'yolo':
            anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
            # Extract anchors
            anchors = [float(x) for x in module_def['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def['classes'])
            img_height = int(hyperparams['height'])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_height, anchor_idxs, cfg=hyperparams['cfg'])
            modules.add_module('yolo_%d' % i, yolo_layer)

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class Upsample(torch.nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class YOLOLayer(nn.Module):

    def __init__(self, anchors, nC, img_dim, anchor_idxs, cfg):
        super(YOLOLayer, self).__init__()

        anchors = [(a_w, a_h) for a_w, a_h in anchors]  # (pixels)
        nA = len(anchors)

        self.anchors = anchors
        self.nA = nA  # number of anchors (3)
        self.nC = nC  # number of classes (80)
        self.bbox_attrs = 5 + nC
        self.img_dim = img_dim  # from hyperparams in cfg file, NOT from parser

        if anchor_idxs[0] == (nA * 2):  # 6
            stride = 32
        elif anchor_idxs[0] == nA:  # 3
            stride = 16
        else:
            stride = 8

        if cfg.endswith('yolov3-tiny.cfg'):
            stride *= 2

        # Build anchor grids
        nG = int(self.img_dim / stride)  # number grid points
        self.grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).float()
        self.grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).float()
        self.scaled_anchors = torch.FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, nA, 1, 1))
        self.weights = class_weights()

        self.loss_means = torch.ones(6)
        self.tx, self.ty, self.tw, self.th = [], [], [], []
        self.yolo_layer = anchor_idxs[0] / nA  # 2, 1, 0

    def forward(self, p, targets=None, batch_report=False, var=None):
        FT = torch.cuda.FloatTensor if p.is_cuda else torch.FloatTensor
        bs = p.shape[0]  # batch size
        nG = p.shape[2]  # number of grid points

        if p.is_cuda and not self.grid_x.is_cuda:
            self.grid_x, self.grid_y = self.grid_x.cuda(), self.grid_y.cuda()
            self.anchor_w, self.anchor_h = self.anchor_w.cuda(), self.anchor_h.cuda()
            self.weights, self.loss_means = self.weights.cuda(), self.loss_means.cuda()

        # p.view(12, 255, 13, 13) -- > (12, 3, 13, 13, 80)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        # Training
        if targets is not None:
            MSELoss = nn.MSELoss()
            BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
            CrossEntropyLoss = nn.CrossEntropyLoss()

            # Get outputs
            x = torch.sigmoid(p[..., 0])  # Center x
            y = torch.sigmoid(p[..., 1])  # Center y
            p_conf = p[..., 4]  # Conf
            p_cls = p[..., 5:]  # Class

            # Width and height (yolo method)
            w = p[..., 2]  # Width
            h = p[..., 3]  # Height
            width = torch.exp(w.data) * self.anchor_w
            height = torch.exp(h.data) * self.anchor_h

            # Width and height (power method)
            # w = torch.sigmoid(p[..., 2])  # Width
            # h = torch.sigmoid(p[..., 3])  # Height
            # width = ((w.data * 2) ** 2) * self.anchor_w
            # height = ((h.data * 2) ** 2) * self.anchor_h

            p_boxes = None
            if batch_report:
                # Predicted boxes: add offset and scale with anchors (in grid space, i.e. 0-13)
                gx = x.data + self.grid_x[:, :, :nG, :nG]
                gy = y.data + self.grid_y[:, :, :nG, :nG]
                p_boxes = torch.stack((gx - width / 2,
                                       gy - height / 2,
                                       gx + width / 2,
                                       gy + height / 2), 4)  # x1y1x2y2

            tx, ty, tw, th, mask, tcls, TP, FP, FN, TC = \
                build_targets(p_boxes, p_conf, p_cls, targets, self.scaled_anchors, self.nA, self.nC, nG, batch_report)

            tcls = tcls[mask]
            if x.is_cuda:
                tx, ty, tw, th, mask, tcls = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda(), mask.cuda(), tcls.cuda()

            # Compute losses
            nT = sum([len(x) for x in targets])  # number of targets
            nM = mask.sum().float()  # number of anchors (assigned to targets)
            nB = len(targets)  # batch size
            k = nM / nB
            if nM > 0:
                lx = k * MSELoss(x[mask], tx[mask])
                ly = k * MSELoss(y[mask], ty[mask])
                lw = k * MSELoss(w[mask], tw[mask])
                lh = k * MSELoss(h[mask], th[mask])

                # self.tx.extend(tx[mask].data.numpy())
                # self.ty.extend(ty[mask].data.numpy())
                # self.tw.extend(tw[mask].data.numpy())
                # self.th.extend(th[mask].data.numpy())
                # print([np.mean(self.tx), np.std(self.tx)],[np.mean(self.ty), np.std(self.ty)],[np.mean(self.tw), np.std(self.tw)],[np.mean(self.th), np.std(self.th)])
                # [0.5040668, 0.2885492] [0.51384246, 0.28328574] [-0.4754091, 0.57951087] [-0.25998235, 0.44858757]
                # [0.50184494, 0.2858976] [0.51747805, 0.2896323] [0.12962963, 0.6263085] [-0.2722081, 0.61574113]
                # [0.5032071, 0.28825334] [0.5063132, 0.2808862] [0.21124361, 0.44760725] [0.35445485, 0.6427766]
                # import matplotlib.pyplot as plt
                # plt.hist(self.x)

                # lconf = k * BCEWithLogitsLoss(p_conf[mask], mask[mask].float())

                lcls = (k / 4) * CrossEntropyLoss(p_cls[mask], torch.argmax(tcls, 1))
                # lcls = (k * 10) * BCEWithLogitsLoss(p_cls[mask], tcls.float())
            else:
                lx, ly, lw, lh, lcls, lconf = FT([0]), FT([0]), FT([0]), FT([0]), FT([0]), FT([0])

            # lconf += k * BCEWithLogitsLoss(p_conf[~mask], mask[~mask].float())
            lconf = (k * 64) * BCEWithLogitsLoss(p_conf, mask.float())

            # Sum loss components
            balance_losses_flag = False
            if balance_losses_flag:
                k = 1 / self.loss_means.clone()
                loss = (lx * k[0] + ly * k[1] + lw * k[2] + lh * k[3] + lconf * k[4] + lcls * k[5]) / k.mean()

                self.loss_means = self.loss_means * 0.99 + \
                                  FT([lx.data, ly.data, lw.data, lh.data, lconf.data, lcls.data]) * 0.01
            else:
                loss = lx + ly + lw + lh + lconf + lcls

            # Sum False Positives from unassigned anchors
            FPe = torch.zeros(self.nC)
            if batch_report:
                i = torch.sigmoid(p_conf[~mask]) > 0.5
                if i.sum() > 0:
                    FP_classes = torch.argmax(p_cls[~mask][i], 1)
                    FPe = torch.bincount(FP_classes, minlength=self.nC).float().cpu()  # extra FPs

            return loss, loss.item(), lx.item(), ly.item(), lw.item(), lh.item(), lconf.item(), lcls.item(), \
                   nT, TP, FP, FPe, FN, TC

        else:
            stride = self.img_dim / nG
            p[..., 0] = torch.sigmoid(p[..., 0]) + self.grid_x  # x
            p[..., 1] = torch.sigmoid(p[..., 1]) + self.grid_y  # y
            p[..., 2] = torch.exp(p[..., 2]) * self.anchor_w  # width
            p[..., 3] = torch.exp(p[..., 3]) * self.anchor_h  # height
            p[..., 4] = torch.sigmoid(p[..., 4])  # p_conf
            p[..., :4] *= stride

            # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
            return p.view(bs, self.nA * nG * nG, 5 + self.nC)




class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, cfg_path, img_size=416):
        super(Darknet, self).__init__()

        self.module_defs = parse_model_config(cfg_path)
        self.module_defs[0]['cfg'] = cfg_path
        self.module_defs[0]['height'] = img_size
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        self.loss_names = ['loss', 'x', 'y', 'w', 'h', 'conf', 'cls', 'nT', 'TP', 'FP', 'FPe', 'FN', 'TC']

    def forward(self, x, targets=None, batch_report=False, var=0, tracker=None, tracker_str_stack=''):
        self.losses = defaultdict(float)
        is_training = targets is not None
        layer_outputs = []
        output = []
        # dependency instrumentation
        tracker_str_stack_curtop = tracker_str_stack+'module_list:'
        in_out_module_tracker= OrderedDict()
        tracker_prev_module_name = None
        # end of dependency instrumentation
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            tracker_str_stack_current = tracker_str_stack_curtop+str(i)+':'
            if module_def['type'] in ['convolutional', 'upsample', 'maxpool']:
                # start dependency instrumentation
                if tracker is not None:
                    if module_def['type'] == 'convolutional':
                        # we traverse what is in there and
                        entry_module, exit_module = \
                            create_entries_for_conv_group(module, tracker, tracker_str_stack_current)
                        out_set = {exit_module}
                        out_set.add(entry_module)
                        exit_modules = [out_set]
                        print(exit_modules)
                        top_module = module
                    else:
                        entry_module = None
                        exit_modules = in_out_module_tracker[tracker_prev_module_name].out_module_names
                        top_module = in_out_module_tracker[tracker_prev_module_name].op

                    in_out_module_tracker[tracker_str_stack_current] = in_out_record(
                        in_module_name=entry_module,
                        out_module_names= exit_modules,
                        op=top_module
                    )
                # end dependency instrumentation
                x = module(x)
            elif module_def['type'] == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
                # instrumentation begin
                # we want to add proper dependencies between source and dests
                # look for all the prev layers's name
                if tracker is not None:
                    tracker_in_out_source_names = []
                    for k in layer_i:
                        if k < 0:
                            suffix = i+k
                        else:
                            suffix = k
                        cur_source_tracker_name = tracker_str_stack_curtop+str(suffix)+':'
                        tracker_in_out_source_names.append(cur_source_tracker_name)
                    out_module_names =[]
                    for tracker_in_out_source_name in tracker_in_out_source_names:
                        cur_source_out = in_out_module_tracker[tracker_in_out_source_name]
                        # confirm we dont have route going into another route
                        assert len(cur_source_out.out_module_names) == 1,\
                            "I thought this is only from sequential or shortcut"
                        out_module_names.append(cur_source_out.out_module_names[0])

                    in_out_module_tracker[tracker_str_stack_current] = in_out_record(
                        in_module_name=None,
                        out_module_names= out_module_names,
                        op='route'
                    )
                # instrumentation end
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
                #instrumentation stuff
                if tracker is not None:
                    tracker_in_out_source_names = [tracker_str_stack_curtop+str(i-1)+':']
                    suffix = layer_i if layer_i > 0 else i+layer_i
                    tracker_in_out_source_names.append(tracker_str_stack_curtop + str(suffix) + ':')
                    out_module_names = set()
                    for tracker_in_out_source_name in tracker_in_out_source_names:
                        cur_source_out = in_out_module_tracker[tracker_in_out_source_name]
                        assert len(cur_source_out.out_module_names) == 1, \
                            "I thought this is only from sequential or shortcut"
                        out_module_names = out_module_names.union(cur_source_out.out_module_names[0])
                    in_out_module_tracker[tracker_str_stack_current] = in_out_record(
                        in_module_name=None,
                        out_module_names= [out_module_names],
                        op='shortcut'
                    )

                #end of instrumentation
            elif module_def['type'] == 'yolo':
                # Train phase: get loss
                if is_training:
                    x, *losses = module[0](x, targets, batch_report, var)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module(x)
                # instrumentation begin
                if tracker is not None:
                    in_out_module_tracker[tracker_str_stack_current] = in_out_record(
                        in_module_name=None,
                        out_module_names=None,
                        op=module
                    )
                # instrumentation end
                output.append(x)
            layer_outputs.append(x)

            tracker_prev_module_name = tracker_str_stack_current

        # instrumentation
        # go through all short_cut layer's out_module_name and form cliques...put them into each short_cut layer's out_module_name

        shortcuts = {}
        for module_group_name, module_group_content in in_out_module_tracker.items():
            if module_group_content.op == 'shortcut':
                shortcuts[module_group_name] = module_group_content.out_module_names[0]

        for module_group_name, module_group_content in in_out_module_tracker.items():
            if module_group_content.op == 'shortcut':
                module_group_out_name_set = module_group_content.out_module_names[0]
                for shortcut_name, shortcut_members  in shortcuts.items():
                    if len(module_group_out_name_set & shortcut_members) != 0:
                        module_group_out_name_set.update(shortcut_members)
            elif module_group_content.op == 'route':
                for route_group_name_set in module_group_content.out_module_names:
                    for shortcut_name, shortcut_members in shortcuts.items():
                        if len(route_group_name_set & shortcut_members) != 0:
                            route_group_name_set.update(shortcut_members)
        # go through the layers in in_out_module_tracker
        prev_module_content = None
        for module_group_name, module_group_content in in_out_module_tracker.items():
            entry_module = module_group_content.in_module_name
            if entry_module is not None and prev_module_content is not None:
                # we gonna build the link from the prev layer to this layer
                link_prev_io_entry_to_current_module(tracker, prev_module_content, entry_module, self)
            prev_module_content = module_group_content

        set_of_sibling_sets = []
        for shortcut_name, shortcut_members in shortcuts.items():
            if shortcut_members not in set_of_sibling_sets:
                set_of_sibling_sets.append(shortcut_members)

        for cur_clique in set_of_sibling_sets:
            print(cur_clique)
            print(len(cur_clique))
        for cur_clique in set_of_sibling_sets:
            link_shortcut_siblings(tracker, cur_clique, self)
        # go through all all route and go through all the shortcut's set, if there is intersect, we update its
        save_tracker( tracker, 'yolov3.obj')
        # end of instrumentation

        if is_training:
            if batch_report:
                self.losses['TC'] /= 3  # target category
                metrics = torch.zeros(3, len(self.losses['FPe']))  # TP, FP, FN

                ui = np.unique(self.losses['TC'])[1:]
                for i in ui:
                    j = self.losses['TC'] == float(i)
                    metrics[0, i] = (self.losses['TP'][j] > 0).sum().float()  # TP
                    metrics[1, i] = (self.losses['FP'][j] > 0).sum().float()  # FP
                    metrics[2, i] = (self.losses['FN'][j] == 3).sum().float()  # FN
                metrics[1] += self.losses['FPe']

                self.losses['TP'] = metrics[0].sum()
                self.losses['FP'] = metrics[1].sum()
                self.losses['FN'] = metrics[2].sum()
                self.losses['metrics'] = metrics
            else:
                self.losses['TP'] = 0
                self.losses['FP'] = 0
                self.losses['FN'] = 0

            self.losses['nT'] /= 3
            self.losses['TC'] = 0

        ONNX_export = False
        if ONNX_export:
            # Produce a single-layer *.onnx model (upsample ops not working in PyTorch 1.0 export yet)
            output = output[0].squeeze().transpose(0, 1)  # first layer reshaped to 85 x 507
            output[5:] = torch.nn.functional.softmax(torch.sigmoid(output[5:]) * output[4:5], dim=0)  # SSD-like conf
            return output[5:], output[:4]  # ONNX scores, boxes
        #print("===================================")
        #for out_module_name, out_module_content in in_out_module_tracker.items():
        #    print(out_module_name)

        #    print(out_module_content.out_module_names)
        #    print(out_module_content.op)

        return sum(output) if is_training else torch.cat(output, 1)


def load_weights(self, weights_path, cutoff=-1):
    # Parses and loads the weights stored in 'weights_path'
    # @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)

    if weights_path.endswith('darknet53.conv.74'):
        cutoff = 75

    # Open the weights file
    fp = open(weights_path, 'rb')
    header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

    # Needed to write header when saving weights
    self.header_info = header

    self.seen = header[3]
    weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
    fp.close()

    ptr = 0
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            if module_def['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w


"""
    @:param path    - path of the new weights file
    @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
"""


def save_weights(self, path, cutoff=-1):
    fp = open(path, 'wb')
    self.header_info[3] = self.seen
    self.header_info.tofile(fp)

    # Iterate through layers
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            # If batch norm, load bn first
            if module_def['batch_normalize']:
                bn_layer = module[1]
                bn_layer.bias.data.cpu().numpy().tofile(fp)
                bn_layer.weight.data.cpu().numpy().tofile(fp)
                bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                bn_layer.running_var.data.cpu().numpy().tofile(fp)
            # Load conv bias
            else:
                conv_layer.bias.data.cpu().numpy().tofile(fp)
            # Load conv weights
            conv_layer.weight.data.cpu().numpy().tofile(fp)

    fp.close()
