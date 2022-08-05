import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torch.nn.functional as F

from models.net import MobileNetV1 as MobileNetV1
from models.net import FPN, Decode, SSH, conv_bn
from models.residual import HgResBlock
from models.net import conv_bn1X1
from models.semodule_MA import CBAMLayer, GCBAMLayer


def _sigmoid(x):
    x = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return x


class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        # pdb.set_trace()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 10)


def _make_fc(inplanes, outplanes):
    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, 1),
        nn.BatchNorm2d(outplanes),
        nn.ReLU(True))


class RetinaFace(nn.Module):
    def __init__(self, cfg=None, phase='train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace, self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']

        self.decode = Decode(in_channels_list, out_channels)
        self.res = HgResBlock(out_channels, out_channels)
        self.fc = _make_fc(out_channels, out_channels)
        self.mkscore = nn.Conv2d(out_channels, 1, 1)

        self.mkatt0 = nn.Conv2d(out_channels, 1, 1)
        self.mkatt1 = nn.Conv2d(out_channels, 1, 1)
        self.mkatt2 = nn.Conv2d(out_channels, 1, 1)

        self.fpn = FPN(in_channels_list, out_channels)

        self.downsample = nn.MaxPool2d(2, 2)
        self.downsample1 = nn.MaxPool2d(4, 4)


        self.merge0 = conv_bn(out_channels, out_channels)
        self.merge1 = conv_bn(out_channels, out_channels)
        self.merge2 = conv_bn(out_channels, out_channels)

        self.layer1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1)
        self.layer2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1)
        self.layer3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1)

        self.cse = GCBAMLayer(out_channels * 3, reduction=3)
        self.ca0 = CBAMLayer(out_channels)
        self.ca1 = CBAMLayer(out_channels)
        self.ca2 = CBAMLayer(out_channels)

        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def forward(self, inputs):
        InterLayer = self.body(inputs)

        # FPN
        fpn = self.fpn(InterLayer)

        # decoder
        out = self.decode(InterLayer)
        out = self.res(out)
        out = self.fc(out)
        map_out = self.mkscore(out)

        # Feature Enhancement Module
        map_0 = _sigmoid(map_out)
        map_1 = self.downsample(map_0)
        map_2 = self.downsample(map_1)
#        pdb.set_trace()
        layers = list(InterLayer.values())
        layer1 = self.layer1(layers[0])
        layer2 = self.layer2(layers[1])
        layer3 = self.layer3(layers[2])

        gw = self.cse([layer1, layer2, layer3])
        cw0 = self.ca0(fpn[0])
        cw1 = self.ca1(fpn[1])
        cw2 = self.ca2(fpn[2])

        ca0 = gw[0] * cw0 * fpn[0] + fpn[0]
        ca1 = gw[1] * cw1 * fpn[1] + fpn[1]
        ca2 = gw[2] * cw2 * fpn[2] + fpn[2]

        enhance0 = ca0 * (map_0 + 1) + fpn[0]
        enhance1 = ca1 * (map_1 + 1) + fpn[1]
        enhance2 = ca2 * (map_2 + 1) + fpn[2]

        enhance0 = self.merge0(enhance0)
        enhance1 = self.merge1(enhance1)
        enhance2 = self.merge2(enhance2)

        # SSH
        feature1 = self.ssh1(enhance0)
        feature2 = self.ssh2(enhance1)
        feature3 = self.ssh3(enhance2)
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return map_0, output
