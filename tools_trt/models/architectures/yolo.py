#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================


class PPYOLO(object):
    def __init__(self, backbone, fpn, head):
        super(PPYOLO, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.head = head

    def __call__(self, x, network, state_dict, im_size=None):
        backbone_dic = {}
        fpn_dic = {}
        head_dic = {}
        others = {}
        for key, value in state_dict.items():
            if 'tracked' in key:
                continue
            if 'backbone' in key:
                backbone_dic[key] = value
            elif 'fpn' in key:
                fpn_dic[key] = value
            elif 'head' in key:
                head_dic[key] = value
            else:
                others[key] = value

        body_feats = self.backbone(x, network, backbone_dic)
        fpn_feats = self.fpn(body_feats, network, fpn_dic)
        out = self.head(fpn_feats, im_size, network, head_dic)
        # return out
        return fpn_feats[1]



