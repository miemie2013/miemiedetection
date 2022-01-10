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

    def forward(self, x, network, state_dict, im_size=None):
        body_feats = self.backbone.forward(x, network, state_dict)
        # fpn_feats = self.fpn(body_feats)
        # out = self.head(fpn_feats, im_size)
        # return out
        return body_feats



