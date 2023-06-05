# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

__all__ = ['DINOHead']

class DINOHead(nn.Module):
    __inject__ = ['loss']

    def __init__(self, loss='DINOLoss'):
        super(DINOHead, self).__init__()
        self.loss = loss

    def forward(self, out_transformer, body_feats, inputs=None):
        (dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits, dn_meta) = out_transformer
        if self.training:
            assert inputs is not None
            assert 'gt_bbox' in inputs and 'gt_class' in inputs

            if dn_meta is not None:
                if isinstance(dn_meta, list):
                    raise NotImplementedError
                else:
                    dn_out_bboxes, dec_out_bboxes = torch.split(dec_out_bboxes, dn_meta['dn_num_split'], 2)
                    dn_out_logits, dec_out_logits = torch.split(dec_out_logits, dn_meta['dn_num_split'], 2)
            else:
                dn_out_bboxes, dn_out_logits = None, None

            out_bboxes = torch.cat([enc_topk_bboxes.unsqueeze(0), dec_out_bboxes], dim=0)
            out_logits = torch.cat([enc_topk_logits.unsqueeze(0), dec_out_logits], dim=0)

            return self.loss(
                out_bboxes,
                out_logits,
                inputs['gt_bbox'],
                inputs['gt_class'],
                inputs['pad_gt_mask'],
                dn_out_bboxes=dn_out_bboxes,
                dn_out_logits=dn_out_logits,
                dn_meta=dn_meta)
        else:
            return (dec_out_bboxes[-1], dec_out_logits[-1], None)
