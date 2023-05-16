#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================

import torch
import torch.nn as nn

from mmdet.slim import DistillModel


class PPYOLOEDistillModel(DistillModel):
    """
    Build PPYOLOE distill model, only used in PPYOLOE
    Args:
    """

    def __init__(self, student_model, teacher_model, distill_loss):
        super(PPYOLOEDistillModel, self).__init__(student_model, teacher_model, distill_loss)

    def forward(self, inputs, alpha=0.125):
        if self.training:
            with torch.no_grad():
                teacher_loss = self.teacher_model(inputs)
            if hasattr(self.teacher_model.yolo_head, "assigned_labels"):
                self.student_model.yolo_head.assigned_labels, self.student_model.yolo_head.assigned_bboxes, self.student_model.yolo_head.assigned_scores, self.student_model.yolo_head.mask_positive = \
                    self.teacher_model.yolo_head.assigned_labels, self.teacher_model.yolo_head.assigned_bboxes, self.teacher_model.yolo_head.assigned_scores, self.teacher_model.yolo_head.mask_positive
                delattr(self.teacher_model.yolo_head, "assigned_labels")
                delattr(self.teacher_model.yolo_head, "assigned_bboxes")
                delattr(self.teacher_model.yolo_head, "assigned_scores")
                delattr(self.teacher_model.yolo_head, "mask_positive")
            student_loss = self.student_model(inputs)

            logits_loss, feat_loss = self.distill_loss(self.teacher_model,
                                                       self.student_model)
            det_total_loss = student_loss['loss']
            total_loss = alpha * (det_total_loss + logits_loss + feat_loss)
            student_loss['loss'] = total_loss
            student_loss['det_loss'] = det_total_loss
            student_loss['logits_loss'] = logits_loss
            student_loss['feat_loss'] = feat_loss
            return student_loss
        else:
            return self.student_model(inputs)
