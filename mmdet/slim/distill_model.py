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
from loguru import logger

class DistillModel(nn.Module):
    def __init__(self, student_model, teacher_model, distill_loss):
        super(DistillModel, self).__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad_(False)
        self.distill_loss = distill_loss

    def parameters(self):
        return self.student_model.parameters()

    def forward(self, inputs):
        if self.training:
            student_loss = self.student_model(inputs)
            with torch.no_grad():
                teacher_loss = self.teacher_model(inputs)

            loss = self.distill_loss(self.teacher_model, self.student_model)
            student_loss['distill_loss'] = loss
            student_loss['teacher_loss'] = teacher_loss['loss']
            student_loss['loss'] += student_loss['distill_loss']
            return student_loss
        else:
            return self.student_model(inputs)

