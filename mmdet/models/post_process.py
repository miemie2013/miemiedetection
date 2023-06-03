import torch
import torch.nn.functional as F
import torch.nn as nn

from mmdet.models import bbox_cxcywh_to_xyxy
from mmdet.models.ops import gather_nd


class DETRPostProcess(object):
    __shared__ = ['num_classes', 'use_focal_loss', 'with_mask']
    __inject__ = []

    def __init__(self,
                 num_classes=80,
                 num_top_queries=100,
                 dual_queries=False,
                 dual_groups=0,
                 use_focal_loss=False,
                 with_mask=False,
                 mask_threshold=0.5,
                 use_avg_mask_score=False,
                 bbox_decode_type='origin'):
        super(DETRPostProcess, self).__init__()
        assert bbox_decode_type in ['origin', 'pad']

        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.dual_queries = dual_queries
        self.dual_groups = dual_groups
        self.use_focal_loss = use_focal_loss
        self.with_mask = with_mask
        self.mask_threshold = mask_threshold
        self.use_avg_mask_score = use_avg_mask_score
        self.bbox_decode_type = bbox_decode_type

    def _mask_postprocess(self, mask_pred, score_pred, index):
        mask_score = F.sigmoid(gather_nd(mask_pred, index))
        mask_pred = (mask_score > self.mask_threshold).to(mask_score.dtype)
        if self.use_avg_mask_score:
            avg_mask_score = (mask_pred * mask_score).sum([-2, -1]) / (
                mask_pred.sum([-2, -1]) + 1e-6)
            score_pred *= avg_mask_score

        return mask_pred[0].to(torch.int32), score_pred

    def __call__(self, head_out, im_shape, scale_factor, pad_shape):
        """
        Decode the bbox and mask.

        Args:
            head_out (tuple): bbox_pred, cls_logit and masks of bbox_head output.
            im_shape (Tensor): The shape of the input image without padding.
            scale_factor (Tensor): The scale factor of the input image.
            pad_shape (Tensor): The shape of the input image with padding.
        Returns:
            bbox_pred (Tensor): The output prediction with shape [N, 6], including
                labels, scores and bboxes. The size of bboxes are corresponding
                to the input image, the bboxes may be used in other branch.
            bbox_num (Tensor): The number of prediction boxes of each batch with
                shape [bs], and is N.
        """
        bboxes, logits, masks = head_out
        if self.dual_queries:
            num_queries = logits.shape[1]
            logits, bboxes = logits[:, :int(num_queries // (self.dual_groups + 1)), :], \
                             bboxes[:, :int(num_queries // (self.dual_groups + 1)), :]

        bbox_pred = bbox_cxcywh_to_xyxy(bboxes)
        # calculate the original shape of the image
        origin_shape = torch.floor(im_shape / scale_factor + 0.5)
        # torch的split和paddle有点不同，torch的第二个参数表示的是每一份的大小，paddle的第二个参数表示的是分成几份。
        img_h, img_w = torch.split(origin_shape, 1, -1)
        if self.bbox_decode_type == 'pad':
            # calculate the shape of the image with padding
            out_shape = pad_shape / im_shape * origin_shape
            out_shape = out_shape.flip(1).tile([1, 2]).unsqueeze(1)
        elif self.bbox_decode_type == 'origin':
            out_shape = origin_shape.flip(1).tile([1, 2]).unsqueeze(1)
        else:
            raise Exception(
                f'Wrong `bbox_decode_type`: {self.bbox_decode_type}.')
        bbox_pred *= out_shape

        scores = torch.sigmoid(logits) if self.use_focal_loss else F.softmax(logits, dim=-1)[:, :, :-1]

        device = bboxes.device
        if not self.use_focal_loss:
            scores, labels = scores.max(-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(
                    scores, self.num_top_queries, dim=-1)
                batch_ind = torch.arange(
                    end=scores.shape[0], device=device).unsqueeze(-1).tile(
                        [1, self.num_top_queries])
                index = torch.stack([batch_ind, index], dim=-1)
                labels = gather_nd(labels, index)
                bbox_pred = gather_nd(bbox_pred, index)
        else:
            scores, index = torch.topk(
                scores.flatten(1), self.num_top_queries, dim=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
            batch_ind = torch.arange(end=scores.shape[0], device=device).unsqueeze(-1).tile(
                [1, self.num_top_queries])
            index = torch.stack([batch_ind, index], dim=-1)
            bbox_pred = gather_nd(bbox_pred, index)

        mask_pred = None
        if self.with_mask:
            assert masks is not None
            masks = F.interpolate(
                masks, scale_factor=4, mode="bilinear", align_corners=False)
            # TODO: Support prediction with bs>1.
            # remove padding for input image
            h, w = im_shape.to(torch.int32)[0]
            masks = masks[..., :h, :w]
            # get pred_mask in the original resolution.
            img_h = img_h[0].to(torch.int32)
            img_w = img_w[0].to(torch.int32)
            masks = F.interpolate(
                masks,
                size=(img_h, img_w),
                mode="bilinear",
                align_corners=False)
            mask_pred, scores = self._mask_postprocess(masks, scores, index)

        bbox_pred = torch.cat(
            [
                labels.unsqueeze(-1).to(torch.float32), scores.unsqueeze(-1),
                bbox_pred
            ],
            dim=-1)
        bbox_num = torch.ones((1, ), dtype=torch.int32) * self.num_top_queries
        bbox_num = bbox_num.tile([bbox_pred.shape[0]])
        bbox_pred = bbox_pred.reshape([-1, 6])
        return bbox_pred, bbox_num, mask_pred


