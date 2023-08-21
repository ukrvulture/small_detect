#!/usr/bin/python3

# Wrapper of SAM-class providing multi-mask prediction and its config.


import cv2
import segment_anything


class SamAutoMaskGeneratorConfig:
    def __init__(self):
        self.pred_iou_thresh = 0.8
        self.stability_score_thresh = 0.7
        self.box_nms_thresh = 0.7
        self.min_mask_region_area = 30


class SamMultiMaskInference:
    def __init__(self, sam_auto_mask_generator_config,
        sam_checkpoint='sam_vit_b_01ec64.pth', sam_model_type='vit_b', device='cuda'):
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type=sam_model_type
        self.device=device

        self.sam_model = segment_anything.sam_model_registry[sam_model_type](
            checkpoint=sam_checkpoint)
        self.sam_model.to(device=device)

        self.mask_generator = segment_anything.SamAutomaticMaskGenerator(self.sam_model,
            pred_iou_thresh=sam_auto_mask_generator_config.pred_iou_thresh,
            stability_score_thresh=sam_auto_mask_generator_config.stability_score_thresh,
            box_nms_thresh=sam_auto_mask_generator_config.box_nms_thresh,
            min_mask_region_area=sam_auto_mask_generator_config.min_mask_region_area)

    def generate_all_masks(self, img_rgba):
        # noinspection PyUnresolvedReferences
        img_rgb = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)
        masks = self.mask_generator.generate(img_rgb)

        binary_masks = []
        for mask in  sorted(masks, key=(lambda mask: mask['area']), reverse=True):
            binary_masks.append(mask['segmentation'])
        return binary_masks
