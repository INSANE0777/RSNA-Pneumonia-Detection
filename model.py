"""Model architecture for RSNA Pneumonia Detection.

Uses Faster R-CNN with ResNet50-FPN backbone (transfer learning).
Also supports custom CNN + bounding box regression as an alternative.
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator

import config


def get_faster_rcnn_model(num_classes=2, pretrained=True, trainable_backbone_layers=3):
    """Create Faster R-CNN model with ResNet50-FPN backbone.

    Args:
        num_classes: Number of classes (background + pneumonia = 2).
        pretrained: Whether to use pretrained weights.
        trainable_backbone_layers: Number of trainable backbone layers.

    Returns:
        Faster R-CNN model.
    """
    # Load pretrained Faster R-CNN
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers
    )

    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_model(num_classes=2, pretrained=True, model_type='fasterrcnn'):
    """Get detection model.

    Args:
        num_classes: Number of classes.
        pretrained: Use pretrained weights (transfer learning).
        model_type: 'fasterrcnn' or 'fasterrcnn_custom'.

    Returns:
        Model instance.
    """
    if model_type == 'fasterrcnn':
        return get_faster_rcnn_model(num_classes, pretrained)
    elif model_type == 'fasterrcnn_custom':
        return get_faster_rcnn_custom(num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def get_faster_rcnn_custom(num_classes=2, pretrained=True):
    """Create Faster R-CNN with custom anchor sizes for medical imaging.

    Medical images often have different object scales than COCO.
    Loads backbone weights from pretrained model but initializes RPN head
    randomly since anchor sizes differ.
    """
    # Custom anchor generator tuned for chest X-ray lesions
    anchor_generator = AnchorGenerator(
        sizes=((16, 32, 64, 128, 256),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # Build model without pretrained weights (custom anchors mismatch RPN head)
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        rpn_anchor_generator=anchor_generator,
        box_detections_per_img=10,
        box_score_thresh=0.05,
        box_nms_thresh=0.3,
    )

    # Load pretrained backbone weights only
    if pretrained:
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        pretrained_state = weights.get_state_dict(progress=True)

        # Filter out RPN head keys (shape mismatch with custom anchors)
        model_state = model.state_dict()
        filtered_state = {}
        for k, v in pretrained_state.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered_state[k] = v

        model.load_state_dict(filtered_state, strict=False)
        loaded = len(filtered_state)
        total = len(model_state)
        print(f"  Loaded {loaded}/{total} pretrained weights (RPN head initialized randomly)")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def freeze_backbone(model, layers_to_freeze=2):
    """Freeze early backbone layers for transfer learning.

    Args:
        model: Faster R-CNN model.
        layers_to_freeze: Number of ResNet layers to freeze.
    """
    # Freeze backbone body layers
    children = list(model.backbone.body.children())
    for child in children[:layers_to_freeze]:
        for param in child.parameters():
            param.requires_grad = False

    print(f"Frozen first {layers_to_freeze} backbone layers")


def unfreeze_all(model):
    """Unfreeze all model parameters for fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    print("Unfrozen all layers")
