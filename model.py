from __future__ import annotations

import json
import os
from scipy.spatial.distance import cdist
from typing import Any, Callable, Optional, Union

import numpy as np
import PIL
import supervision as sv
import timm
import torch
import torch.nn as nn
from safetensors.torch import save_file
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.transforms import Compose, ToPILImage
from tqdm.auto import tqdm

from .log import get_logger
from .torch_utils import load_safetensors_checkpoint, parse_device_spec

logger = get_logger(__name__)


def _initialize_reid_model_from_timm(
    cls,
    model_name_or_checkpoint_path: str,
    device: Optional[str] = "auto",
    get_pooled_features: bool = True,
    **kwargs,
):
    if model_name_or_checkpoint_path not in timm.list_models(
        filter=model_name_or_checkpoint_path, pretrained=True
    ):
        probable_model_name_list = timm.list_models(
            f"*{model_name_or_checkpoint_path}*", pretrained=True
        )
        if len(probable_model_name_list) == 0:
            raise ValueError(
                f"Model {model_name_or_checkpoint_path} not found in timm. "
                + "Please check the model name and try again."
            )
        logger.warning(
            f"Model {model_name_or_checkpoint_path} not found in timm. "
            + f"Using {probable_model_name_list[0]} instead."
        )
        model_name_or_checkpoint_path = probable_model_name_list[0]
    if not get_pooled_features:
        kwargs["global_pool"] = ""
    model = timm.create_model(
        model_name_or_checkpoint_path, pretrained=True, num_classes=0, **kwargs
    )
    config = resolve_data_config(model.pretrained_cfg)
    transforms = create_transform(**config)
    model_metadata = {
        "model_name_or_checkpoint_path": model_name_or_checkpoint_path,
        "get_pooled_features": get_pooled_features,
        "kwargs": kwargs,
    }
    return cls(model, device, transforms, model_metadata)


def _initialize_reid_model_from_checkpoint(cls, checkpoint_path: str):
    state_dict, config = load_safetensors_checkpoint(checkpoint_path)
    reid_model_instance = _initialize_reid_model_from_timm(
        cls, **config["model_metadata"]
    )
    if config["projection_dimension"]:
        reid_model_instance._add_projection_layer(
            projection_dimension=config["projection_dimension"]
        )
    for k, v in state_dict.items():
        state_dict[k].to(reid_model_instance.device)
    reid_model_instance.backbone_model.load_state_dict(state_dict)
    return reid_model_instance


class ReIDModel:
    """
    A ReID model that is used to extract features from detection crops for trackers
    that utilize appearance features.

    Args:
        backbone_model (nn.Module): The torch model to use as the backbone.
        device (Optional[str]): The device to run the model on.
        transforms (Optional[Union[Callable, list[Callable]]]): The transforms to
            apply to the input images.
        model_metadata (dict[str, Any]): Metadata about the model architecture.
    """

    def __init__(
        self,
        backbone_model: nn.Module,
        device: Optional[str] = "auto",
        transforms: Optional[Union[Callable, list[Callable]]] = None,
        model_metadata: dict[str, Any] = {},
    ):
        self.backbone_model = backbone_model
        self.device = parse_device_spec(device or "auto")
        self.backbone_model.to(self.device)
        self.backbone_model.eval()
        self.inference_transforms = Compose(
            [ToPILImage(), *transforms]
            if isinstance(transforms, list)
            else [ToPILImage(), transforms]
        )
        self.model_metadata = model_metadata

    @classmethod
    def from_timm(
        cls,
        model_name_or_checkpoint_path: str,
        device: Optional[str] = "auto",
        get_pooled_features: bool = True,
        **kwargs,
    ) -> ReIDModel:
        """
        Create a `ReIDModel` with a [timm](https://huggingface.co/docs/timm)
        model as the backbone.

        Args:
            model_name_or_checkpoint_path (str): Name of the timm model to use or
                path to a safetensors checkpoint. If the exact model name is not
                found, the closest match from `timm.list_models` will be used.
            device (str): Device to run the model on.
            get_pooled_features (bool): Whether to get the pooled features from the
                model or not.
            **kwargs: Additional keyword arguments to pass to
                [`timm.create_model`](https://huggingface.co/docs/timm/en/reference/models#timm.create_model).

        Returns:
            ReIDModel: A new instance of `ReIDModel`.
        """
        if os.path.exists(model_name_or_checkpoint_path):
            return _initialize_reid_model_from_checkpoint(
                cls, model_name_or_checkpoint_path
            )
        else:
            return _initialize_reid_model_from_timm(
                cls,
                model_name_or_checkpoint_path,
                device,
                get_pooled_features,
                **kwargs,
            )

    def extract_features(
        self, detections: sv.Detections, frame: Union[np.ndarray, PIL.Image.Image]
    ) -> np.ndarray:
        """
        Extract features from detection crops in the frame.

        Args:
            detections (sv.Detections): Detections from which to extract features.
            frame (np.ndarray or PIL.Image.Image): The input frame.

        Returns:
            np.ndarray: Extracted features for each detection.
        """
        if len(detections) == 0:
            return np.array([])

        if isinstance(frame, PIL.Image.Image):
            frame = np.array(frame)

        features = []
        with torch.inference_mode():
            for box in detections.xyxy:
                crop = sv.crop_image(image=frame, xyxy=[*box.astype(int)])
                tensor = self.inference_transforms(crop).unsqueeze(0).to(self.device)
                feature = (
                    torch.squeeze(self.backbone_model(tensor)).cpu().numpy().flatten()
                )
                features.append(feature)

        return np.array(features)


def calculate_distance_matrix(
    features1: np.ndarray,
    features2: np.ndarray,
    metric: str = "cosine"
) -> np.ndarray:
    """
    Calculates the pairwise distance matrix between two sets of feature embeddings.

    Args:
        features1 (np.ndarray): A (N, D) array of N feature embeddings,
                                each of dimension D.
        features2 (np.ndarray): A (M, D) array of M feature embeddings,
                                each of dimension D.
        metric (str): The distance metric to use. See `scipy.spatial.distance.cdist`.
                      Defaults to "cosine".

    Returns:
        np.ndarray: A (N, M) distance matrix where element (i, j) is the
                    distance between features1[i] and features2[j].
                    Returns an empty array if either feature set is empty,
                    or if dimensions are mismatched in a way cdist can't handle.
    """
    if features1.ndim == 1:
        features1 = features1.reshape(1, -1)
    if features2.ndim == 1:
        features2 = features2.reshape(1, -1)

    if features1.size == 0 or features2.size == 0:
        # Return an empty array of appropriate shape if one is empty
        # Or a 0x0 if both are empty in a way, though cdist might handle this.
        # Let's ensure N or M is 0 in the output shape.
        return np.zeros((features1.shape[0], features2.shape[0]))

    # Ensure features are 2D for cdist
    if features1.ndim != 2 or features2.ndim != 2:
        # Or raise an error, but returning empty might be safer for some downstream tasks
        # that expect an array.
        logger.error("Feature arrays must be 2-dimensional for cdist.")
        return np.zeros((features1.shape[0], features2.shape[0]))

    try:
        distance_matrix = cdist(features1, features2, metric=metric)
    except ValueError as e:
        logger.error(f"Error calculating distance matrix with cdist: {e}")
        # This can happen if dimensions mismatch, e.g. (N,D1) and (M,D2) where D1!=D2
        # Return an empty matrix or re-raise, depending on desired strictness.
        # For now, returning an empty matrix of the expected output shape.
        return np.zeros((features1.shape[0], features2.shape[0]))

    return distance_matrix
