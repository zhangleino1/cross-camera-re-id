# -*- coding: utf-8 -*-
"""
跨摄像头行人重识别 (Cross-Camera Person Re-Identification) 模型
整合了日志记录、PyTorch工具函数和ReID模型的完整实现

包含功能：
- 日志记录工具
- PyTorch设备解析和模型检查点加载
- ReID模型特征提取
- 距离矩阵计算
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import PIL
import supervision as sv
import timm
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from scipy.spatial.distance import cdist
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.transforms import Compose, ToPILImage
from tqdm.auto import tqdm

# 设置模型缓存目录到项目本地
MODELS_CACHE_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODELS_CACHE_DIR, exist_ok=True)
os.environ['TORCH_HOME'] = MODELS_CACHE_DIR


# =============================================================================
# 日志记录模块
# =============================================================================

# 基本日志配置，保持最小化以避免副作用
logging.basicConfig(
    stream=sys.stderr, 
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def get_logger(name: Optional[str]) -> logging.Logger:
    """获取指定名称的日志记录器"""
    return logging.getLogger(name)


# =============================================================================
# PyTorch 工具函数模块
# =============================================================================

def parse_device_spec(device_spec: Union[str, torch.device]) -> torch.device:
    """
    将字符串或torch.device转换为有效的torch.device。
    支持的字符串: 'auto', 'cpu', 'cuda', 'cuda:N' (如 'cuda:0'), 或 'mps'。
    
    Args:
        device_spec (Union[str, torch.device]): 设备规格说明。可以是有效的torch.device对象
            或上述描述的字符串之一。

    Returns:
        torch.device: 对应的torch.device对象。

    Raises:
        ValueError: 如果设备规格无法识别或提供的GPU索引超出可用设备范围。
    """
    if isinstance(device_spec, torch.device):
        return device_spec

    device_str = device_spec.lower()
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif device_str == "cpu":
        return torch.device("cpu")
    elif device_str == "cuda":
        return torch.device("cuda")
    elif device_str == "mps":
        return torch.device("mps")
    else:
        match = re.match(r"^cuda:(\d+)$", device_str)
        if match:
            index = int(match.group(1))
            if index < 0:
                raise ValueError(f"GPU索引必须为非负数，得到 {index}。")
            if index >= torch.cuda.device_count():
                raise ValueError(
                    f"请求 cuda:{index} 但只有 {torch.cuda.device_count()} "
                    + "个GPU可用。"
                )
            return torch.device(f"cuda:{index}")

        raise ValueError(f"无法识别的设备规格: {device_spec}")


def load_safetensors_checkpoint(
    checkpoint_path: str, device: str = "cpu"
) -> Tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """
    加载safetensors检查点到张量字典和元数据字典中。

    Args:
        checkpoint_path (str): safetensors检查点的路径。
        device (str): 加载检查点的设备。

    Returns:
        Tuple[dict[str, torch.Tensor], dict[str, Any]]: 包含state_dict和config的元组。
    """
    state_dict = {}
    with safe_open(checkpoint_path, framework="pt", device=device) as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
        metadata = f.metadata()
        config = json.loads(metadata["config"]) if "config" in metadata else {}
    
    model_metadata = config.pop("model_metadata") if "model_metadata" in config else {}
    if "kwargs" in model_metadata:
        kwargs = model_metadata.pop("kwargs")
        model_metadata = {**kwargs, **model_metadata}
    config["model_metadata"] = model_metadata
    return state_dict, config


# =============================================================================
# ReID 模型模块
# =============================================================================

logger = get_logger(__name__)


def _initialize_reid_model_from_timm(
    cls,
    model_name_or_checkpoint_path: str,
    device: Optional[str] = "auto",
    get_pooled_features: bool = True,
    **kwargs,
):
    """从timm初始化ReID模型的内部函数"""
    # 确保使用本地缓存目录
    os.environ['TORCH_HOME'] = MODELS_CACHE_DIR
    
    if model_name_or_checkpoint_path not in timm.list_models(
        filter=model_name_or_checkpoint_path, pretrained=True
    ):
        probable_model_name_list = timm.list_models(
            f"*{model_name_or_checkpoint_path}*", pretrained=True
        )
        if len(probable_model_name_list) == 0:
            raise ValueError(
                f"模型 {model_name_or_checkpoint_path} 在timm中未找到。"
                + "请检查模型名称并重试。"
            )
        logger.warning(
            f"模型 {model_name_or_checkpoint_path} 在timm中未找到。"
            + f"使用 swin_base_patch4_window12_384.ms_in22k 代替。"
        )
        model_name_or_checkpoint_path = 'swin_base_patch4_window12_384.ms_in22k'
    
    if not get_pooled_features:
        kwargs["global_pool"] = ""
    
    # 设置缓存目录并创建模型
    original_torch_home = os.environ.get('TORCH_HOME')
    os.environ['TORCH_HOME'] = MODELS_CACHE_DIR
    
    try:
        model = timm.create_model(
            model_name_or_checkpoint_path, pretrained=True, num_classes=0, **kwargs
        )
    finally:
        # 恢复原始设置
        if original_torch_home:
            os.environ['TORCH_HOME'] = original_torch_home
        else:
            os.environ.pop('TORCH_HOME', None)
    
    config = resolve_data_config(model.pretrained_cfg)
    transforms = create_transform(**config)
    model_metadata = {
        "model_name_or_checkpoint_path": model_name_or_checkpoint_path,
        "get_pooled_features": get_pooled_features,
        "kwargs": kwargs,
    }
    return cls(model, device, transforms, model_metadata)


def _initialize_reid_model_from_checkpoint(cls, checkpoint_path: str):
    """从检查点初始化ReID模型的内部函数"""
    state_dict, config = load_safetensors_checkpoint(checkpoint_path)
    reid_model_instance = _initialize_reid_model_from_timm(
        cls, **config["model_metadata"]
    )
    if config.get("projection_dimension"):
        reid_model_instance._add_projection_layer(
            projection_dimension=config["projection_dimension"]
        )
    for k, v in state_dict.items():
        state_dict[k] = state_dict[k].to(reid_model_instance.device)
    reid_model_instance.backbone_model.load_state_dict(state_dict)
    return reid_model_instance


class ReIDModel:
    """
    用于从检测裁剪中提取特征的ReID模型，适用于利用外观特征的跟踪器。

    Args:
        backbone_model (nn.Module): 用作骨干网络的torch模型。
        device (Optional[str]): 运行模型的设备。
        transforms (Optional[Union[Callable, list[Callable]]]): 应用于输入图像的变换。
        model_metadata (dict[str, Any]): 关于模型架构的元数据。
    """

    def __init__(
        self,
        backbone_model: nn.Module,
        device: Optional[str] = "auto",
        transforms: Optional[Union[Callable, list[Callable]]] = None,
        model_metadata: dict[str, Any] = None,
    ):
        if model_metadata is None:
            model_metadata = {}
            
        self.backbone_model = backbone_model
        self.device = parse_device_spec(device or "auto")
        self.backbone_model.to(self.device)
        self.backbone_model.eval()
        
        if transforms is None:
            self.inference_transforms = ToPILImage()
        else:
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
    ) -> 'ReIDModel':
        """
        使用timm模型作为骨干网络创建ReIDModel。

        Args:
            model_name_or_checkpoint_path (str): 要使用的timm模型名称或
                safetensors检查点的路径。如果找不到确切的模型名称，
                将使用timm.list_models中最接近的匹配。
            device (str): 运行模型的设备。
            get_pooled_features (bool): 是否从模型获取池化特征。
            **kwargs: 传递给timm.create_model的附加关键字参数。

        Returns:
            ReIDModel: ReIDModel的新实例。
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

    def _add_projection_layer(self, projection_dimension: int):
        """添加投影层到模型"""
        # 获取backbone的输出维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            backbone_output = self.backbone_model(dummy_input)
            input_dim = backbone_output.shape[-1]
        
        # 创建投影层
        projection_layer = nn.Linear(input_dim, projection_dimension)
        
        # 将投影层添加到模型
        self.backbone_model = nn.Sequential(
            self.backbone_model,
            projection_layer
        ).to(self.device)

    def extract_features(
        self, detections: sv.Detections, frame: Union[np.ndarray, PIL.Image.Image]
    ) -> np.ndarray:
        """
        从帧中的检测裁剪提取特征。

        Args:
            detections (sv.Detections): 要提取特征的检测结果。
            frame (np.ndarray or PIL.Image.Image): 输入帧。

        Returns:
            np.ndarray: 每个检测的提取特征。
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
    计算两组特征嵌入之间的成对距离矩阵。

    Args:
        features1 (np.ndarray): 形状为(N, D)的N个特征嵌入数组，每个维度为D。
        features2 (np.ndarray): 形状为(M, D)的M个特征嵌入数组，每个维度为D。
        metric (str): 使用的距离度量。参见scipy.spatial.distance.cdist。
                      默认为"cosine"。

    Returns:
        np.ndarray: 形状为(N, M)的距离矩阵，其中元素(i, j)是
                    features1[i]和features2[j]之间的距离。
                    如果任一特征集为空或维度不匹配导致cdist无法处理，
                    则返回空数组。
    """
    if features1.ndim == 1:
        features1 = features1.reshape(1, -1)
    if features2.ndim == 1:
        features2 = features2.reshape(1, -1)

    if features1.size == 0 or features2.size == 0:
        # 如果其中一个为空，返回适当形状的空数组
        return np.zeros((features1.shape[0], features2.shape[0]))

    # 确保特征为2D以供cdist使用
    if features1.ndim != 2 or features2.ndim != 2:
        logger.error("特征数组必须是2维的才能用于cdist。")
        return np.zeros((features1.shape[0], features2.shape[0]))

    try:
        distance_matrix = cdist(features1, features2, metric=metric)
    except ValueError as e:
        logger.error(f"使用cdist计算距离矩阵时出错: {e}")
        # 当维度不匹配时可能发生这种情况，例如(N,D1)和(M,D2)其中D1!=D2
        # 返回预期输出形状的空矩阵
        return np.zeros((features1.shape[0], features2.shape[0]))

    return distance_matrix


# =============================================================================
# 跟踪器中使用的距离计算函数
# =============================================================================

def _get_appearance_distance_matrix(
    trackers: list,
    detection_features: np.ndarray,
    distance_metric: str = "cosine"
) -> np.ndarray:
    """
    计算跟踪器和检测之间的外观距离矩阵。

    Args:
        trackers: 跟踪器列表，每个跟踪器应有get_feature()方法。
        detection_features (np.ndarray): 从当前检测提取的特征。
        distance_metric (str): 距离度量方法。

    Returns:
        np.ndarray: 外观距离矩阵。
    """
    if len(trackers) == 0 or len(detection_features) == 0:
        return np.zeros((len(trackers), len(detection_features)))

    track_features = np.array([t.get_feature() for t in trackers])
    distance_matrix = cdist(
        track_features, detection_features, metric=distance_metric
    )
    distance_matrix = np.clip(distance_matrix, 0, 1)

    return distance_matrix


def _get_combined_distance_matrix(
    iou_matrix: np.ndarray,
    appearance_dist_matrix: np.ndarray,
    appearance_weight: float = 0.5,
    minimum_iou_threshold: float = 0.1,
    appearance_threshold: float = 0.8
) -> np.ndarray:
    """
    将IOU和外观距离组合成单一距离矩阵。

    Args:
        iou_matrix (np.ndarray): 跟踪器和检测之间的IOU矩阵。
        appearance_dist_matrix (np.ndarray): 外观距离矩阵。
        appearance_weight (float): 外观特征的权重。
        minimum_iou_threshold (float): 最小IOU阈值。
        appearance_threshold (float): 外观距离阈值。

    Returns:
        np.ndarray: 组合距离矩阵。
    """
    iou_distance: np.ndarray = 1 - iou_matrix
    combined_dist = (
        (1 - appearance_weight) * iou_distance 
        + appearance_weight * appearance_dist_matrix
    )

    # 为低于阈值的IOU设置高距离
    mask = iou_matrix < minimum_iou_threshold
    combined_dist[mask] = 1.0

    # 为超过阈值的外观距离设置高距离  
    mask = appearance_dist_matrix > appearance_threshold
    combined_dist[mask] = 1.0

    return combined_dist


