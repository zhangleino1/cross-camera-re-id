#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨摄像头行人重识别测试脚本
使用YOLO11检测 + ReID模型进行人员跟踪和唯一ID分配

功能：
1. 使用OpenCV读取本地视频文件
2. 使用YOLO11进行人员检测和跟踪
3. 使用Swin Transformer ReID模型提取特征
4. 基于外观相似性分配唯一ID
5. 可视化跟踪结果

依赖安装：
pip install ultralytics opencv-python supervision torch timm scipy numpy pillow
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from collections import defaultdict
import argparse

# 导入自定义ReID模型
from reid_model import ReIDModel, calculate_distance_matrix, get_logger

# 导入YOLO和supervision
from ultralytics import YOLO
import supervision as sv

# 设置日志
logger = get_logger(__name__)


def copy_detections(detections):
    """
    安全地复制Detections对象
    """
    try:
        # 尝试使用内置的copy方法
        return detections.copy()
    except AttributeError:
        # 如果没有copy方法，手动创建新的Detections对象
        return sv.Detections(
            xyxy=detections.xyxy.copy() if detections.xyxy is not None else None,
            confidence=detections.confidence.copy() if detections.confidence is not None else None,
            class_id=detections.class_id.copy() if detections.class_id is not None else None,
            tracker_id=detections.tracker_id.copy() if detections.tracker_id is not None else None
        )


class PersonTracker:
    """基于ReID的人员跟踪器"""
    
    def __init__(
        self, 
        reid_model: ReIDModel,
        similarity_threshold: float = 0.6,
        max_disappeared: int = 30,
        feature_history_size: int = 5
    ):
        """
        初始化跟踪器
        
        Args:
            reid_model: ReID特征提取模型
            similarity_threshold: 相似性阈值，低于此值认为是同一人
            max_disappeared: 跟踪目标消失的最大帧数
            feature_history_size: 保存的历史特征数量
        """
        self.reid_model = reid_model
        self.similarity_threshold = similarity_threshold
        self.max_disappeared = max_disappeared
        self.feature_history_size = feature_history_size
        
        # 跟踪状态
        self.next_id = 1
        self.tracked_persons = {}  # person_id -> PersonInfo
        self.disappeared_counts = defaultdict(int)
        
    def update(self, frame: np.ndarray, detections: sv.Detections) -> sv.Detections:
        """
        更新跟踪器状态
        
        Args:
            frame: 当前帧
            detections: YOLO检测结果
            
        Returns:
            带有跟踪ID的检测结果
        """
        # 过滤出人的检测（class_id = 0）
        if detections.class_id is not None:
            person_mask = detections.class_id == 0
            person_detections = detections[person_mask]
        else:
            person_detections = detections
            
        if len(person_detections) == 0:
            # 更新消失计数
            self._update_disappeared_counts([])
            return detections
            
        # 提取当前检测的特征
        current_features = self.reid_model.extract_features(person_detections, frame)
        
        if len(current_features) == 0:
            logger.warning("无法提取特征")
            return detections
            
        # 分配ID
        assigned_ids = self._assign_ids(current_features)
        
        # 更新跟踪状态
        self._update_tracking_state(person_detections, current_features, assigned_ids)
        
        # 创建带ID的检测结果
        result_detections = copy_detections(person_detections)
        result_detections.tracker_id = np.array(assigned_ids)
        
        return result_detections
    
    def _assign_ids(self, current_features: np.ndarray) -> List[int]:
        """基于特征相似性分配ID"""
        assigned_ids = []
        
        if not self.tracked_persons:
            # 没有已跟踪的人，分配新ID
            for _ in range(len(current_features)):
                assigned_ids.append(self.next_id)
                self.next_id += 1
            return assigned_ids
            
        # 获取已跟踪人员的特征
        tracked_ids = list(self.tracked_persons.keys())
        tracked_features = np.array([
            self.tracked_persons[pid]['avg_feature'] 
            for pid in tracked_ids
        ])
        
        # 计算相似性矩阵
        distance_matrix = calculate_distance_matrix(
            current_features, tracked_features, metric="cosine"
        )
        
        # 使用贪心算法进行匹配
        used_tracked_ids = set()
        
        for i in range(len(current_features)):
            best_match_idx = None
            best_distance = float('inf')
            
            for j, tracked_id in enumerate(tracked_ids):
                if tracked_id in used_tracked_ids:
                    continue
                    
                distance = distance_matrix[i, j]
                if distance < self.similarity_threshold and distance < best_distance:
                    best_distance = distance
                    best_match_idx = j
            
            if best_match_idx is not None:
                # 匹配到已有人员
                matched_id = tracked_ids[best_match_idx]
                assigned_ids.append(matched_id)
                used_tracked_ids.add(matched_id)
                logger.debug(f"检测{i}匹配到ID{matched_id}，距离{best_distance:.3f}")
            else:
                # 新人员
                new_id = self.next_id
                assigned_ids.append(new_id)
                self.next_id += 1
                logger.info(f"检测{i}分配新ID{new_id}")
        
        return assigned_ids
    
    def _update_tracking_state(
        self, 
        detections: sv.Detections, 
        features: np.ndarray, 
        assigned_ids: List[int]
    ):
        """更新跟踪状态"""
        current_ids = set(assigned_ids)
        
        # 更新已匹配的人员
        for i, person_id in enumerate(assigned_ids):
            if person_id not in self.tracked_persons:
                self.tracked_persons[person_id] = {
                    'features': [features[i]],
                    'avg_feature': features[i],
                    'bbox': detections.xyxy[i],
                    'last_seen': time.time()
                }
            else:
                # 更新特征历史
                person_info = self.tracked_persons[person_id]
                person_info['features'].append(features[i])
                
                # 保持特征历史大小
                if len(person_info['features']) > self.feature_history_size:
                    person_info['features'] = person_info['features'][-self.feature_history_size:]
                
                # 更新平均特征
                person_info['avg_feature'] = np.mean(person_info['features'], axis=0)
                person_info['bbox'] = detections.xyxy[i]
                person_info['last_seen'] = time.time()
            
            # 重置消失计数
            if person_id in self.disappeared_counts:
                del self.disappeared_counts[person_id]
        
        # 更新消失计数
        self._update_disappeared_counts(current_ids)
    
    def _update_disappeared_counts(self, current_ids: set):
        """更新消失的人员计数"""
        all_tracked_ids = set(self.tracked_persons.keys())
        disappeared_ids = all_tracked_ids - current_ids
        
        for person_id in disappeared_ids:
            self.disappeared_counts[person_id] += 1
            
            # 移除长时间消失的人员
            if self.disappeared_counts[person_id] > self.max_disappeared:
                logger.info(f"移除长时间消失的人员ID: {person_id}")
                del self.tracked_persons[person_id]
                del self.disappeared_counts[person_id]


def setup_models(device: str = "auto") -> Tuple[YOLO, ReIDModel]:
    """设置YOLO和ReID模型"""
    logger.info("正在加载模型...")
    
    # 加载YOLO11模型
    try:
        yolo_model = YOLO("./models/yolo11m.pt")  # 使用nano版本以提高速度
        logger.info("YOLO11模型加载成功")
    except Exception as e:
        logger.error(f"YOLO11模型加载失败: {e}")
        raise
    
    # 加载ReID模型（Swin Transformer）
    try:
        reid_model = ReIDModel.from_timm(
            model_name_or_checkpoint_path='resnetv2_50.a1h_in1k',
            device=device
        )
        logger.info("Swin Transformer ReID模型加载成功")
    except Exception as e:
        logger.error(f"ReID模型加载失败: {e}")
        # 备用模型
        logger.info("尝试加载备用模型ResNet50...")
        try:
            reid_model = ReIDModel.from_timm(
                model_name_or_checkpoint_path='resnetv2_50.a1h_in1k',
                device=device
            )
            logger.info("swin_base_patch4_window12_384.ms_in22k ReID模型加载成功")
        except Exception as e2:
            logger.error(f"备用模型也加载失败: {e2}")
            raise
    
    return yolo_model, reid_model


def create_annotator():
    """创建可视化标注器"""
    try:
        # Try the newer API first
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.8)
        return box_annotator, label_annotator
    except AttributeError:
        try:
            # Try alternative names
            box_annotator = sv.RoundBoxAnnotator(thickness=2)
            label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.8)
            return box_annotator, label_annotator
        except AttributeError:
            # Return None if no annotators available
            return None, None


def process_video(
    video_path: str,
    output_path: Optional[str] = None,
    device: str = "auto",
    max_frames: Optional[int] = None,
    show_video: bool = True
):
    """
    处理视频文件
    
    Args:
        video_path: 输入视频路径
        output_path: 输出视频路径（可选）
        device: 推理设备
        max_frames: 最大处理帧数（用于测试）
        show_video: 是否显示视频
    """
    # 检查输入文件
    if not Path(video_path).exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 设置模型
    yolo_model, reid_model = setup_models(device)
    
    # 创建跟踪器
    tracker = PersonTracker(
        reid_model=reid_model,
        similarity_threshold=0.7,  # 调整相似性阈值
        max_disappeared=30,
        feature_history_size=5
    )
    
    # 创建标注器
    box_annotator, label_annotator = create_annotator()
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"视频信息: {width}x{height}, {fps}FPS, {total_frames}帧")
    
    # 设置输出视频写入器
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        logger.info(f"输出视频: {output_path}")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 限制处理帧数（用于测试）
            if max_frames and frame_count > max_frames:
                break
            
            # YOLO检测
            results = yolo_model.track(frame, persist=True, verbose=False)
            
            if len(results) > 0 and results[0].boxes is not None:
                # 转换为supervision格式
                detections = sv.Detections.from_ultralytics(results[0])
                
                # ReID跟踪
                tracked_detections = tracker.update(frame, detections)
                
                # 可视化
                annotated_frame = frame.copy()
                
                if len(tracked_detections) > 0:
                    # 绘制边界框
                    annotated_frame = box_annotator.annotate(
                        scene=annotated_frame,
                        detections=tracked_detections
                    )
                    
                    # 添加ID标签
                    if tracked_detections.tracker_id is not None:
                        labels = [
                            f"Person ID:{track_id}" 
                            for track_id in tracked_detections.tracker_id
                        ]
                        annotated_frame = label_annotator.annotate(
                            scene=annotated_frame,
                            detections=tracked_detections,
                            labels=labels
                        )
            else:
                annotated_frame = frame.copy()
            
            # 添加帧信息
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_count}/{total_frames if max_frames is None else min(max_frames, total_frames)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # 显示FPS
            if frame_count > 1:
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time
                cv2.putText(
                    annotated_frame,
                    f"FPS: {current_fps:.1f}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            
            # 保存帧
            if writer:
                writer.write(annotated_frame)
            
            # 显示视频
            if show_video:
                cv2.imshow("Person ReID Tracking", annotated_frame)
                
                # 按'q'退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("用户请求退出")
                    break
            
            # 进度信息
            if frame_count % 100 == 0:
                logger.info(f"已处理 {frame_count} 帧")
    
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在退出...")
    
    finally:
        # 清理资源
        cap.release()
        if writer:
            writer.release()
        if show_video:
            cv2.destroyAllWindows()
        
        # 统计信息
        elapsed_time = time.time() - start_time
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        logger.info(f"处理完成: {frame_count}帧, 平均FPS: {avg_fps:.2f}")
        logger.info(f"跟踪到的唯一人员数: {len(tracker.tracked_persons)}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ReID人员跟踪测试")
    parser.add_argument("--video_path", help="输入视频路径" , default='./219797.mov')
    parser.add_argument("--output", "-o", help="输出视频路径" ,default='./out.mov')
    parser.add_argument("--device", default="cuda", help="推理设备 (auto/cpu/cuda)")
    parser.add_argument("--max-frames", type=int, help="最大处理帧数（测试用）")
    parser.add_argument("--no-display", action="store_true", help="不显示视频窗口")
    
    args = parser.parse_args()
    
    logger.info("=== ReID人员跟踪测试开始 ===")
    logger.info(f"输入视频: {args.video_path}")
    logger.info(f"设备: {args.device}")
    
    try:
        process_video(
            video_path=args.video_path,
            output_path=args.output,
            device=args.device,
            max_frames=args.max_frames,
            show_video=not args.no_display
        )
        logger.info("=== 测试完成 ===")
    except Exception as e:
        logger.error(f"处理失败: {e}")
        raise


if __name__ == "__main__":
    main()
