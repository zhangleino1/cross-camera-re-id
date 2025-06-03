#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版ReID跟踪测试脚本
用于快速验证功能，使用默认视频或webcam

使用方法：
python simple_reid_test.py                    # 使用webcam
python simple_reid_test.py --video path.mp4   # 使用视频文件
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import time

# 导入必要的库
try:
    from reid_model import ReIDModel, calculate_distance_matrix, get_logger
    from ultralytics import YOLO
    import supervision as sv
except ImportError as e:
    print(f"导入错误: {e}")
    print("请安装必要的依赖:")
    print("pip install ultralytics opencv-python supervision torch timm scipy numpy pillow")
    exit(1)

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


def simple_test(video_source=0, max_frames=500):
    """
    简单的ReID跟踪测试
    
    Args:
        video_source: 视频源（0=webcam, 或视频文件路径）
        max_frames: 最大处理帧数
    """
    logger.info("=== 简化版ReID跟踪测试 ===")
    
    # 1. 加载模型
    logger.info("正在加载模型...")
    try:
        # 使用轻量级模型加快速度
        yolo_model = YOLO("./models/yolo11m.pt")
        logger.info("✓ YOLO11模型加载成功")
    except Exception as e:
        logger.error(f"✗ YOLO模型加载失败: {e}")
        return
    
    try:
        # 首先尝试轻量级模型
        reid_model = ReIDModel.from_timm(
            model_name_or_checkpoint_path='resnetv2_50.a1h_in1k',
            device='cuda:0'
        )
        logger.info("✓ MobileNetV3 ReID模型加载成功")
    except Exception as e:
        logger.warning(f"MobileNetV3加载失败: {e}")
        try:
            # 备用模型
            reid_model = ReIDModel.from_timm(
                model_name_or_checkpoint_path='resnetv2_50.a1h_in1k',
                device='auto'
            )
            logger.info("✓ ResNet18 ReID模型加载成功")
        except Exception as e2:
            logger.error(f"✗ ReID模型加载失败: {e2}")
            return
    
    # 2. 打开视频源
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error(f"✗ 无法打开视频源: {video_source}")
        return
    
    logger.info(f"✓ 视频源已打开: {video_source}")
    
    # 3. 简单的跟踪器变量
    person_features = {}  # id -> feature
    next_id = 1
    similarity_threshold = 0.7
      # 4. 创建标注器
    try:
        # Try the newer API first
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.8)
    except AttributeError:
        try:
            # Try alternative names
            box_annotator = sv.RoundBoxAnnotator(thickness=2)
            label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.8)
        except AttributeError:
            # Fallback to basic visualization
            box_annotator = None
            label_annotator = None
            logger.warning("无法创建标注器，将使用基本可视化")
    
    frame_count = 0
    start_time = time.time()
    
    logger.info("开始处理视频... 按'q'退出")
    
    try:
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                logger.info("视频结束")
                break
            
            frame_count += 1
            
            # 每10帧处理一次以提高速度
            if frame_count % 10 != 0:
                cv2.imshow("ReID Test", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # YOLO检测
            results = yolo_model(frame, verbose=False)
            
            if len(results) > 0 and results[0].boxes is not None:
                # 转换检测结果
                detections = sv.Detections.from_ultralytics(results[0])
                
                # 过滤人员检测（class_id = 0）
                if detections.class_id is not None:
                    person_mask = detections.class_id == 0
                    person_detections = detections[person_mask]
                else:
                    person_detections = detections
                
                if len(person_detections) > 0:
                    # 提取特征
                    try:
                        current_features = reid_model.extract_features(person_detections, frame)
                        
                        if len(current_features) > 0:
                            # 简单的ID分配
                            assigned_ids = []
                            
                            for i, feature in enumerate(current_features):
                                best_id = None
                                best_similarity = float('inf')
                                
                                # 与已有特征比较
                                for existing_id, existing_feature in person_features.items():
                                    # 计算余弦距离
                                    distance = calculate_distance_matrix(
                                        feature.reshape(1, -1),
                                        existing_feature.reshape(1, -1),
                                        metric="cosine"
                                    )[0, 0]
                                    
                                    if distance < similarity_threshold and distance < best_similarity:
                                        best_similarity = distance
                                        best_id = existing_id
                                
                                if best_id is not None:
                                    # 匹配到已有ID
                                    assigned_ids.append(best_id)
                                    # 更新特征（简单平均）
                                    person_features[best_id] = (person_features[best_id] + feature) / 2
                                else:
                                    # 新的人员
                                    assigned_ids.append(next_id)
                                    person_features[next_id] = feature
                                    next_id += 1
                            
                            # 添加ID到检测结果
                            person_detections.tracker_id = np.array(assigned_ids)
                              # 可视化
                            if box_annotator is not None:
                                annotated_frame = box_annotator.annotate(
                                    scene=frame.copy(),
                                    detections=person_detections
                                )
                                
                                if label_annotator is not None:
                                    labels = [f"Person {track_id}" for track_id in assigned_ids]
                                    annotated_frame = label_annotator.annotate(
                                        scene=annotated_frame,
                                        detections=person_detections,
                                        labels=labels
                                    )
                            else:
                                # 基本可视化 - 手动绘制边界框
                                annotated_frame = frame.copy()
                                for i, (track_id, box) in enumerate(zip(assigned_ids, person_detections.xyxy)):
                                    x1, y1, x2, y2 = map(int, box)
                                    # 绘制边界框
                                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    # 绘制标签
                                    label = f"Person {track_id}"
                                    cv2.putText(annotated_frame, label, (x1, y1-10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        else:
                            annotated_frame = frame.copy()
                    except Exception as e:
                        logger.warning(f"特征提取失败: {e}")
                        annotated_frame = frame.copy()
                else:
                    annotated_frame = frame.copy()
            else:
                annotated_frame = frame.copy()
            
            # 添加信息
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_count}, Persons: {len(person_features)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # 显示FPS
            if frame_count > 1:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            
            # 显示结果
            cv2.imshow("ReID Test", annotated_frame)
            
            # 检查退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("用户请求退出")
                break
            
            # 进度信息
            if frame_count % 100 == 0:
                logger.info(f"已处理 {frame_count} 帧，检测到 {len(person_features)} 个人")
    
    except KeyboardInterrupt:
        logger.info("收到中断信号")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # 统计
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        logger.info(f"测试完成: {frame_count}帧, 平均FPS: {avg_fps:.2f}")
        logger.info(f"检测到唯一人员: {len(person_features)}个")


def main():
    parser = argparse.ArgumentParser(description="简化版ReID跟踪测试")
    parser.add_argument("--video", help="视频文件路径（默认使用webcam）", default='./219797.mov')
    parser.add_argument("--max-frames", type=int, default=500, help="最大处理帧数")
    
    args = parser.parse_args()
    
    # 确定视频源
    if args.video:
        if not Path(args.video).exists():
            logger.error(f"视频文件不存在: {args.video}")
            return
        video_source = args.video
        logger.info(f"使用视频文件: {args.video}")
    else:
        video_source = 0
        logger.info("使用webcam")
    
    simple_test(video_source, args.max_frames)


if __name__ == "__main__":
    main()
