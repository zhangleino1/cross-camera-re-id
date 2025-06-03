# cross-camera-re-id

### 🧠 项目简介

本项目致力于研究和实现**跨摄像头行人重识别**（Person Re-Identification, Re-ID）技术，旨在解决在不同摄像头视角下对同一目标行人进行准确匹配与识别的问题。该技术广泛应用于智能安防、视频监控、城市安全等领域。

在实际场景中，由于摄像头之间的角度、光照、分辨率、时间差异等因素，使得同一人在不同摄像头下的外观存在较大差异。本项目通过图像预处理、特征提取、深度学习模型训练等手段，提升跨摄像头行人匹配的准确率。

---

### 🚀 主要功能 / 特点

- 支持从多个摄像头中提取行人图像
- 基于深度学习的特征提取与相似度匹配（如 ResNet、Transformer、Siamese 网络等）
- 提供多种 Re-ID 模型训练与评估流程
- 支持主流 Re-ID 数据集（Market-1501、DukeMTMC-reID、CUHK03 等）
- 可视化结果展示与性能分析工具

---

### 🛠 技术栈

- Python
- PyTorch / TensorFlow（根据具体实现选择）
- OpenCV
- FastAPI / Flask（可选，用于部署接口服务）
- Docker（可选，便于部署和复现）

---

## 使用

The core component is the `ReIDModel` class. Here's a basic example of how to use it:

```python
import numpy as np
import supervision as sv
import torch # Ensure torch is installed
import PIL.Image # Ensure Pillow is installed
from reid_standalone import ReIDModel # Or from reid_standalone.model import ReIDModel

def main():
    # 1. Instantiate the ReIDModel
    # Replace 'osnet_x0_25' with your desired model from timm or a checkpoint path
    # Ensure you have timm installed: pip install timm
    try:
        reid_model = ReIDModel.from_timm(
            model_name_or_checkpoint_path='resnet18', # Using a common lightweight model for example
            device='auto'
        )
        print("ReID model loaded successfully.")
    except Exception as e:
        print(f"Error loading ReID model: {e}")
        print("Please ensure 'timm' is installed and the model name is correct.")
        print("You might need to install model dependencies like torchvision if not already present.")
        return

    # 2. Prepare a dummy frame (e.g., read from a file or create a blank image)
    # Ensure you have numpy and Pillow installed: pip install numpy Pillow
    try:
        frame_width = 640
        frame_height = 480
        dummy_frame_pil = PIL.Image.new('RGB', (frame_width, frame_height), color = 'gray')
        dummy_frame_np = np.array(dummy_frame_pil)
        print(f"Dummy frame created with shape: {dummy_frame_np.shape}")
    except Exception as e:
        print(f"Error creating dummy frame: {e}")
        return

    # 3. Prepare dummy detections
    # Ensure you have supervision installed: pip install supervision
    # Format: [x_min, y_min, x_max, y_max]
    boxes = np.array([
        [100, 100, 200, 300],
        [250, 150, 350, 350]
    ])
    # Optionally, add confidence or class_id if your workflow uses them,
    # though they are not directly used by ReIDModel.extract_features itself.
    detections = sv.Detections(
        xyxy=boxes,
        # confidence=np.array([0.9, 0.85]), # Example
        # class_id=np.array([0, 0])         # Example
    )
    print(f"Dummy detections created: {detections}")

    # 4. Extract features
    # The frame should be a NumPy array (H, W, C) in RGB order.
    # If your images are BGR (e.g. from OpenCV), convert them to RGB.
    try:
        if len(detections.xyxy) > 0:
            features = reid_model.extract_features(detections, dummy_frame_np)
            print(f"Extracted features shape: {features.shape}")
            print("First feature vector:", features[0][:10]) # Print first 10 elements of the first feature
        else:
            print("No detections to extract features from.")
            features = np.array([])

    except Exception as e:
        print(f"Error during feature extraction: {e}")
        # This might happen if the input to the model is not as expected,
        # or if there's an issue with the model itself or its dependencies.
        # For example, if the crop is empty or too small.
        return

    print("Example finished.")

if __name__ == '__main__':
    # Note: To run this example, you need to ensure that the necessary
    # libraries (torch, timm, supervision, numpy, Pillow, scipy) are installed.
    # You might need to run:
    # pip install torch torchvision timm supervision numpy Pillow scipy
    main()
```

This README provides a good starting point for users of the `reid_standalone` module.

## Comparing Features

Once you have extracted features for detections, you can compare them using the `calculate_distance_matrix` function. This function computes the pairwise distances between two sets of feature embeddings.

```python
from reid_standalone import calculate_distance_matrix
# Assuming 'reid_model' is an instantiated ReIDModel
# And 'dummy_frame_np' is a prepared frame

# Example: Get features for a "query" detection
query_boxes = np.array([[50, 50, 100, 100]]) # A single query detection
query_detections = sv.Detections(xyxy=query_boxes)
query_features = reid_model.extract_features(query_detections, dummy_frame_np)
# query_features will be shape (1, D) where D is feature dimension

# Example: Get features for a "gallery" of detections
gallery_boxes = np.array([
    [150, 150, 200, 200], # Gallery detection 1
    [55, 55, 105, 105],   # Gallery detection 2 (similar to query)
    [300, 300, 350, 350]  # Gallery detection 3 (different)
])
gallery_detections = sv.Detections(xyxy=gallery_boxes)
gallery_features = reid_model.extract_features(gallery_detections, dummy_frame_np)
# gallery_features will be shape (3, D)

if query_features.size > 0 and gallery_features.size > 0:
    # Compare the query features with gallery features using cosine distance
    distance_matrix = calculate_distance_matrix(query_features, gallery_features, metric="cosine")
    print("\nDistance Matrix (query vs gallery):")
    print(distance_matrix)

    # Interpretation:
    # The distance_matrix will have shape (N, M), where N is the number of query features
    # and M is the number of gallery features.
    # For cosine distance, a smaller value (closer to 0) indicates higher similarity.
    # A value of 0 means identical (normalized) features.
    # A value of 1 (or close to 1) means very dissimilar.
    # A value of 2 can occur for opposite vectors if not perfectly normalized.

    # Example: Find the gallery detection most similar to the query
    if distance_matrix.shape[0] == 1: # If only one query image
        most_similar_gallery_idx = np.argmin(distance_matrix[0])
        min_distance = distance_matrix[0, most_similar_gallery_idx]
        print(f"The most similar gallery detection to the query is at index: {most_similar_gallery_idx} with distance: {min_distance:.4f}")

    # You can then set a threshold on the distance to determine if it's a match.
    # For example, if min_distance < 0.5, consider it a match.
    # This threshold is application-dependent and often requires tuning.

else:
    print("Could not extract features for comparison.")

# Note: You'll need scipy installed for calculate_distance_matrix:
# pip install scipy
```


---




### 🤝 贡献者

欢迎任何开发者提交 Issue 和 Pull Request！

---

### 📄 License

MIT License

---
