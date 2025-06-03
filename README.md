# Cross-Camera Person Re-Identification (跨摄像头行人重识别)

### 🧠 项目简介

本项目致力于研究和实现**跨摄像头行人重识别**（Person Re-Identification, Re-ID）技术，旨在解决在不同摄像头视角下对同一目标行人进行准确匹配与识别的问题。该技术广泛应用于智能安防、视频监控、城市安全等领域。

本项目基于YOLO11检测和Swin Transformer ReID模型，实现了高效的人员跟踪和重识别系统。在实际场景中，由于摄像头之间的角度、光照、分辨率、时间差异等因素，使得同一人在不同摄像头下的外观存在较大差异。本项目通过深度学习模型训练、特征提取、相似度匹配等手段，提升跨摄像头行人匹配的准确率。

---

### 🚀 主要功能 / 特点

- ✅ **多模型支持**: 支持Swin Transformer、ResNet、MobileNet等多种ReID模型
- ✅ **实时检测**: 基于YOLO11的高效人员检测
- ✅ **智能跟踪**: 基于外观特征的跨帧人员身份关联
- ✅ **设备自适应**: 自动检测并使用最佳计算设备（CUDA/MPS/CPU）
- ✅ **灵活配置**: 可调节相似性阈值、跟踪参数等
- ✅ **可视化输出**: 实时显示跟踪结果和性能指标
- ✅ **多数据源**: 支持摄像头实时输入和视频文件处理
- ✅ **易于使用**: 提供简化版和完整版测试脚本

---

### 🛠 技术栈

- **深度学习框架**: PyTorch, timm
- **计算机视觉**: OpenCV, YOLO11, Supervision
- **科学计算**: NumPy, SciPy
- **开发语言**: Python 3.8+
- **硬件加速**: CUDA, Apple Metal Performance Shaders

---









### 🤝 贡献者

欢迎任何开发者提交 Issue 和 Pull Request！

---

### 📄 License

MIT License

---

## 🚀 快速开始

### 1. 环境准备

#### 系统要求
- Python 3.8+
- Windows 10/11 或 Linux/macOS
- （推荐）NVIDIA GPU with CUDA support

#### 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd cross-camera-re-id

# 安装依赖
pip install -r requirements.txt

# 或使用国内源加速
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

**主要依赖：**
```
ultralytics>=8.0.0
opencv-python>=4.8.0
supervision>=0.16.0
torch>=2.0.0
timm>=0.9.0
scipy>=1.10.0
numpy>=1.24.0
pillow>=9.0.0
```

### 2. 运行测试

#### 简化版测试（推荐新手）

```bash
# 使用webcam测试
python simple_reid_test.py

# 使用视频文件测试
python simple_reid_test.py --video your_video.mp4

# 限制处理帧数（快速测试）
python simple_reid_test.py --video your_video.mp4 --max-frames 100
```

#### 完整版测试

```bash
# 基础用法
python test_reid_tracking.py --video_path your_video.mp4

# 保存输出视频
python test_reid_tracking.py --video_path your_video.mp4 --output tracked_output.mp4

# 指定设备
python test_reid_tracking.py --video_path your_video.mp4 --device cuda

# 后台运行（不显示窗口）
python test_reid_tracking.py --video_path your_video.mp4 --output result.mp4 --no-display

# 测试模式（只处理前200帧）
python test_reid_tracking.py --video_path your_video.mp4 --max-frames 200
```

---

## 📁 项目结构

```
cross-camera-re-id/
├── reid_model.py              # ReID模型核心实现
├── test_reid_tracking.py      # 完整版跟踪测试脚本
├── simple_reid_test.py        # 简化版测试脚本
├── listmodel.py              # 查看可用模型列表
├── requirements.txt          # 项目依赖
├── README.md                # 项目文档
├── models/                  # 模型文件目录
│   └── yolo11m.pt          # YOLO11模型文件
└── examples/               # 示例视频和输出
```

---

## ⚙️ 核心模块

### ReID模型 (`reid_model.py`)
- 支持多种预训练模型（Swin Transformer, ResNet, MobileNet等）
- 自动设备检测（CUDA/MPS/CPU）
- 特征提取和距离计算
- 模型检查点保存/加载

### 跟踪器 (`PersonTracker`)
- 基于特征相似性的ID分配
- 特征历史管理
- 长时间消失目标清理
- 可调节的相似性阈值

---

## 🎯 使用场景

### 1. 单摄像头人员跟踪
```bash
# 基础跟踪
python simple_reid_test.py --video single_camera.mp4
```

### 2. 跨摄像头人员重识别
```bash
# 处理多个视频，手动比较ID
python test_reid_tracking.py --video_path camera1.mp4 --output camera1_tracked.mp4
python test_reid_tracking.py --video_path camera2.mp4 --output camera2_tracked.mp4
```

### 3. 实时摄像头监控
```bash
# 使用默认摄像头
python simple_reid_test.py
```

---

## 🔧 参数配置

### ReID模型配置

```python
# 推荐模型配置
reid_model = ReIDModel.from_timm(
    model_name_or_checkpoint_path='swin_base_patch4_window12_384.ms_in22k',  # 高精度模型
    device='auto',  # 自动设备选择
)
```

**模型选择建议：**
- `swin_base_patch4_window12_384.ms_in22k` - 高精度，较慢
- `resnet50` - 平衡性能
- `mobilenetv3_large_100` - 快速，移动端友好
- `resnet18` - 轻量级，快速

### 跟踪器配置

```python
tracker = PersonTracker(
    reid_model=reid_model,
    similarity_threshold=0.7,  # 相似性阈值 (0.0-1.0，越小越严格)
    max_disappeared=30,        # 最大消失帧数
    feature_history_size=5     # 特征历史大小
)
```

---

## 📊 性能基准

### 测试环境
- CPU: Intel i7-10700K
- GPU: NVIDIA RTX 3080
- RAM: 32GB
- 视频: 1080p@30fps

### 性能表现
| 模型 | 精度 | 速度(FPS) | 显存占用 |
|------|------|-----------|----------|
| Swin-Base | 高 | 8-12 | 4GB |
| ResNet50 | 中等 | 15-20 | 2GB |
| MobileNetV3 | 中等 | 25-30 | 1GB |
| ResNet18 | 低 | 30-35 | 0.5GB |

---

## 🔧 性能优化

### 1. 模型选择策略
- **高精度场景**: `swin_base_patch4_window12_384.ms_in22k`
- **平衡场景**: `resnet50`
- **实时场景**: `mobilenetv3_large_100`
- **资源受限**: `resnet18`

### 2. 参数调优
```python
# 降低检测频率（每N帧检测一次）
if frame_count % 5 == 0:  # 每5帧检测一次
    # 执行检测和跟踪

# 调整相似性阈值
similarity_threshold = 0.6  # 更严格的匹配
similarity_threshold = 0.8  # 更宽松的匹配
```

---

## 🐛 常见问题

### 1. 模型加载失败
```
错误: ReID模型加载失败
解决: 检查网络连接，模型会自动下载
备选: 使用更轻量的模型如resnet18
```

### 2. CUDA内存不足
```
错误: CUDA out of memory
解决: 使用--device cpu 或选择更小的模型
```

### 3. 检测效果不佳
```
问题: 人员ID频繁变化
解决: 
- 降低similarity_threshold（如0.5）
- 增加feature_history_size
- 使用更高精度的ReID模型
```

### 4. 属性错误
```
错误: 'Detections' object has no attribute 'copy'
解决: 项目已包含兼容性修复，更新到最新版本
```

---

## 📈 输出结果示例

### 终端输出
```
=== ReID人员跟踪测试开始 ===
输入视频: test_video.mp4
设备: cuda
✓ YOLO11模型加载成功
✓ Swin Transformer ReID模型加载成功
视频信息: 1920x1080, 30FPS, 1500帧
已处理 100 帧
处理完成: 1500帧, 平均FPS: 15.2
跟踪到的唯一人员数: 5
=== 测试完成 ===
```

### 视频输出特性
- 🎯 实时边界框标注
- 🏷️ 唯一人员ID标签
- 📊 帧计数和FPS显示
- 🎨 不同ID用不同颜色区分

---

## 🔮 扩展功能

### 查看可用模型
```bash
python listmodel.py
```

### 自定义ReID模型
```python
# 添加新的ReID模型
reid_model = ReIDModel.from_timm(
    model_name_or_checkpoint_path='your_custom_model',
    device='auto'
)
```

### 自定义距离度量
```python
# 修改距离计算方式
distance_matrix = calculate_distance_matrix(
    features1, features2, 
    metric="euclidean"  # 或 "cosine", "manhattan"
)
```

---

## 🤝 贡献者

欢迎任何开发者提交 Issue 和 Pull Request！

### 贡献指南
1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

---

## 📞 技术支持

如遇到问题，请检查：
1. ✅ Python版本 >= 3.8
2. ✅ 所有依赖正确安装
3. ✅ 视频文件格式支持（mp4/avi/mov）
4. ✅ 设备兼容性（CUDA版本等）

---

## 📄 许可证

MIT License - 详见项目根目录LICENSE文件

---

## 🎉 致谢

感谢以下开源项目：
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [supervision](https://github.com/roboflow/supervision)
- [PyTorch](https://pytorch.org/)
