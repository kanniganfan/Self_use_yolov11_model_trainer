# YOLO 训练器

> ⚠️ 注意：本软件为个人自用工具，开源仅供学习参考。软件功能和界面都是根据个人使用习惯定制的，可能不适合所有用户。欢迎fork后根据自己的需求进行修改。

这是一个基于 Ultralytics YOLO 的模型训练图形界面工具，可以帮助你快速标注数据并训练自己的目标检测模型。

## 功能特点

- 支持多种 YOLO 模型选择（从 YOLOv8n 到 YOLOv8x）
- 图形化界面标注工具
  - 左键拖动标注目标
  - 右键拖动移动图片
  - 滚轮缩放图片
- 实时显示训练进度
- 自动生成数据集配置
- 自定义训练参数

## 快速开始

### 方法一：使用打包版本（推荐）

1. 从 [Releases](https://github.com/kanniganfan/Self_use_yolov11_model_trainer/releases) 下载最新版本的 `YOLO训练器.exe`
2. 直接双击运行即可

### 方法二：从源码运行

1. 克隆仓库：
   ```bash
   git clone https://github.com/kanniganfan/Self_use_yolov11_model_trainer.git
   cd Self_use_yolov11_model_trainer
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 下载预训练模型：
   ```bash
   python download_models.py
   ```

4. 运行程序：
   ```bash
   python src/main.py
   ```

## 使用教程

### 数据标注

1. 在左侧输入要检测的目类别（每行一个），例如：
   ```
   人
   猫
   狗
   ```

2. 点击"选择数据集文件夹"按钮，选择包含图片的文件夹
3. 在右侧列表中选择要标注的图片
4. 在下拉框中选择当前要标注的类别
5. 使用鼠标在图片上标注：
   - 左键拖动：绘制标注框
   - 右键拖动：移动图片
   - 滚轮：缩放图片
6. 点击"保存当前标注"保存标注结果

### 模型训练

1. 完成数据标注后，选择要使用的模型：
   - YOLOv8n：速度最快，精度较低
   - YOLOv8s：速度和精度均衡
   - YOLOv8m：中等精度
   - YOLOv8l：较高精度
   - YOLOv8x：最高精度，速度最慢

2. 设置训练参数：
   - 训练轮数：建议 100-300 轮
   - 图像大小：建议 640（可根据目标大小调整）

3. 点击"开始训练"，等待训练完成

### 训练结果

训练完成后，在 `runs/detect/train` 目录下可以找到：
- `best.pt`：训练过程中性能最好的模型
- `last.pt`：最后一次保存的模型
- `results.csv`：训练过程的详细指标数据
- `confusion_matrix.png`：混淆矩阵
- `results.png`：训练过程的指标变化图

## 系统要求

- Windows 10 或更高版本
- 4GB 以上内存
- NVIDIA显卡（推荐）：
  - 如果使用打包版本（.exe），需要安装 [NVIDIA显卡驱动](https://www.nvidia.cn/Download/index.aspx)
  - 如果从源码运行，还需要安装 [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)（建议版本 11.8 或更高）
- 如果没有NVIDIA显卡，也可以使用CPU训练，但速度会较慢

## 注意事项

- 标注时请确保标注框足够准确
- 每个类别建议至少准备 100 张以上的图片
- 训练时间取决于图片数量、模型大小和硬件配置
- 如果使用 CPU 训练，速度会较慢

## 常见问题

1. Q: 为什么训练很慢？
   A: 如果使用 CPU 训练会比较慢，建议使用支持 CUDA 的显卡。

2. Q: 标注的框位置不准怎么办？
   A: 可以使用右键拖动移动图片，滚轮缩放图片来获得更精确的视角。

3. Q: 训练时内存不足怎么办？
   A: 可以尝试使用更小的模型（如 YOLOv8n）或减小图像大小。

## 许可证

MIT License

## 问题反馈

如果你遇到任何问题或有改进建议，欢迎在 [Issues](https://github.com/kanniganfan/Self_use_yolov11_model_trainer/issues) 中提出。 