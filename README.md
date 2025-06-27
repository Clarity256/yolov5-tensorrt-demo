# YOLOv5 TensorRT C++ Demo

一个使用 C++, TensorRT 和 OpenCV 加速 YOLOv5 处理视频的示例项目。

(A C++ demo for accelerating YOLOv5 with TensorRT and OpenCV for video processing.)

## 特性

- 使用 C++ 实现，性能高效。
- 通过 TensorRT 进行深度学习推理加速 (支持FP16)。
- 使用 OpenCV 进行图像处理和视频流读写。
- 跨平台支持 (已在 x86 PC 和 NVIDIA Jetson 上测试)。

## 环境要求

- Ubuntu 18.04 / 20.04 或 Jetson JetPack
- NVIDIA 驱动
- CUDA Toolkit
- cuDNN
- TensorRT
- OpenCV (>= 4.x)
- CMake (>= 3.10)

## 如何使用

### 1. 模型转换

本项目需要一个 `.engine` 文件。请使用 YOLOv5 官方仓库将 `.pt` 文件转换为 `.onnx`，然后使用 TensorRT 的 `trtexec` 工具将其转换为 `.engine` 文件。

```bash
# 步骤1: pt -> onnx
python export.py --weights yolov5s.pt --include onnx

# 步骤2: onnx -> engine
/usr/src/tensorrt/bin/trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s.engine --fp16