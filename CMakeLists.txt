# CMake 最低版本要求
cmake_minimum_required(VERSION 3.10)

# 项目名称
project(YoloTensorRT)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 寻找依赖包 ---
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# --- 寻找 TensorRT (适配 TensorRT 8.x+) ---
# 1. 寻找头文件目录，现在以 NvOnnxParser.h 为标志
find_path(TENSORRT_INCLUDE_DIR NvOnnxParser.h
    HINTS /usr/include/x86_64-linux-gnu)

# 2. 寻找 nvinfer 核心库
find_library(TENSORRT_INFER_LIBRARY nvinfer
    HINTS /usr/lib/x86_64-linux-gnu)

# 3. 寻找 nvonnxparser 库 (替换了旧的 nvparsers)
find_library(TENSORRT_ONNX_PARSER_LIBRARY nvonnxparser
    HINTS /usr/lib/x86_64-linux-gnu)

# --- 检查 TensorRT 是否找到 ---
if (NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_INFER_LIBRARY OR NOT TENSORRT_ONNX_PARSER_LIBRARY)
    message(FATAL_ERROR "无法找到 TensorRT 的头文件或库文件。请确保 libnvinfer-dev 和 libnvonnxparsers-dev 已安装。")
endif()
message(STATUS "成功找到 TensorRT (新版API)!")
message(STATUS "  - 头文件路径: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "  - nvinfer 库: ${TENSORRT_INFER_LIBRARY}")
message(STATUS "  - nvonnxparser 库: ${TENSORRT_ONNX_PARSER_LIBRARY}")


# --- 创建可执行文件 ---
add_executable(yolo_trt_demo main.cpp)

# --- 为目标 yolo_trt_demo 添加依赖 ---

# 1. 添加头文件搜索路径
target_include_directories(yolo_trt_demo PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)

# 2. 链接库文件
target_link_libraries(yolo_trt_demo PRIVATE
    ${OpenCV_LIBS}
    ${TENSORRT_INFER_LIBRARY}
    ${TENSORRT_ONNX_PARSER_LIBRARY} # <-- 链接正确的 onnx parser 库
    ${CUDA_LIBRARIES}
    cudart
)