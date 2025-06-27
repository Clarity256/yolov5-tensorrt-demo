#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <memory>

// OpenCV 库
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp> // NMS需要

// CUDA 运行时 API
#include <cuda_runtime_api.h>

// TensorRT 头文件
#include "NvInfer.h"
//#include "NvParsers.h" // 如果需要从ONNX解析则需要
#include "NvOnnxParser.h" // 用于解析ONNX模型 (适用于 TensorRT 8.x+)

// --- 全局配置 ---

// 输入图像尺寸
const int INPUT_W = 640;
const int INPUT_H = 640;

// 推理的置信度阈值和NMS阈值
const float CONF_THRESHOLD = 0.4;
const float NMS_THRESHOLD = 0.45;

// COCO数据集的80个类别名称
const std::vector<std::string> CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",

    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};


// TensorRT 日志记录器
// TensorRT的很多操作是异步的，通过Logger来报告错误、警告和信息
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // 只打印错误和严重警告
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

// --- 预处理函数 ---
// 将OpenCV的Mat图像预处理为符合模型输入的格式
// 包含 letterbox 缩放、颜色通道转换 (BGR->RGB)、归一化 (0-255 -> 0-1)、格式转换 (HWC -> CHW)
void preprocess(cv::Mat& img, float* blob) {
    // 1. Letterbox 缩放
    // 计算缩放比例，保持宽高比
    int w = img.cols;
    int h = img.rows;
    float r = std::min(static_cast<float>(INPUT_W) / w, static_cast<float>(INPUT_H) / h);
    int new_unpad_w = static_cast<int>(w * r);
    int new_unpad_h = static_cast<int>(h * r);

    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(new_unpad_w, new_unpad_h));

    // 创建一个用灰色填充的画布
    cv::Mat padded_img(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    // 将缩放后的图像复制到画布中央
    resized_img.copyTo(padded_img(cv::Rect(0, 0, new_unpad_w, new_unpad_h)));

    // 2. BGR -> RGB, 归一化, HWC -> CHW
    // HWC (Height, Width, Channels)
    // CHW (Channels, Height, Width)
    int i = 0;
    for (int row = 0; row < INPUT_H; ++row) {
        for (int col = 0; col < INPUT_W; ++col) {
            // 指向像素(row, col)的指针
            cv::Vec3b pixel = padded_img.at<cv::Vec3b>(row, col);
            // 写入R通道数据
            blob[i] = static_cast<float>(pixel[2]) / 255.0f;
            // 写入G通道数据
            blob[i + INPUT_W * INPUT_H] = static_cast<float>(pixel[1]) / 255.0f;
            // 写入B通道数据
            blob[i + 2 * INPUT_W * INPUT_H] = static_cast<float>(pixel[0]) / 255.0f;
            i++;
        }
    }
}


// --- 后处理函数 ---
void postprocess(float* output, cv::Mat& img) {
    // YOLOv5的输出格式是 [batch, num_detections, 5 + num_classes]
    // 在我们的例子中，batch=1, num_detections=25200, num_classes=80
    // 所以每个检测的格式是 [x_center, y_center, width, height, confidence, class_score_0, class_score_1, ...]

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    // 计算缩放和平移因子，用于将检测框坐标从640x640映射回原始图像尺寸
    float scale = std::min(static_cast<float>(INPUT_W) / img.cols, static_cast<float>(INPUT_H) / img.rows);
    int pad_x = 0; // 如果需要左右填充，则计算pad
    int pad_y = 0; // 如果需要上下填充，则计算pad (在本实现中我们只做了左上角对齐)

    int num_detections = 25200; // (80*80 + 40*40 + 20*20) * 3 anchors
    int num_elements_per_detection = 85; // 4 (box) + 1 (conf) + 80 (classes)

    // 遍历所有检测结果
    for (int i = 0; i < num_detections; ++i) {
        float* detection = output + i * num_elements_per_detection;
        float confidence = detection[4];

        // 过滤掉置信度低的检测
        if (confidence > CONF_THRESHOLD) {
            // 找到分数最高的类别
            float* class_scores = detection + 5;
            int class_id = std::max_element(class_scores, class_scores + 80) - class_scores;
            float class_confidence = class_scores[class_id];

            // 最终分数 = 物体置信度 * 类别置信度
            if (class_confidence > 0.0) { // 可以设置一个类别置信度阈值
                float cx = detection[0];
                float cy = detection[1];
                float w = detection[2];
                float h = detection[3];

                // 将中心点+宽高格式转换为左上角+右下角格式
                int left = static_cast<int>((cx - w / 2 - pad_x) / scale);
                int top = static_cast<int>((cy - h / 2 - pad_y) / scale);
                int width = static_cast<int>(w / scale);
                int height = static_cast<int>(h / scale);

                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back(confidence);
                class_ids.push_back(class_id);
            }
        }
    }

    // 执行非极大值抑制 (NMS)
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, indices);

    // 绘制最终的检测框
    for (int idx : indices) {
        cv::Rect box = boxes[idx];
        int class_id = class_ids[idx];
        std::string label = CLASS_NAMES[class_id] + " " + cv::format("%.2f", confidences[idx]);
        
        // 绘制矩形框
        cv::rectangle(img, box, cv::Scalar(0, 255, 0), 2);
        
        // 绘制标签
        int baseLine;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::rectangle(img, cv::Point(box.x, box.y - label_size.height), 
                      cv::Point(box.x + label_size.width, box.y), cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(img, label, cv::Point(box.x, box.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}


int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "用法: " << argv[0] << " <engine_file_path> <video_file_path>" << std::endl;
        return -1;
    }
    std::string engine_path = argv[1];
    std::string video_path = argv[2];

    // --- 1. 初始化 ---
    Logger logger;

    // 读取编译好的TensorRT引擎文件
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "错误: 无法打开引擎文件 " << engine_path << std::endl;
        return -1;
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* engine_data = new char[size];
    file.read(engine_data, size);
    file.close();

    // --- 2. 创建TensorRT运行时和引擎 ---
    // IRuntime是反序列化引擎的入口
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    // ICudaEngine是推理的核心，代表了一个优化的模型
    std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(engine_data, size));
    delete[] engine_data;
    // IExecutionContext包含了特定的批处理大小和权重的状态，用于执行推理
    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());

    if (!context) {
        std::cerr << "创建执行上下文失败" << std::endl;
        return -1;
    }

    // --- 3. 分配GPU/CPU内存 ---
    void* buffers[2]; // 0: input, 1: output
    
    // 获取输入和输出的名称和索引
    const int input_index = engine->getBindingIndex("images"); // "images" 是ONNX模型中输入节点的名称
    const int output_index = engine->getBindingIndex("output0"); // "output0" 是ONNX模型中输出节点的名称
    if (input_index < 0 || output_index < 0) {
        std::cerr << "未能从引擎中找到输入或输出节点" << std::endl;
        return -1;
    }

    // 获取输入和输出的维度，并计算大小
    auto input_dims = engine->getBindingDimensions(input_index);
    size_t input_size = std::accumulate(input_dims.d + 1, input_dims.d + input_dims.nbDims, 1, std::multiplies<size_t>()) * sizeof(float);
    
    auto output_dims = engine->getBindingDimensions(output_index);
    size_t output_size = std::accumulate(output_dims.d + 1, output_dims.d + output_dims.nbDims, 1, std::multiplies<size_t>()) * sizeof(float);

    // 在GPU上分配内存
    cudaMalloc(&buffers[input_index], input_size);
    cudaMalloc(&buffers[output_index], output_size);

    // 在CPU上为输入(blob)和输出分配内存
    float* blob = new float[input_size / sizeof(float)];
    float* output_buffer = new float[output_size / sizeof(float)];
    
    // 创建CUDA流，用于异步执行
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // --- 4. 视频处理循环 ---
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "错误: 无法打开视频文件 " << video_path << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) {
            break;
        }

        // 记录开始时间
        auto start = std::chrono::high_resolution_clock::now();

        // (1) 图像预处理
        preprocess(frame, blob);

        // (2) 将预处理后的数据从CPU拷贝到GPU
        cudaMemcpyAsync(buffers[input_index], blob, input_size, cudaMemcpyHostToDevice, stream);

        // (3) 执行TensorRT推理
        // 使用 enqueueV2, 它支持动态尺寸，并且是推荐的API
        context->enqueueV2(buffers, stream, nullptr);

        // (4) 将推理结果从GPU拷贝回CPU
        cudaMemcpyAsync(output_buffer, buffers[output_index], output_size, cudaMemcpyDeviceToHost, stream);

        // (5) 等待CUDA流中的所有操作完成
        cudaStreamSynchronize(stream);

        // (6) 结果后处理（NMS）和绘制
        postprocess(output_buffer, frame);

        // 记录结束时间并计算FPS
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        double fps = 1.0 / diff.count();
        
        // 在图像上显示FPS
        cv::putText(frame, "FPS: " + std::to_string(static_cast<int>(fps)), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

        // (7) 显示结果
        cv::imshow("YOLOv5 TensorRT Demo", frame);
        if (cv::waitKey(1) == 27) { // 按 'ESC' 键退出
            break;
        }
    }

    // --- 5. 清理资源 ---
    std::cout << "正在清理资源..." << std::endl;
    cudaStreamDestroy(stream);
    cudaFree(buffers[input_index]);
    cudaFree(buffers[output_index]);
    delete[] blob;
    delete[] output_buffer;

    // context, engine, runtime 的智能指针会自动释放
    cap.release();
    cv::destroyAllWindows();
    std::cout << "完成." << std::endl;

    return 0;
}