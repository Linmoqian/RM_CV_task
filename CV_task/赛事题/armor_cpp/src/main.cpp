#include "armor_detector.h"
#include <iostream>

void processVideo(const std::string &videoPath)
{
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        std::cerr << "无法打开视频: " << videoPath << std::endl;
        return;
    }

    armor::DetectorConfig config;
    // 针对红色装甲板，使用红色通道
    config.binaryThreshold = 220;
    config.maxAngleDiff = 15.0;
    config.maxLenDiffRatio = 1.0;
    config.maxLenGapRatio = 0.8;
    config.maxYGapRatio = 1.5;
    config.maxXGapRatio = 2.2;
    config.minXGapRatio = 0.8;
    config.maxDistanceRatio = 3.0;
    config.minDistanceRatio = 0.8;

    armor::ArmorDetector detector(config);

    cv::Mat frame;
    while (cap.read(frame))
    {
        auto armors = detector.detect(frame);
        detector.drawResult(frame, armors);

        cv::putText(frame, "Armors: " + std::to_string(armors.size()),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Armor Detection", frame);
        if (cv::waitKey(30) == 'q')
            break;
    }

    cap.release();
    cv::destroyAllWindows();
}

void processCamera(int cameraId = 0)
{
    cv::VideoCapture cap(cameraId);
    if (!cap.isOpened())
    {
        std::cerr << "无法打开摄像头: " << cameraId << std::endl;
        return;
    }

    armor::DetectorConfig config;
    config.binaryThreshold = 220;

    armor::ArmorDetector detector(config);

    std::cout << "摄像头已启动，按 'q' 退出" << std::endl;

    cv::Mat frame;
    while (cap.read(frame))
    {
        auto armors = detector.detect(frame);
        detector.drawResult(frame, armors);

        cv::putText(frame, "Armors: " + std::to_string(armors.size()),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Armor Detection", frame);
        if (cv::waitKey(1) == 'q')
            break;
    }

    cap.release();
    cv::destroyAllWindows();
}

void processImage(const std::string &imagePath)
{
    cv::Mat frame = cv::imread(imagePath);
    if (frame.empty())
    {
        std::cerr << "无法读取图片: " << imagePath << std::endl;
        return;
    }

    armor::DetectorConfig config;
    config.binaryThreshold = 220;

    armor::ArmorDetector detector(config);
    auto armors = detector.detect(frame);
    detector.drawResult(frame, armors);

    std::cout << "检测到 " << armors.size() << " 个装甲板" << std::endl;
    for (size_t i = 0; i < armors.size(); i++)
    {
        std::cout << "  [" << i + 1 << "] center: (" << armors[i].center.x << ", "
                  << armors[i].center.y << ") confidence: " << armors[i].confidence << std::endl;
    }

    cv::imshow("Armor Detection", frame);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "用法:" << std::endl;
        std::cout << "  " << argv[0] << " --camera [ID]    摄像头模式" << std::endl;
        std::cout << "  " << argv[0] << " --video <path>   视频模式" << std::endl;
        std::cout << "  " << argv[0] << " --image <path>   图片模式" << std::endl;
        return 0;
    }

    std::string mode = argv[1];

    if (mode == "--camera")
    {
        int cameraId = (argc > 2) ? std::stoi(argv[2]) : 0;
        processCamera(cameraId);
    }
    else if (mode == "--video")
    {
        if (argc < 3)
        {
            std::cerr << "请指定视频路径" << std::endl;
            return 1;
        }
        processVideo(argv[2]);
    }
    else if (mode == "--image")
    {
        if (argc < 3)
        {
            std::cerr << "请指定图片路径" << std::endl;
            return 1;
        }
        processImage(argv[2]);
    }
    else
    {
        std::cerr << "未知模式: " << mode << std::endl;
        return 1;
    }

    return 0;
}
