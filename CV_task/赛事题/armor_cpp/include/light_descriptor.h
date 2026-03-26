#ifndef ARMOR_DETECTOR_H
#define ARMOR_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

namespace armor
{

/**
 * @brief 灯条描述符
 */
struct LightDescriptor
{
    float width;
    float length;
    float angle;
    float area;
    cv::Point2f center;

    LightDescriptor() = default;

    explicit LightDescriptor(const cv::RotatedRect &light)
    {
        width = light.size.width;
        length = light.size.height;
        center = light.center;
        angle = light.angle;
        area = light.size.area();
    }
};

/**
 * @brief 装甲板描述符
 */
struct ArmorDescriptor
{
    LightDescriptor leftLight;
    LightDescriptor rightLight;
    cv::RotatedRect rect;
    cv::Point2f center;
    float confidence;
};

/**
 * @brief 检测器配置
 */
struct DetectorConfig
{
    // 灯条筛选参数
    double minLightArea = 5.0;
    double maxLightRatio = 4.0; // width/height

    // 匹配参数
    double maxAngleDiff = 15.0;
    double maxLenDiffRatio = 1.0;
    double maxLenGapRatio = 0.8;
    double maxYGapRatio = 1.5;
    double maxXGapRatio = 2.2;
    double minXGapRatio = 0.8;
    double maxDistanceRatio = 3.0;
    double minDistanceRatio = 0.8;

    // 图像处理参数
    int binaryThreshold = 220;
    int gaussianKernelSize = 5;
    int dilateKernelSize = 5;
};

/**
 * @brief 装甲板检测器
 */
class ArmorDetector
{
public:
    explicit ArmorDetector(const DetectorConfig &config = DetectorConfig());

    /**
     * @brief 检测装甲板
     * @param frame 输入图像
     * @return 检测到的装甲板列表
     */
    std::vector<ArmorDescriptor> detect(const cv::Mat &frame);

    /**
     * @brief 绘制检测结果
     * @param frame 输入图像
     * @param armors 装甲板列表
     * @param color 绘制颜色
     */
    void drawResult(cv::Mat &frame, const std::vector<ArmorDescriptor> &armors,
                    const cv::Scalar &color = cv::Scalar(0, 0, 255));

private:
    /**
     * @brief 图像预处理
     */
    cv::Mat preprocess(const cv::Mat &frame);

    /**
     * @brief 筛选灯条
     */
    std::vector<LightDescriptor> filterLights(const std::vector<std::vector<cv::Point>> &contours);

    /**
     * @brief 匹配灯条对
     */
    std::vector<ArmorDescriptor> matchLights(const std::vector<LightDescriptor> &lights);

    DetectorConfig config_;
};

} // namespace armor

#endif // ARMOR_DETECTOR_H
