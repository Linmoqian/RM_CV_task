#include "armor_detector.h"
#include <cmath>

namespace armor
{

ArmorDetector::ArmorDetector(const DetectorConfig &config)
    : config_(config)
{
}

std::vector<ArmorDescriptor> ArmorDetector::detect(const cv::Mat &frame)
{
    // 预处理
    cv::Mat processed = preprocess(frame);

    // 轮廓检测
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(processed, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    // 筛选灯条
    std::vector<LightDescriptor> lights = filterLights(contours);

    // 匹配灯条对
    return matchLights(lights);
}

cv::Mat ArmorDetector::preprocess(const cv::Mat &frame)
{
    std::vector<cv::Mat> channels;
    cv::split(frame, channels);

    // 使用蓝色通道（针对蓝色装甲板）
    cv::Mat binary;
    cv::threshold(channels[0], binary, config_.binaryThreshold, 255, cv::THRESH_BINARY);

    // 高斯滤波
    cv::Mat gaussian;
    cv::GaussianBlur(binary, gaussian,
                      cv::Size(config_.gaussianKernelSize, config_.gaussianKernelSize), 0);

    // 膨胀
    cv::Mat dilated;
    cv::Mat element = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(config_.dilateKernelSize, config_.dilateKernelSize));
    cv::dilate(gaussian, dilated, element);

    return dilated;
}

std::vector<LightDescriptor> ArmorDetector::filterLights(
    const std::vector<std::vector<cv::Point>> &contours)
{
    std::vector<LightDescriptor> lights;

    for (const auto &contour : contours)
    {
        double area = cv::contourArea(contour);

        // 面积过滤
        if (area < config_.minLightArea || contour.size() <= 1)
            continue;

        // 椭圆拟合
        cv::RotatedRect lightRect = cv::fitEllipse(contour);

        // 长宽比过滤
        double ratio = lightRect.size.width / lightRect.size.height;
        if (ratio > config_.maxLightRatio)
            continue;

        // 扩大灯柱面积（膨胀效果）
        lightRect.size.height *= 1.2;
        lightRect.size.width *= 1.2;

        lights.emplace_back(lightRect);
    }

    return lights;
}

std::vector<ArmorDescriptor> ArmorDetector::matchLights(const std::vector<LightDescriptor> &lights)
{
    std::vector<ArmorDescriptor> armors;

    for (size_t i = 0; i < lights.size(); i++)
    {
        for (size_t j = i + 1; j < lights.size(); j++)
        {
            const LightDescriptor &leftLight = lights[i];
            const LightDescriptor &rightLight = lights[j];

            // 角度差
            float angleDiff = std::abs(leftLight.angle - rightLight.angle);

            // 长度差比率
            float lenDiffRatio = std::abs(leftLight.length - rightLight.length) /
                                  std::max(leftLight.length, rightLight.length);

            // 初步筛选
            if (angleDiff > config_.maxAngleDiff || lenDiffRatio > config_.maxLenDiffRatio)
                continue;

            // 计算距离
            float dx = leftLight.center.x - rightLight.center.x;
            float dy = leftLight.center.y - rightLight.center.y;
            float distance = std::sqrt(dx * dx + dy * dy);

            // 均长
            float meanLen = (leftLight.length + rightLight.length) / 2;

            // 各种比率
            float lenGapRatio = std::abs(leftLight.length - rightLight.length) / meanLen;
            float yGapRatio = std::abs(dy) / meanLen;
            float xGapRatio = std::abs(dx) / meanLen;
            float distanceRatio = distance / meanLen;

            // 详细筛选
            if (lenGapRatio > config_.maxLenGapRatio ||
                yGapRatio > config_.maxYGapRatio ||
                xGapRatio > config_.maxXGapRatio ||
                xGapRatio < config_.minXGapRatio ||
                distanceRatio > config_.maxDistanceRatio ||
                distanceRatio < config_.minDistanceRatio)
                continue;

            // 创建装甲板
            ArmorDescriptor armor;
            armor.leftLight = leftLight;
            armor.rightLight = rightLight;

            // 计算中心
            armor.center = cv::Point2f(
                (leftLight.center.x + rightLight.center.x) / 2,
                (leftLight.center.y + rightLight.center.y) / 2);

            // 创建旋转矩形
            armor.rect = cv::RotatedRect(
                armor.center,
                cv::Size(distance, meanLen),
                (leftLight.angle + rightLight.angle) / 2);

            // 计算置信度
            armor.confidence = (1 - lenDiffRatio) * (1 - yGapRatio / config_.maxYGapRatio);

            armors.push_back(armor);
        }
    }

    return armors;
}

void ArmorDetector::drawResult(cv::Mat &frame, const std::vector<ArmorDescriptor> &armors,
                                const cv::Scalar &color)
{
    for (const auto &armor : armors)
    {
        cv::Point2f vertices[4];
        armor.rect.points(vertices);

        for (int i = 0; i < 4; i++)
        {
            cv::line(frame, vertices[i], vertices[(i + 1) % 4], color, 2);
        }

        // 绘制中心点
        cv::circle(frame, armor.center, 5, cv::Scalar(0, 255, 0), -1);
    }
}

} // namespace armor
