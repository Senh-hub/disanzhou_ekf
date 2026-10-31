#pragma once
#include <array>
#include <string>
#include <opencv2/core.hpp>

// 第三周只传递检测角点、编号、颜色与置信度，不再包含灯条/数字裁剪图。
enum class BanType { XIAO, DA, INVALID };
struct Zhuangjiaban {
    std::array<cv::Point2f, 4> dian{}; // 左上、左下、右下、右上
    cv::Point2f center;
    BanType type = BanType::INVALID;
    std::string number = "?";
    float score = 0.0f;
    int color = -1;
};
