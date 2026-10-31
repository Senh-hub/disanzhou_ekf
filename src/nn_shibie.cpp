// 复用第二周装甲板网络输入输出协议，仅保留本周所需的检测。
#include "nn_shibie.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <vector>
#include <opencv2/dnn.hpp>
namespace
{
float zhuan_score(float v)
{
    if (
        v >= 0.0f &&
        v <= 1.0f
    )
    {
        return v;
    }
    return
        1.0f /
        (
            1.0f +
            std::exp(-v)
        );
}
}
NnShibie::NnShibie(
    const std::string& model_path,
    float yuzhi,
    float nms
)
:
    yuzhi_(yuzhi),
    nms_(nms)
{
    if (
        !std::filesystem::exists(
            model_path
        )
    )
    {
        keyong_ = false;
        return;
    }
    std::shared_ptr<ov::Model> model =
        core_.read_model(
            model_path
        );
    const ov::Shape input_shape =
        model->input().get_shape();
    if (
        input_shape.size() != 4
    )
    {
        throw std::runtime_error(
            "NN输入维度不是4维"
        );
    }
    if (
        input_shape[1] != 3
    )
    {
        throw std::runtime_error(
            "NN输入通道数不是3"
        );
    }
    input_gao_ =
        static_cast<int>(
            input_shape[2]
        );
    input_kuan_ =
        static_cast<int>(
            input_shape[3]
        );
    compiled_ =
        core_.compile_model(
            model,
            "CPU"
        );
    request_ =
        compiled_.create_infer_request();
    keyong_ = true;
}
bool NnShibie::keyong() const
{
    return keyong_;
}
cv::Mat NnShibie::yuchuli(
    const cv::Mat& tu,
    float& s,
    int& px,
    int& py
) const
{
    if (tu.empty())
    {
        return {};
    }
    float sx =
        input_kuan_ /
        static_cast<float>(
            tu.cols
        );
    float sy =
        input_gao_ /
        static_cast<float>(
            tu.rows
        );
    s =
        std::min(
            sx,
            sy
        );
    int new_kuan =
        static_cast<int>(
            std::round(
                tu.cols * s
            )
        );
    int new_gao =
        static_cast<int>(
            std::round(
                tu.rows * s
            )
        );
    cv::Mat resize_tu;
    cv::resize(
        tu,
        resize_tu,
        cv::Size(
            new_kuan,
            new_gao
        )
    );
    px =
        (
            input_kuan_ -
            new_kuan
        )
        / 2;
    py =
        (
            input_gao_ -
            new_gao
        )
        / 2;
    cv::Mat canvas(
        input_gao_,
        input_kuan_,
        CV_8UC3,
        cv::Scalar(
            114,
            114,
            114
        )
    );
    resize_tu.copyTo(
        canvas(
            cv::Rect(
                px,
                py,
                new_kuan,
                new_gao
            )
        )
    );
    return canvas;
}
BanType NnShibie::panduan_type(
    const std::array<cv::Point2f, 4>& p
)
{
    float kuan =
        0.5f
        *
        (
            static_cast<float>(
                cv::norm(
                    p[0] -
                    p[3]
                )
            )
            +
            static_cast<float>(
                cv::norm(
                    p[1] -
                    p[2]
                )
            )
        );
    float gao =
        0.5f
        *
        (
            static_cast<float>(
                cv::norm(
                    p[0] -
                    p[1]
                )
            )
            +
            static_cast<float>(
                cv::norm(
                    p[3] -
                    p[2]
                )
            )
        );
    float bili =
        kuan /
        (
            gao +
            1e-6f
        );
    if (
        bili >
        3.2f
    )
    {
        return BanType::DA;
    }
    return BanType::XIAO;
}
std::vector<Zhuangjiaban>
NnShibie::jiance(
    const cv::Mat& tu
)
{
    if (
        !keyong_ ||
        tu.empty()
    )
    {
        return {};
    }
    float s = 1.0f;
    int px = 0;
    int py = 0;
    cv::Mat input =
        yuchuli(
            tu,
            s,
            px,
            py
        );
    if (input.empty())
    {
        return {};
    }
    ov::Shape tensor_shape = {
        1,
        3,
        static_cast<std::size_t>(
            input_gao_
        ),
        static_cast<std::size_t>(
            input_kuan_
        )
    };
    ov::Tensor input_tensor(
        ov::element::f32,
        tensor_shape
    );
    float* data =
        input_tensor.data<float>();
    const int area =
        input_gao_ *
        input_kuan_;
    for (
        int y = 0;
        y < input_gao_;
        ++y
    )
    {
        const cv::Vec3b* row =
            input.ptr<cv::Vec3b>(
                y
            );
        for (
            int x = 0;
            x < input_kuan_;
            ++x
        )
        {
            const cv::Vec3b& pixel =
                row[x];
            int id =
                y *
                input_kuan_
                +
                x;
            data[id] =
                pixel[2]
                /
                255.0f;
            data[
                area +
                id
            ] =
                pixel[1]
                /
                255.0f;
            data[
                area * 2 +
                id
            ] =
                pixel[0]
                /
                255.0f;
        }
    }
    request_.set_input_tensor(
        input_tensor
    );
    request_.infer();
    ov::Tensor output =
        request_.get_output_tensor();
    ov::Shape shape =
        output.get_shape();
    if (
        shape.size() != 3
    )
    {
        throw std::runtime_error(
            "NN输出不是3维"
        );
    }
    if (
        shape[0] != 1 ||
        shape[1] != 21
    )
    {
        throw std::runtime_error(
            "NN输出不是预期的 [1,21,N]"
        );
    }
    int num =
        static_cast<int>(
            shape[2]
        );
    std::size_t total =
        21
        *
        static_cast<std::size_t>(
            num
        );
    std::vector<float> out_data(
        total
    );
    if (
        output.get_element_type()
        ==
        ov::element::f32
    )
    {
        const float* src =
            output.data<const float>();
        std::copy(
            src,
            src + total,
            out_data.begin()
        );
    }
    else if (
        output.get_element_type()
        ==
        ov::element::f16
    )
    {
        const ov::float16* src =
            output.data<const ov::float16>();
        for (
            std::size_t i = 0;
            i < total;
            ++i
        )
        {
            out_data[i] =
                static_cast<float>(
                    src[i]
                );
        }
    }
    else
    {
        throw std::runtime_error(
            "NN输出数据类型不是FP32或FP16"
        );
    }
    auto at =
        [&](int channel, int id)
        {
            return
                out_data[
                    static_cast<std::size_t>(
                        channel
                    )
                    *
                    num
                    +
                    id
                ];
        };
    std::vector<Zhuangjiaban>
        houxuan;
    std::vector<cv::Rect>
        boxes;
    std::vector<float>
        scores;
    for (
        int i = 0;
        i < num;
        ++i
    )
    {
        int color_id = 0;
        float color_score =
            zhuan_score(
                at(
                    0,
                    i
                )
            );
        for (
            int k = 1;
            k < 4;
            ++k
        )
        {
            float sc =
                zhuan_score(
                    at(
                        k,
                        i
                    )
                );
            if (
                sc >
                color_score
            )
            {
                color_score =
                    sc;
                color_id =
                    k;
            }
        }
        int cls_id = 0;
        float cls_score =
            zhuan_score(
                at(
                    4,
                    i
                )
            );
        for (
            int k = 1;
            k < 9;
            ++k
        )
        {
            float sc =
                zhuan_score(
                    at(
                        4 + k,
                        i
                    )
                );
            if (
                sc >
                cls_score
            )
            {
                cls_score =
                    sc;
                cls_id =
                    k;
            }
        }
        float conf =
            color_score
            *
            cls_score;
        if (
            conf <
            yuzhi_
        )
        {
            continue;
        }
        Zhuangjiaban ban;
        bool inside =
            true;
        for (
            int k = 0;
            k < 4;
            ++k
        )
        {
            float x =
                (
                    at(
                        13 + k * 2,
                        i
                    )
                    -
                    static_cast<float>(
                        px
                    )
                )
                /
                s;
            float y =
                (
                    at(
                        14 + k * 2,
                        i
                    )
                    -
                    static_cast<float>(
                        py
                    )
                )
                /
                s;
            ban.dian[k] =
                cv::Point2f(
                    x,
                    y
                );
            if (
                x < 0.0f ||
                x >=
                    static_cast<float>(
                        tu.cols
                    )
                ||
                y < 0.0f ||
                y >=
                    static_cast<float>(
                        tu.rows
                    )
            )
            {
                inside =
                    false;
            }
        }
        if (!inside)
        {
            continue;
        }
        std::vector<cv::Point2f> p(
            ban.dian.begin(),
            ban.dian.end()
        );
        cv::Rect box =
            cv::boundingRect(
                p
            );
        if (
            box.width <= 0 ||
            box.height <= 0
        )
        {
            continue;
        }
        ban.center =
            (
                ban.dian[0]
                +
                ban.dian[1]
                +
                ban.dian[2]
                +
                ban.dian[3]
            )
            *
            0.25f;
        ban.type =
            panduan_type(
                ban.dian
            );
        ban.score =
            conf;
        ban.color =
            color_id;
        static const char*
        labels[9] = {
            "sentry",
            "hero",
            "engineer",
            "3",
            "4",
            "5",
            "outpost",
            "base_bottom",
            "base_top"
        };
        ban.number =
            labels[
                cls_id
            ];
        houxuan.push_back(
            ban
        );
        boxes.push_back(
            box
        );
        scores.push_back(
            conf
        );
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxes(
        boxes,
        scores,
        yuzhi_,
        nms_,
        keep
    );
    std::vector<Zhuangjiaban>
        out;
    out.reserve(
        keep.size()
    );
    for (
        int id :
        keep
    )
    {
        out.push_back(
            houxuan[id]
        );
    }
    return out;
}