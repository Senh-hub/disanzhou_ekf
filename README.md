# disanzhou_ekf：第三周独立工程
神经网络检测 → PnP位姿 → IMU时间同步/坐标变换 → 整车EKF → 四板重投影、状态CSV与曲线。



## 使用


 使用 C++17、CMake、OpenCV、OpenVINO 环境；以下命令在本项目根目录执行：


cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target disanzhou_ekf week3_tests -j2
ctest --test-dir build --output-on-failure
./build/disanzhou_ekf config/disanzhou.yaml
python3 -m pip install numpy matplotlib
python3 tools/hua_quxian.py result/week3/zhuangtai.csv
```



## 内容

| 目录 | 内容 |
|---|---|
| app/main.cpp | 本周唯一业务程序入口 |
| include、src | 检测、位姿/协方差、IMU、EKF与投影 |
| config | 本周相机参数、9ms偏移、模型假设 |
| model/armor_v8.onnx | 装甲板检测模型 |
| data/imu/imu.txt | 已提供的531行IMU |
| tools/hua_quxian.py | 绘制11个状态和不确定性 |
| tests | 数学与同步测试 |

## 输出

运行成功后生成 result/week3/zhengche_result.mp4、zhuangtai.csv、run_summary.txt；再运行绘图脚本生成 zhuangtai_quxian.png 和 .pdf。

## 必须了解的配置假设




