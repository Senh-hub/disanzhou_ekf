

本次作业在第二周装甲板识别的基础上，使用 PnP 解算装甲板位姿，再通过 EKF 估计整车状态。输出处理视频、状态曲线和 CSV 数据。


项目结构、代码设计与运行方法
1. 项目结构
本次项目名为 disanzhou_ekf，复用了第二周的网络识别部分，主要文件如下：
文件	作用
app/main.cpp	主程序，连接视频读取、识别、估计和输出
src/nn_shibie.cpp	网络推理和装甲板角点检测
src/xiangji_canshu.cpp	相机参数读取
include/imu_tongbu.hpp	IMU 数据读取、四元数插值和姿态转换
include/zhengche_ekf.hpp	EKF 状态预测、匹配和更新
include/guance_touying.hpp	PnP、观测协方差和四板重投影
config/disanzhou.yaml	输入路径、同步和滤波参数
config/xiangji_shipin.yaml	相机内参和畸变参数
tools/hua_quxian.py	根据 CSV 绘制状态曲线
tests/week3_tests.cpp	同步、雅可比、换板和漏检等测试


程序的处理顺序为：读取图像和 IMU 数据，进行时间同步，识别装甲板角点，求解 PnP 并转换坐标，完成 EKF 预测与更新，最后输出状态和重投影画面。
2. 时间同步与坐标转换
IMU 文件每行的数据是时间戳和四元数，四元数顺序为 w、x、y、z。根据给定的 9 ms 偏移，图像时间为 t 时，使用 IMU 中 t−0.009 秒对应的姿态。查询时间落在两条数据之间时，使用 SLERP 插值；超出数据范围时不外推。


PnP 使用本周提供的相机内参和畸变参数，根据四个角点及装甲板尺寸计算位姿。小装甲板尺寸设为 135×55 mm，大装甲板设为 225×55 mm。

PnP 得到的是相机坐标系下的结果，再结合 IMU 姿态转换到统一的参考坐标系。由于缺少实测外参，目前假定相机坐标系为 x 向右、y 向下、z 向前，IMU 坐标系为 x 向前、y 向左、z 向上，四元数表示 IMU 到参考系的旋转。安装平移设为零，也没有估计拍摄平台自身的平移。因此，结果是在这些假设下得到的，不能直接当作准确的世界坐标。
3. EKF 状态与运动模型
PnP 得到的是装甲板的位置，不能直接当成车中心的位置。因为车辆旋转时，装甲板会绕着车中心运动，所以需要建立整车模型。

状态向量为：
[cx, vx, cy, vy, z0, vz, yaw, omega, r0, r1, dz]

其中 z0 表示第 0、2 块板的中心高度，dz 表示第 1、3 块板相对它们的高度差；r0、r1 分别表示两对相对装甲板的半径。

运动模型采用短时间匀速平移和匀角速度旋转：
cx_new  = cx  + vx    × dt
cy_new  = cy  + vy    × dt
z0_new  = z0  + vz    × dt
yaw_new = yaw + omega × dt

其余状态在预测时保持不变，dt 使用相邻图像时间戳的差值。实际目标可能加速或者减速，因此需要加入过程噪声，再通过当前观测修正预测。
4. 观测模型与协方差
观测量是装甲板中心的 x、y、z 和外法向 yaw。四块装甲板编号为 0、1、2、3，相邻两块间隔 π/2。

对于第 j 块板，观测模型可以写成：
该板角度 = yaw + j × π/2

偶数编号使用半径 r0，奇数编号使用半径 r1

板中心 x = cx + 对应半径 × cos(该板角度)
板中心 y = cy + 对应半径 × sin(该板角度)

偶数编号板的高度 = z0
奇数编号板的高度 = z0 + dz

程序根据整车状态计算装甲板应该出现的位置，再与实际观测比较。由于模型中包含 sin 和 cos，所以使用 EKF，通过雅可比矩阵对当前状态附近的关系进行近似。

协方差部分主要包括 P、Q、R：
- P 表示当前状态估计的不确定性。
- Q 表示运动预测的不确定性。
- R 表示观测的不确定性。
预测时根据运动模型传播 P，并加入 Q。Q 随 dt 变化，平移和旋转分别配置噪声，半径和高度差使用较小的随机变化。

R 从角点像素误差出发，通过投影雅可比近似传播到位姿，再转换为位置和 yaw 的误差。像素噪声标准差设为 1.5 px，位置和 yaw 的噪声底限标准差分别设为 0.02 m 和 0.15 rad。相机标定的 0.1448 px 重投影误差不等于实际检测误差，所以没有直接把它当成 R。

更新时，程序根据预测和观测各自的不确定性决定修正多少。P 使用 Joseph 形式更新，并进行对称化处理，减少数值误差。不过，目前还没有完整考虑外参和同步误差，协方差的准确性仍需进一步验证。
5. 匹配、换板与丢失处理
程序锁定同颜色、同编号的目标。IPPE 可能为一块板提供两个位姿候选，先排除负深度以及重投影 RMSE 超过 5 px 的结果，再与预测的四块模型板进行比较。

匹配使用 NIS。我的理解是，它不仅比较观测与预测差了多少，还考虑这个差别相对于当前不确定性是否合理。程序逐次选择通过门限且 NIS 最小的匹配，门限为 18.47。同一个检测和同一块模型板在一帧内最多使用一次。

换板时，改变的是匹配到的板编号，不能把两块板之间约 π/2 的差别直接当成车体突然转动。程序保持 yaw 连续累计，只将角度残差归一化到 −π 到 π。

短时间没有有效观测时只进行预测；超过 0.6 秒没有有效更新时重新初始化，并增加 segment 编号。曲线在不同 segment 之间断开。另外，半径超出 0.05～0.8 m，或者高度差绝对值超过 0.5 m 的更新会被拒绝。
6. 四板重投影
根据估计出的中心、角度、半径和高度差，计算四块装甲板的中心，再按尺寸生成各自的四个三维角点。之后转换回相机坐标系，使用相机内参和畸变参数投影到图像上。

目前板面倾角暂设为 0°，如果与实际安装角度不同，也会影响重投影效果。

7. 依赖配置与构建运行
本次使用 WSL Ubuntu 和 VS Code，依赖为 C++17、CMake、OpenCV、OpenVINO Runtime。绘图使用 Python 3、NumPy 和 Matplotlib。

OpenCV 和 OpenVINO 沿用第二周配置好的环境。如果 CMake 找不到 OpenVINO，需要通过 OpenVINO_DIR 指定实际安装中的配置目录。

运行前，将文件放到以下位置：
视频：data/shipin/shipin.avi
IMU：data/imu/imu.txt
模型：model/armor_v8.onnx

在项目根目录编译和运行：
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target disanzhou_ekf week3_tests -j2
ctest --test-dir build --output-on-failure
./build/disanzhou_ekf config/disanzhou.yaml

运行完成后绘制曲线：
python3 tools/hua_quxian.py result/week3/zhuangtai.csv

如果缺少绘图库，可以安装：
sudo apt update
sudo apt install -y python3-numpy python3-matplotlib



