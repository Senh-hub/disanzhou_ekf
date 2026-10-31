cmake_minimum_required(VERSION 3.16)
project(disanzhou_ekf LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d dnn videoio imgcodecs)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

add_library(disanzhou_core
    src/nn_shibie.cpp
    src/xiangji_canshu.cpp
)
target_include_directories(disanzhou_core PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(disanzhou_core PUBLIC ${OpenCV_LIBS} openvino::runtime)

add_executable(disanzhou_ekf app/main.cpp)
target_link_libraries(disanzhou_ekf PRIVATE disanzhou_core)

include(CTest)
if(BUILD_TESTING)
    add_executable(week3_tests tests/week3_tests.cpp)
    target_include_directories(week3_tests PRIVATE include ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(week3_tests PRIVATE ${OpenCV_LIBS})
    add_test(NAME week3_math COMMAND week3_tests)
    set_tests_properties(week3_math PROPERTIES TIMEOUT 30)
endif()
