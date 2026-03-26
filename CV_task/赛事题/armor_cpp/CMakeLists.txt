cmake_minimum_required(VERSION 3.16)
project(armor_detector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")

find_package(OpenCV REQUIRED)

# 头文件目录
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# 源文件
file(GLOB_RECURSE SOURCES "src/*.cpp")

# 可执行文件
add_executable(armor_detector ${SOURCES})

target_link_libraries(armor_detector ${OpenCV_LIBS})

# 安装
install(TARGETS armor_detector DESTINATION bin)
