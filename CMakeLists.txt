# CMakeのバージョン
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

include(FindPkgConfig)

SET(CMAKE_CXX_COMPILER "clang++")

# リリースビルド用オプション
SET(CMAKE_C_FLAGS_RELEASE "-Wall -O3")

# デバッグビルド用オプション
SET(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS_DEBUG "-g -pg")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -pg -O3")
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -O3")
set(CMAKE_CXX_FLAGS_MINSIZEREL "")

# ビルドのタイプはリリース
SET(CMAKE_BUILD_TYPE Release)
#SET(CMAKE_BUILD_TYPE Debug)
# 環境変数の登録


# プロジェクト名
PROJECT(HFMD_demo_noQt)

# 実行に必要なライブラリのパッケージを見つける
FIND_PACKAGE(OpenCV)
FIND_PACKAGE(Boost)
pkg_check_modules(LIBUSB REQUIRED libusb-1.0)
include_directories (${LIBUSB_INCLUDE_DIRS})
link_directories (${LIBUSB_LIBRARY_DIRS})

FIND_PATH(OPEN_NI_INCLUDE "XnOpenNI.h" "OpenNIConfig.h" HINTS "$ENV{OPEN_NI_INCLUDE}" "/usr/include/ni" "/usr/include/openni" "/opt/ros/groovy/include/openni_camera")
FIND_LIBRARY(OPEN_NI_LIBRARY NAMES OpenNI libOpenNI HINTS $ENV{OPEN_NI_LIB} "/usr/lib")
LINK_DIRECTORIES($ENV{OPEN_NI_LIB})
INCLUDE_DIRECTORIES(${OPEN_NI_INCLUDE})
LINK_LIBRARIES(${OPEN_NI_LIBRARY})
#FIND_PATH(XN_NITE_INCLUDE "libXnVNite.so" HINTS "$ENV{XN_NITE_INSTALL_PATH}" "/usr/include/nite")
#FIND_LIBRARY(XN_NITE_LIBRARY NAMES libXnVNite_1_5_2.so HINTS $ENV{XN_NITE_INSTALL_PATH} "/usr/lib")
#LINK_DIRECTORIES($ENV{XN_NITE_LIB_INSTALL_PATH} "/usr/lib")
#INCLUDE_DIRECTORIES(${XN_NITE_INCLUDE})
#LINK_LIBRARIES(${XN_NITE_LIBRARY})
set(LIBS ${LIBS} ${OPENNI_LIBRARIES})

# バージョン番号
# SET(serial "1.0.0")

# ビルドするためにインクルードするヘッダーファイルの場所
INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}")
INCLUDE_DIRECTORIES("/usr/local/include")
INCLUDE_DIRECTORIES(${OPENCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

# ライブラリの場所
LINK_DIRECTORIES("/usr/local/lib")

SET(RANDOM_FOREST source/CRForest.cpp source/util.cpp source/CRTree.cpp source/HoG.cpp source/CDataset.cpp source/ctlkinect.cpp)

# 実行ファイルを生成
ADD_EXECUTABLE(HFMD_demo_noQt source/objectPoseEstimation.cpp ${RANDOM_FOREST})
#ADD_EXECUTABLE(learning source/learning.cpp ${RANDOM_FOREST})

# ライブラリとリンクする
TARGET_LINK_LIBRARIES(HFMD_demo_noQt ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} ${SSE_FLAGS} ${OPEN_NI_LIBRARIES})
#TARGET_LINK_LIBRARIES(learning ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})

# バージョン番号の適用
# SET_TARGET_PROPERTIES(objectPoseEstimation PROPERTIES VERSION ${serial})
# SET_TARGET_PROPERTIES(learning PROPERTIES VERSION ${serial})
