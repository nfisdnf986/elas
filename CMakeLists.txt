# project
cmake_minimum_required (VERSION 2.6)
project (libelas)

#set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/)

# Add to module path, so we can find our cmake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules)

# make an uninstall target
include(${CMAKE_MODULE_PATH}/cmake_uninstall.cmake.in)
add_custom_target(uninstall
    COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake")


find_package(OpenCV REQUIRED)
find_package(GFlags REQUIRED)
find_package(GLog REQUIRED)
find_package(Protobuf REQUIRED)
find_package(HAL REQUIRED)
find_package(Calibu REQUIRED)

# directories
set (LIBELAS_SRC_DIR src)

# include directory
#~ include_directories("${LIBELAS_SRC_DIR}")

# use sse3 instruction set
SET(CMAKE_CXX_FLAGS "-std=c++11 -msse3")

# sources
FILE(GLOB LIBELAS_SRC_FILES "src/*.cpp")

# make release version
# set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)

MESSAGE( STATUS "ghost CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}" )

# build demo program
add_executable(elas ${LIBELAS_SRC_FILES})

include_directories("${LIBELAS_SRC_DIR}"
  ${GFLAGS_INCLUDE_DIR}
  ${GLog_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}/HAL
  ${HAL_INCLUDE_DIRS}
  ${Calibu_INCLUDE_DIRS}
)

target_link_libraries( elas
  ${HAL_LIBRARIES}
  ${OpenCV_LIBS}
  ${GFLAGS_LIBRARIES}
  ${GLog_LIBRARIES}
  ${PROTOBUF_LIBRARY}
  ${Calibu_LIBRARY}
)
