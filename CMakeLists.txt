cmake_minimum_required(VERSION 3.6)
project(Vis)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -Ofast -march=native -ffast-math -fopenmp")

#add_custom_target(Vis COMMAND make -C ${Vis_SOURCE_DIR}
#                         CLION_EXE_DIR=${PROJECT_BINARY_DIR})

include_directories("${PROJECT_SOURCE_DIR}/general")

# 3rd party
###############################################################################
# boost
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})
###############################################################################


###############################################################################
add_executable(Vis main.cpp LargeVis.cpp)
target_link_libraries(Vis pthread gsl gslcblas ${Boost_LIBRARIES})
###############################################################################
#
#LDFLAGS = $(OPTM) -lm -pthread -lgsl -lgslcblas -lboost_timer -lboost_system -DINFO
