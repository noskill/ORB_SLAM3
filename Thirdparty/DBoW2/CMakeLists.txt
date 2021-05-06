cmake_minimum_required(VERSION 2.8)
project(DBoW2)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")

set(HDRS_DBOW2
  DBoW2/BowVector.h
  DBoW2/FORB.h 
  DBoW2/FGOOD.h 
  DBoW2/FClass.h       
  DBoW2/FeatureVector.h
  DBoW2/ScoringObject.h   
  DBoW2/QueryResults.h
  DBoW2/TemplatedVocabulary.h)
set(SRCS_DBOW2
  DBoW2/BowVector.cpp
  DBoW2/FORB.cpp      
  DBoW2/FGOOD.cpp      
  DBoW2/FeatureVector.cpp
  DBoW2/QueryResults.cpp
  DBoW2/ScoringObject.cpp)

set(HDRS_DUTILS
  DUtils/Random.h
  DUtils/Timestamp.h)
set(SRCS_DUTILS
  DUtils/Random.cpp
  DUtils/Timestamp.cpp)

set(SRC_NDARRAY
     DBoW2/ndarray_converter.cpp)

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      find_package(OpenCV 4 QUIET)
      if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
      endif()
   endif()
endif()
set(RAPIDJSON_INCLUDE_DIR "" CACHE FILEPATH "Path to rapidjson includes")

if(NOT EXISTS ${RAPIDJSON_INCLUDE_DIR}/rapidjson/rapidjson.h)
	message(FATAL_ERROR "Can't find rapidjson.h in ${RAPIDJSON_INCLUDE_DIR}/rapidjson")
endif()

find_package(PythonLibs 3 REQUIRED)
find_package (Python3 COMPONENTS NumPy)

include_directories(${OpenCV_INCLUDE_DIRS} ${RAPIDJSON_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
add_library(DBoW2 SHARED ${SRCS_DBOW2} ${SRCS_DUTILS})
target_link_libraries(DBoW2 PRIVATE ${OpenCV_LIBS})

add_library(ndarray STATIC ${SRC_NDARRAY})
set_property(TARGET ndarray PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ndarray PRIVATE ${PYTHON_LIBRARIES} ${OpenCV_LIBS})

add_subdirectory(pybind11)
pybind11_add_module(DBoW DBoW2/binding.cpp)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(DBoW PRIVATE DBoW2 ndarray)
