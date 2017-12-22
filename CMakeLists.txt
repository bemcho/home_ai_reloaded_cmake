cmake_minimum_required(VERSION 3.0)
project(home_ai_reloaded C CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set (CMAKE_CXX_FLAGS "-O3 -Weverything  -Wno-c++98-compat -Wno-global-constructors -Wno-exit-time-destructors -Wno-padded -Wno-missing-prototypes -Wno-missing-variable-declarations")

file(GLOB SOURCE_MAIN
        "${CMAKE_SOURCE_DIR}/headers/*.hpp"
        "${CMAKE_SOURCE_DIR}/src/*.cpp"
        )

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

find_package(Tesseract REQUIRED)
if(Tesseract_FOUND)
    include_directories(${Tesseract_INCLUDE_DIRS})
endif()

find_package(TBB REQUIRED)
if(TBB_FOUND)
    include_directories(${TBB_INCLUDE_DIRS})
endif()

find_library(Clips_LIBRARY NAMES clips
        HINTS
        /usr/lib
        /usr/local/lib)

if(Clips_LIBRARY)
    message(STATUS "Find Clips:   YES")
else()
    message(STATUS "Find Clips:   NO")
endif()

#[[find_package(Leptonica REQUIRED)
if(Leptonica_FOUND)
    include_directories(${Leptonica_INCLUDE_DIRS})
    set(Lept_LIBRARY ${LEPTONICA_LIBRARIES})
endif()]]
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(LIBS ${LIBS} ${OpenCV_LIBS} ${Tesseract_LIBS} ${TBB_LIBRARIES} ${Clips_LIBRARY}  Threads::Threads)
add_executable(home_ai_reloaded ${SOURCE_MAIN} src/ClipsAdapter.cpp)
target_link_libraries( home_ai_reloaded  ${LIBS})