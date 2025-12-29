cmake_minimum_required(VERSION 3.10)


#project name and language
set(PACKAGE_NAME CUT_SAVE) # 이 부분이 문제였음
project( "main" )

#project will use the C++11 standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



# opencv 디렉토리 찾기
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS}) 

# include directories
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories("${PROJECT_SOURCE_DIR}/include")


# Specify the cross-compilers
#set(CMAKE_C_COMPILER /usr//bin/gcc)        # Replace with your GCC path
#set(CMAKE_CXX_COMPILER /mnt/c/Windows/System32/mingw64/bin/g++)

#debugging option(default : release mode)
set(CMAKE_BUILD_TYPE Debug)

#실행파일 추가
add_executable(main src/main.cpp) #check your file path
target_link_libraries(main ${OpenCV_LIBS})

if (WIN32)
  set_target_properties(main PROPERTIES SUFFIX ".exe")
endif()
 