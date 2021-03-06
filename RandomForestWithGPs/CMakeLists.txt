#set(CMAKE_VERBOSE_MAKEFILE ON)
cmake_minimum_required(VERSION 3.7)
project(RandomForestWithGPs)

set(CMAKE_CXX_STANDARD 14)

add_definitions("-march=native")


# add this for terminal execution
if(CMAKE_BUILD_TYPE MATCHES Release)
    add_definitions(-DUSE_SCREEN_OUPUT)
endif(CMAKE_BUILD_TYPE MATCHES Release)

# for pthreads
find_package(Threads REQUIRED)

#for cursers and panel
INCLUDE_DIRECTORIES(/usr/include/ncurses/)

#Boost include:
find_package(Boost COMPONENTS filesystem program_options date_time REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})

set(USE_OPEN_CV_CMAKE 0)
if(USE_OPEN_CV_CMAKE MATCHES 1)
    ADD_DEFINITIONS( -DUSE_OPEN_CV=1)
endif(USE_OPEN_CV_CMAKE MATCHES 1)

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    #Eigen include:
    include_directories(/home/f_soft/foreign_packages/Eigen/3.3.2/include/)

    set(GDAL_LIBRARIES "/home/f_soft/foreign_packages/gdal/1.10/lib/osl42-x86_64")
    set(TBB_LIBRARIES "/home/f_soft/foreign_packages/tbb/4.3.0/lib/osl42-x86_64")

    # opencv
    if(USE_OPEN_CV_CMAKE MATCHES 1)
        set (OpenCV_DIR "/home/f_soft/foreign_packages/OpenCV/3.1.0/share/OpenCV/")
    endif(USE_OPEN_CV_CMAKE MATCHES 1)
else(CMAKE_SYSTEM_NAME MATCHES  "Linux")
    #Eigen include:
    INCLUDE_DIRECTORIES("/usr/local/include/eigen3/")
    #opencv
    if(USE_OPEN_CV_CMAKE MATCHES 1)
        set (OpenCV_DIR "/usr/local/opt/opencv3/share/OpenCV")
    endif(USE_OPEN_CV_CMAKE MATCHES 1)
endif(CMAKE_SYSTEM_NAME MATCHES "Linux")

# opencv

if(USE_OPEN_CV_CMAKE MATCHES 1)
find_package(OpenCV COMPONENTS opencv_core opencv_imgcodecs opencv_highgui opencv_ml opencv_imgproc opencv_flann REQUIRED)
endif(USE_OPEN_CV_CMAKE MATCHES 1)

set(SOURCE_FILES
        src/Base/Types.h
        src/Base/CommandSettings.cc
        src/Base/CommandSettings.h
        src/Base/InformationPackage.cc
        src/Base/InformationPackage.h
        src/Base/Logger.cc
        src/Base/Logger.h
        src/Base/Observer.cc
        src/Base/Observer.h
        src/Base/Predictor.h
        src/Base/ScreenOutput.cc
        src/Base/ScreenOutput.h
        src/Base/Settings.cc
        src/Base/Settings.h
        src/Base/ThreadMaster.cc
        src/Base/ThreadMaster.h
        src/CMAES/boundary_transformation.cc
        src/CMAES/boundary_transformation.h
        src/CMAES/cmaes.cc
        src/CMAES/cmaes.h
        src/Data/ClassKnowledge.cc
        src/Data/ClassKnowledge.h
        src/Data/LabeledVectorX.cc
        src/Data/LabeledVectorX.h
        src/Data/DataBinaryWriter.cc
        src/Data/DataBinaryWriter.h
        src/Data/DataConverter.cc
        src/Data/DataConverter.h
        src/Data/DataReader.cc
        src/Data/DataReader.h
        src/Data/DataWriterForVisu.cc
        src/Data/DataWriterForVisu.h
        src/Data/OnlineStorage.h
        src/Data/OnlineStorage_i.h
        src/Data/TotalStorage.cc
        src/Data/TotalStorage.h
        src/GaussianProcess/Kernel/GaussianKernel.cc
        src/GaussianProcess/Kernel/GaussianKernel.h
        src/GaussianProcess/Kernel/GaussianKernelOptimizer.cc
        src/GaussianProcess/Kernel/GaussianKernelOptimizer.h
        src/GaussianProcess/Kernel/KernelBase.h
        src/GaussianProcess/Kernel/KernelBase_i.h
        src/GaussianProcess/Kernel/KernelType.cc
        src/GaussianProcess/Kernel/KernelType.h
        src/GaussianProcess/Kernel/RandomForestKernel.cc
        src/GaussianProcess/Kernel/RandomForestKernel.h
        src/GaussianProcess/BayesOptimizer.cc
        src/GaussianProcess/BayesOptimizer.h
        src/GaussianProcess/BayesOptimizerIVM.cc
        src/GaussianProcess/BayesOptimizerIVM.h
        src/GaussianProcess/BestHyperParams.cc
        src/GaussianProcess/BestHyperParams.h
        src/GaussianProcess/GaussianProcess.cc
        src/GaussianProcess/GaussianProcess.h
        src/GaussianProcess/GaussianProcessMultiBinary.cc
        src/GaussianProcess/GaussianProcessMultiBinary.h
        src/GaussianProcess/GaussianProcessMultiClass.cc
        src/GaussianProcess/GaussianProcessMultiClass.h
        src/GaussianProcess/GaussianProcessWriter.cc
        src/GaussianProcess/GaussianProcessWriter.h
        src/GaussianProcess/IVM.cc
        src/GaussianProcess/IVM.h
        src/GaussianProcess/IVMMultiBinary.cc
        src/GaussianProcess/IVMMultiBinary.h
        src/GaussianProcess/LevenbergMarquardtSolver.h
        src/RandomForestGaussianProcess/OnlineRandomForestIVMs.cc
        src/RandomForestGaussianProcess/OnlineRandomForestIVMs.h
        src/RandomForestGaussianProcess/RandomForestGaussianProcess.cc
        src/RandomForestGaussianProcess/RandomForestGaussianProcess.h
        src/RandomForestGaussianProcess/RFGPWriter.cc
        src/RandomForestGaussianProcess/RFGPWriter.h
        src/RandomForests/BigDynamicDecisionTree.cc
        src/RandomForests/BigDynamicDecisionTree.h
        src/RandomForests/DecisionTree.cc
        src/RandomForests/DecisionTree.h
        src/RandomForests/DecisionTreeData.h
        src/RandomForests/DynamicDecisionTree_i.h
        src/RandomForests/DynamicDecisionTree.h
        src/RandomForests/DynamicDecisionTreeInterface.h
        src/RandomForests/OnlineRandomForest.cc
        src/RandomForests/OnlineRandomForest.h
        src/RandomForests/RandomForest.cc
        src/RandomForests/RandomForest.h
        src/RandomForests/RandomForestWriter.cc
        src/RandomForests/RandomForestWriter.h
        src/RandomForests/TreeCounter.h
        src/RandomNumberGenerator/RandomGaussianNr.cc
        src/RandomNumberGenerator/RandomGaussianNr.h
        src/RandomNumberGenerator/RandomNumberGeneratorForDT.cc
        src/RandomNumberGenerator/RandomNumberGeneratorForDT.h
        src/RandomNumberGenerator/RandomUniformNr.cc
        src/RandomNumberGenerator/RandomUniformNr.h
        src/Tests/binaryClassGPTest.h
        src/Tests/binaryClassIVMTest.h
        src/Tests/binaryClassORFTest.h
        src/Tests/binaryClassRFTest.h
        src/Tests/multiClassGPTest.h
        src/Tests/multiClassIVMTest.h
        src/Tests/multiClassORFIVMTest.h
        src/Tests/multiClassRFGPTest.h
        src/Tests/performanceMeasurement.h
        src/Tests/tests.h
        src/UnitTests/DataTests.h
        src/Utility/AvgNumber.h
        src/Utility/ColorConverter.h
        src/Utility/ConfusionMatrixPrinter.cc
        src/Utility/ConfusionMatrixPrinter.h
        src/Utility/InLinePercentageFiller.cc
        src/Utility/InLinePercentageFiller.h
        src/Utility/ReadWriterHelper.cc
        src/Utility/ReadWriterHelper.h
        src/Utility/StopWatch.cc
        src/Utility/StopWatch.h
        src/Utility/ThreadSafeOutput.cc
        src/Utility/ThreadSafeOutput.h
        src/Utility/ThreadSafeThreadCounter.cc
        src/Utility/ThreadSafeThreadCounter.h
        src/Utility/TimeFrame.cc
        src/Utility/TimeFrame.h
        src/Utility/Util.h
        src/main.cc src/Base/BaseType.h src/Tests/TestManager.cc
        src/Tests/TestManager.h src/Utility/StringHelper.h
        src/Tests/TestInformation.cc src/Tests/TestInformation.h
        src/Utility/StringHelper.cc src/Utility/GlobalStopWatch.h
        src/Base/Singleton.h src/Base/Thread.h src/Tests/SpeedTests/ArgMaxAndMin.h src/Tests/SpeedTests/EigenVectorX.h src/Tests/SpeedTests/RandomNumbers.h src/Tests/SpeedTests/TestHeader.h src/RandomForests/GlobalLifeTimeMeasurement.cc src/RandomForests/GlobalLifeTimeMeasurement.h src/RandomForests/AcceptanceCalculator.cc src/RandomForests/AcceptanceCalculator.h src/RandomNumberGenerator/RandomExponentialNr.cc src/RandomNumberGenerator/RandomExponentialNr.h)


add_executable(RandomForestWithGPs ${SOURCE_FILES})

#target_compile_options(RandomForestWithGPs PUBLIC -fsanitize=address)
target_link_libraries(RandomForestWithGPs ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(RandomForestWithGPs ${GDAL_LIBRARIES})
target_link_libraries(RandomForestWithGPs ${TBB_LIBRARIES})
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_compile_definitions(RandomForestWithGPs PRIVATE BUILD_SYSTEM_CMAKE=1)
    target_link_libraries(RandomForestWithGPs "/usr/local/lib/" ncurses panel)
else(CMAKE_SYSTEM_NAME MATCHES  "Linux")
    target_compile_definitions(RandomForestWithGPs PRIVATE BUILD_SYSTEM_CMAKE=0)
    target_link_libraries(RandomForestWithGPs "/usr/lib/" ncurses panel)
endif(CMAKE_SYSTEM_NAME MATCHES "Linux")

target_link_libraries(RandomForestWithGPs ${Boost_LIBRARIES} )
#opencv
if(USE_OPEN_CV_CMAKE MATCHES 1)
target_link_libraries(RandomForestWithGPs ${OpenCV_LIBRARIES} )
endif(USE_OPEN_CV_CMAKE MATCHES 1)
#target_link_libraries(RandomForestWithGPs "/home/f_soft/foreign_packages/OpenCV/3.1.0/lib/osl42-x86_64" opencv_core opencv_imgcodecs opencv_highgui opencv_ml opencv_imgproc opencv_flann)