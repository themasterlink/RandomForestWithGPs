cmake_minimum_required(VERSION 3.7)
project(RandomForestWithGPs)

set(CMAKE_CXX_STANDARD 14)

# add this for terminal execution
if(CMAKE_BUILD_TYPE MATCHES Release)
    add_definitions(-DUSE_SCREEN_OUPUT)
endif(CMAKE_BUILD_TYPE MATCHES Release)

# for pthreads
find_package(Threads REQUIRED)

#for cursers and panel
INCLUDE_DIRECTORIES(/usr/include/ncurses/)

#Boost include:
find_package(Boost COMPONENTS system filesystem program_options random date_time thread REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})

#Eigen include:
include_directories(/home/f_soft/foreign_packages/Eigen/3.3.2/include/)

set(GDAL_LIBRARIES "/home/f_soft/foreign_packages/gdal/1.10/lib/osl42-x86_64")
set(TBB_LIBRARIES "/home/f_soft/foreign_packages/tbb/4.3.0/lib/osl42-x86_64")

# opencv
set (OpenCV_DIR "/home/f_soft/foreign_packages/OpenCV/3.1.0/share/OpenCV/")
find_package(OpenCV COMPONENTS opencv_core opencv_imgcodecs opencv_highgui opencv_ml opencv_imgproc opencv_flann REQUIRED)
#include_directories("/home/f_soft/foreign_packages/OpenCV/3.1.0/include/")

set(SOURCE_FILES
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
        src/Data/ClassData.cc
        src/Data/ClassData.h
        src/Data/ClassKnowledge.cc
        src/Data/ClassKnowledge.h
        src/Data/ClassPoint.cc
        src/Data/ClassPoint.h
        src/Data/Data.h
        src/Data/DataBinaryWriter.cc
        src/Data/DataBinaryWriter.h
        src/Data/DataConverter.cc
        src/Data/DataConverter.h
        src/Data/DataPoint.cc
        src/Data/DataPoint.h
        src/Data/DataReader.cc
        src/Data/DataReader.h
        src/Data/DataSets.h
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
        src/RandomForests/DynamicDecisionTree.cc
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
        src/kernelCalc.h
        src/main.cc)

add_executable(RandomForestWithGPs ${SOURCE_FILES})
target_link_libraries(RandomForestWithGPs ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(RandomForestWithGPs ${GDAL_LIBRARIES})
target_link_libraries(RandomForestWithGPs ${TBB_LIBRARIES})
target_link_libraries(RandomForestWithGPs "/usr/lib/" ncurses panel)
target_link_libraries(RandomForestWithGPs ${Boost_LIBRARIES} )
target_link_libraries(RandomForestWithGPs ${OpenCV_LIBRARIES} )
#target_link_libraries(RandomForestWithGPs "/home/f_soft/foreign_packages/OpenCV/3.1.0/lib/osl42-x86_64" opencv_core opencv_imgcodecs opencv_highgui opencv_ml opencv_imgproc opencv_flann)