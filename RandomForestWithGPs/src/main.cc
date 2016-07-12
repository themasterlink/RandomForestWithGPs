//============================================================================
// Name        : RandomForestWithGPs.cpp
// Author      : 
// Version     :
// Copyright   : 
// Description :
//============================================================================

#include <iostream>
#include <Eigen/Dense>
#include "kernelCalc.h"
#include "Utility/Settings.h"
#include "Data/DataReader.h"
#include "Data/DataConverter.h"
#include "Data/DataWriterForVisu.h"
#include "RandomForests/RandomForest.h"
#include "RandomForests/RandomForestWriter.h"
#include "GaussianProcess/GaussianProcessMultiClass.h"
#include "RandomForestGaussianProcess/RandomForestGaussianProcess.h"
#include <unsupported/Eigen/NonLinearOptimization>
#include "GaussianProcess/BayesOptimizer.h"
#include <boost/numeric/ublas/assignment.hpp> // <<= op assigment

#include "GaussianProcess/GaussianProcessBinary.h"
// just for testing

void executeForMultiClass(const std::string& path){

	Data data;
	Labels labels;
	DataReader::readFromFile(data, labels, path);

	const int dataPoints = data.size();
	Eigen::MatrixXd dataMat;

	dataMat.conservativeResize(data[0].rows(), data.size());
	int i = 0;
	for(Data::iterator it = data.begin(); it != data.end(); ++it){
		dataMat.col(i++) = *it;
	}
	const int amountOfClass = 2;
	/*std::vector<Data> dataPerClass(amountOfClass);
	for(int i = 0; i < data.size(); ++i){
		dataPerClass[labels[i]].push_back(data[i]);
	}*/
	Eigen::VectorXd y(Eigen::VectorXd::Zero(dataPoints * amountOfClass));
	for(int i = 0; i < labels.size(); ++i){
		y[labels[i] * dataPoints + i] = 1;
	}
	std::fstream f("t.txt", std::ios::out);
	//f << "dataMat:\n" << dataMat << std::endl;
	std::vector<Eigen::MatrixXd> cov;

	Eigen::MatrixXd covariance;
	Kernel kernel;
	kernel.init(dataMat);
	kernel.calcCovariance(covariance);
	f << "covariance: \n" << covariance << std::endl;
	f << "labels: \n";
	for(int i = 0; i < labels.size(); ++i){
		f << "           " << labels[i];
	}
	f << std::endl;
	for(int i = 0; i < amountOfClass; ++i){ // calc the covariance matrix for each f_c
		Eigen::MatrixXd cov_c = covariance; //  * y.segment(i*dataPoints, dataPoints).transpose();
		for(int j = 0; j < dataPoints; ++j){
			if(i == labels[j]){
				for(int k = 0; k < dataPoints; ++k){
					cov_c(j,k) = 0;
					cov_c(k,j) = 0;
				}
			}
		}
		f << "cov_c: \n" << cov_c << std::endl;
		cov.push_back(cov_c);
	}

	f << std::endl;
	f << std::endl;
	f << std::endl;
	//f << cov << std::endl;
	f << y.transpose() << std::endl;
	f.close();
	GaussianProcessMultiClass::magicFunc(amountOfClass,dataPoints, cov, y);
}

void executeForBinaryClass(const std::string& path){
	Data data;
	Labels labels;
	Data testData;
	Labels testLabels;
	const bool useRealData = true;
	std::map<std::string, Data > datas;
	if(useRealData){
		DataReader::readFromFiles(datas, "../realData/");
		int labelCounter = 0;
		const int amountOfElements = datas.begin()->second.size();
		const double fac = 0.10;
		for(std::map<std::string, Data >::iterator itData = datas.begin(); itData != datas.end(); ++itData){
			for(int i = 0; i < amountOfElements; ++i){
				if(i < fac * amountOfElements){
					// train data
					data.push_back(itData->second[i]);
					labels.push_back(labelCounter);
				}else{ //  if(i < (fac) * amountOfElements + 200)
					// test data
					testData.push_back(itData->second[i]);
					testLabels.push_back(labelCounter);
				}
			}
			++labelCounter;
		}
	}else{
		DataReader::readFromFile(data, labels, "../testData/trainInput.txt");
	}
	std::cout << "Data has dim: " << data[0].rows() << std::endl;
	std::cout << "Training size: " << data.size() << std::endl;
	// for binary case:
	if(useRealData && datas.size() == 2){
		const int firstPoints = 35;
		Eigen::VectorXd y;
		Eigen::MatrixXd dataMat;
		DataConverter::toRandDataMatrix(data, labels, dataMat, y, firstPoints);

		GaussianProcessBinary gp;
		gp.init(dataMat, y);
		bayesopt::Parameters par = initialize_parameters_to_default();
		par.noise = 1e-12;
		par.epsilon = 0.2;
		par.surr_name = "sGaussianProcessML";

		BayesOptimizer bayOpt(gp, par);
		vectord result(2);
		vectord lowerBound(2);
		lowerBound[0] = 0.1;
		lowerBound[1] = 0.1;
		vectord upperBound(2);
		upperBound[0] = gp.getLenMean();
		upperBound[1] = 1.3;
		bayOpt.setBoundingBox(lowerBound, upperBound);
		bayOpt.optimize(result);
		std::cout << RED << "Result: " << result[0] << ", "<< result[1] << RESET << std::endl;

		gp.getKernel().setHyperParams(result[0], result[1], 0.95);

		Eigen::VectorXd y2;
		Eigen::MatrixXd dataMat2;
		DataConverter::toRandDataMatrix(data, labels, dataMat2, y2, 10000000);
		std::cout << "Init with: " << data.size() << std::endl;
		gp.init(dataMat2, y2);
		gp.trainWithoutKernelOptimize();
/*
		const int dataPoints = data.size();
		Eigen::VectorXd y2(dataPoints);
		for(int i = 0; i < dataPoints; ++i){
			y2[i] = labels[i] != 0 ? 1 : -1; // just two classes left!
		}
		Eigen::MatrixXd dataMat2;

		DataConverter::toDataMatrix(dataRef, dataMat2, dataRef.size());
		gp.init(dataMat2, y2);
 */
		//gp.m_kernel.newRandHyperParams();
		//gp.m_kernel.setHyperParams(gp.m_kernel.len(),gp.m_kernel.sigmaF(),1);
		//gp.train();//WithoutKernelChange(dataMat2, y2); // train only the latent functions
		std::cout << "Start predicting for " << testData.size() << " points!" << std::endl;
		const Data& dataRef = testData;
		const Labels& labelsRef = testLabels;
		int wright = 0;
		int amountOfBelow = 0;
		int amountOfAbove = 0;
		std::cout << "Test" << std::endl;
		for(int j = dataRef.size() - 1; j >= 0 ; --j){
			double prob = gp.predict(dataRef[j]);
			if(prob > 0.5 && labelsRef[j] == 1){
				++wright;
			}else if(prob < 0.5 && labelsRef[j] == 0){
				++wright;
			}else{
				std::cout << "Prob: " << prob << ", label is: " << labelsRef[j] << std::endl;
			}
			if(prob > 0.5){
				++amountOfAbove;
			}else{
				++amountOfBelow;
			}
		}
		std::cout << RED;
		std::cout << "Amount of wright: " << (double) wright / dataRef.size() * 100.0 << "%" << std::endl;
		std::cout << "Amount of above: " << (double) amountOfAbove / dataRef.size() * 100.0 << "%" << std::endl;
		std::cout << "Amount of below: " << (double) amountOfBelow / dataRef.size() * 100.0 << "%" << std::endl;
		std::cout << "len: " << gp.getKernel().len() << ", sigmaF: " << gp.getKernel().sigmaF() <<std::endl;
		std::cout << RESET;

	}else{
		const int firstPoints = 10000000; // all points
		Eigen::VectorXd y;
		Eigen::MatrixXd dataMat;
		DataConverter::toRandDataMatrix(data, labels, dataMat, y, firstPoints);

		GaussianProcessBinary gp;
		gp.init(dataMat, y);
		bayesopt::Parameters par = initialize_parameters_to_default();
		std::cout << "noise: " << par.noise << std::endl;
 		//par.noise = 1-6;
		//par.init_method = 50;
		//par.n_iterations = 1000;
		par.noise = 1e-12;
		par.epsilon = 0.2;
		par.surr_name = "sGaussianProcessML";

		BayesOptimizer bayOpt(gp, par);

		vectord result(2);
		vectord lowerBound(2);
		lowerBound[0] = 0.1; // max(0.1,gp.getLenMean() - 2 * gp.getKernel().getLenVar());
		lowerBound[1] = 0.1;
		vectord upperBound(2);
		upperBound[0] = gp.getLenMean();// + gp.getKernel().getLenVar();
		upperBound[1] = 1.6;//max(0.5,gp.getKernel().getLenVar() / gp.getLenMean() * 0.5);

		bayOpt.setBoundingBox(lowerBound, upperBound);
		bayOpt.optimize(result);
		//gp.getKernel().setHyperParams(0.620284,0.55, 0.95);
		gp.getKernel().setHyperParams(result[0], result[1], 0.95);
		gp.trainWithoutKernelOptimize();
		std::cout << "Start predicting!" << std::endl;
		int wright = 0;
		int amountOfBelow = 0;
		int amountOfAbove = 0;
		for(int j = 0; j < data.size(); ++j){
			double prob = gp.predict(data[j]);
			std::cout << "Prob: " << prob << ", label is: " << labels[j] << std::endl;
			if(prob > 0.5 && labels[j] == 1){
				++wright;
			}else if(prob < 0.5 && labels[j] == 0){
				++wright;
			}
			if(prob > 0.5){
				++amountOfAbove;
			}else{
				++amountOfBelow;
			}
		}
		std::cout << RED;
		std::cout << "Amount of wright: " << (double) wright / data.size() * 100.0 << "%" << std::endl;
		std::cout << "Amount of above: " << (double) amountOfAbove / data.size() * 100.0 << "%" << std::endl;
		std::cout << "Amount of below: " << (double) amountOfBelow / data.size() * 100.0 << "%" << std::endl;
		std::cout << "len: " << gp.getKernel().len() << ", sigmaF: " << gp.getKernel().sigmaF() <<std::endl;
		std::cout << RESET;
		DataWriterForVisu::generateGrid("out2.txt", gp, 50, data);
	}
	//DataReader::readFromFile(data, labels, path);

	/*gp.setValues(dataMat, y);
	Eigen::VectorXd x(3);
	x(0) = 0.5; // length
	x(1) = 0.5; // sigmaF
	x(2) = 0.5; // sigmaN
	std::cout << "x: " << x << std::endl;

	OptimizeFunctor functor(&gp);
	Eigen::LevenbergMarquardt<OptimizeFunctor, double> lm(functor);
	printLine();
	lm.parameters.ftol = 1e-6;
	lm.parameters.xtol = 1e-6;
	lm.parameters.maxfev = 1000; // Max iterations
	int status = lm.minimize(x);
	std::cout << "LM status: " << status << std::endl;
	printLine();
	getchar();*/
/*
	const int dim = data[0].rows();
	Eigen::Vector2d dimVec;
	dimVec << 0,1;
	double min = 1000000;
	double max = -1000000;
	for(Data::const_iterator it = data.cbegin(); it != data.cend(); ++it){
		for(int i = 0; i < 2; ++i){
			int j = dimVec[i];
			if(min > (*it)[j]){
				min = (*it)[j];
			}
			if(max < (*it)[j]){
				max = (*it)[j];
			}
		}
	}
	const double diff = max - min;
	min -= diff * 0.2;
	max += diff * 0.2;
	std::cout << "min: " << min << std::endl;
	std::cout << "max: " << max << std::endl;
	const int amountOfPointsOnOneAxis = 50;
	const double stepSize = (1. / amountOfPointsOnOneAxis) * (max - min);
	std::ofstream file;
	file.open("visu.txt");
	Data points;
	points.reserve(amountOfPointsOnOneAxis * (amountOfPointsOnOneAxis + 1));
	int amount = 0;
	DoubleLabels dlabels;
	int k = 0;
	for(double xVal = max; xVal >= min; xVal -= stepSize){
		std::cout << "\rDone: " << k++ / (double) amountOfPointsOnOneAxis * 100.0 << "%           ";
		flush(std::cout);
		for(double yVal = min; yVal <= max; yVal+= stepSize){
			DataElement ele(dim);
			ele << xVal, yVal;
			points.push_back(ele);
			const double label = gp.predict(ele);
			dlabels.push_back(label);
			++amount;
		}
	}
	for(int i = 0; i < amount; ++i){
		if(i == 0){
			file << points[i][0] << " " << points[i][1] << " " << 0.0 << "\n";
		}else if(i == 1){
			file << points[i][0] << " " << points[i][1] << " " << 1.0 << "\n";
		}else{
			file << points[i][0] << " " << points[i][1] << " " << dlabels[i] << "\n";
		}
	}
	file.close();
	std::cout << "finish" << std::endl;
	//system("../PythonScripts/plotData2.py");
	 */
}

void executeForRFBinaryClass(){
	Data data;
	Data testData;
	Labels labels;
	Labels testLabels;
	std::map<std::string, Data > datas;
	DataReader::readFromFiles(datas, "../realData/");
	int labelCounter = 0;
	for(std::map<std::string, Data >::iterator itData = datas.begin(); itData != datas.end(); ++itData){
		const int amountOfElements = itData->second.size();

		for(int i = 0; i < amountOfElements; ++i){
			if(i < amountOfElements * 0.8){
				// train data
				data.push_back(itData->second[i]);
				labels.push_back(labelCounter);
			}else{
				// test data
				testData.push_back(itData->second[i]);
				testLabels.push_back(labelCounter);
			}
		}
		++labelCounter;
	}

	std::cout << "Training size: " << data.size() << std::endl;
	std::cout << "Test size:     " << testData.size() << std::endl;
	std::vector<int> heights(5);
	heights[0] = 2;
	heights[1] = 4;
	heights[2] = 6;
	heights[3] = 8;
	heights[4] = 12;
	std::vector<int> trees(5);
	trees[0] = 100;
	trees[1] = 1000;
	trees[2] = 10000;
	trees[3] = 100000;
	trees[4] = 1000000;

	for(int i = 0; i < heights.size(); ++i){
		int maxTree = trees.size();
		if(heights[i] > 10){
			maxTree = 3;
		}else if(heights[i] > 7){
			maxTree = 4;
		}
		for(int j = 0; j < maxTree; ++j){
			const int height = heights[i];
			const int amountOfTrees = trees[j];
			// for binary case:
			const int dataPoints = data.size();
			std::cout << "Amount of trees: " << amountOfTrees << " with height: " << height << std::endl;
			bool useFixedValuesForMinMaxUsedData;
			Settings::getValue("MinMaxUsedData.useFixedValuesForMinMaxUsedData", useFixedValuesForMinMaxUsedData);
			Eigen::Vector2i minMaxUsedData;
			if(useFixedValuesForMinMaxUsedData){
				int minVal = 0, maxVal = 0;
				Settings::getValue("MinMaxUsedData.minValue", minVal);
				Settings::getValue("MinMaxUsedData.maxValue", maxVal);
				minMaxUsedData << minVal, maxVal;
			}else{
				double minVal = 0, maxVal = 0;
				Settings::getValue("MinMaxUsedData.minValueFraction", minVal);
				Settings::getValue("MinMaxUsedData.maxValueFraction", maxVal);
				minMaxUsedData << (int) (minVal * data.size()),  (int) (maxVal * data.size());
			}
			std::cout << "Min max used data, min: " << minMaxUsedData[0] << " max: " << minMaxUsedData[1] << "\n";

			RandomForest forest(height, amountOfTrees, data[0].rows());
			forest.train(data, labels, data[0].rows(), minMaxUsedData);

			int right = 0;
			Labels predictedLabels;
			forest.predictData(testData, predictedLabels);
			for(int j = 0; j < testData.size(); ++j){
				if(testLabels[j] == predictedLabels[j]){
					++right;
				}
			}
			std::cout << RED << "Amount of right: " << (double) right / testData.size() * 100.0 << "%" << RESET << std::endl;
		}
	}
}

int main(){

	std::cout << "Start" << std::endl;
	// read in Settings
	Settings::init("../Settings/init.json");
	std::string path;
	Settings::getValue("RealData.folderPath", path);

	bool useGP;
	Settings::getValue("OnlyGp.useGP", useGP);
	if(useGP){
		executeForBinaryClass(path);
		return 0;
	}
	//executeForRFBinaryClass();
	//return 0;

	DataSets dataSets;
	DataReader::readFromFiles(dataSets, path);
	RandomForestGaussianProcess rfGp(dataSets, 4, 5);

//	DataWriterForVisu::generateGrid("out.txt", rfGp, 40, data, 0, 1);

	std::cout << "finish" << std::endl;
	return 0;
	Data data;
	Labels labels;
	DataReader::readFromFile(data, labels, path);
	bool useFixedValuesForMinMaxUsedData;
	Settings::getValue("MinMaxUsedData.useFixedValuesForMinMaxUsedData", useFixedValuesForMinMaxUsedData);
	Eigen::Vector2i minMaxUsedData;
	if(useFixedValuesForMinMaxUsedData){
		int minVal = 0, maxVal = 0;
		Settings::getValue("MinMaxUsedData.minValue", minVal);
		Settings::getValue("MinMaxUsedData.maxValue", maxVal);
		minMaxUsedData << minVal, maxVal;
	}else{
		double minVal = 0, maxVal = 0;
		Settings::getValue("MinMaxUsedData.minValueFraction", minVal);
		Settings::getValue("MinMaxUsedData.maxValueFraction", maxVal);
		minMaxUsedData << (int) (minVal * data.size()),  (int) (maxVal * data.size());
	}
	std::cout << "Min max used data, min: " << minMaxUsedData[0] << " max: " << minMaxUsedData[1] << "\n";

	Data testData;
	Labels testLabels;
	Settings::getValue("Test.path", path);
	DataReader::readFromFile(testData, testLabels, path);

	std::cout << "Finished reading" << std::endl;
	int dim = 2;
	if(data.size() > 0){
		dim = data[0].rows();
	}

	int height;
	int amountOfTrees;
	Settings::getValue("Forest.Trees.height", height, 7);
	Settings::getValue("Forest.amountOfTrees", amountOfTrees, 1000);
	std::cout << "Amount of trees: " << amountOfTrees << " with height: " << height << std::endl;

	RandomForest forest(height, amountOfTrees, dim);
	forest.train(data, labels, dim, minMaxUsedData);

	Labels guessedLabels;
	forest.predictData(testData, guessedLabels);

	int wrong = 0;
	for(int i = 0; i < testData.size(); ++i){
		if(guessedLabels[i] != testLabels[i]){
			++wrong;
		}
	}
	std::cout << "Other: Amount of test size: " << testData.size() << std::endl;
	std::cout << "Other: Amount of wrong: " << wrong / (double) testData.size() << std::endl;



	bool doWriting;
	Settings::getValue("WriteBinarySaveOfTrees.doWriting", doWriting, false);
	if(doWriting){
		std::string path;
		Settings::getValue("WriteBinarySaveOfTrees.path", path);
		RandomForestWriter::writeToFile(path, forest);
	}
	RandomForest newForest(0,0,0);
	RandomForestWriter::readFromFile("../testData/trees2.binary", newForest);
	Labels guessedLabels2;
	newForest.addForest(forest);
	newForest.predictData(testData, guessedLabels2);

	wrong = 0;
	for(int i = 0; i < testData.size(); ++i){
		if(guessedLabels2[i] != testLabels[i]){
			++wrong;
		}
	}
	std::cout << "Amount of combined trees: " << newForest.getNrOfTrees() << std::endl;
	std::cout << "Read: Amount of test size: " << testData.size() << std::endl;
	std::cout << "Read: Amount of wrong: " << wrong / (double) testData.size() << std::endl;


	int printX, printY;
	Settings::getValue("Write2D.doWriting", doWriting, false);
	Settings::getValue("Write2D.printX", printX, 0);
	Settings::getValue("Write2D.printY", printY, 1);
	if(doWriting){
		Settings::getValue("Write2D.gridPath", path);
		StopWatch sw;
		DataWriterForVisu::generateGrid(path, forest, 200, data, printX, printY);
		Settings::getValue("Write2D.testPath", path);
		DataWriterForVisu::writeData(path, testData, testLabels, printX, printY);
		std::cout << "Time for write: " << sw.elapsedSeconds() << std::endl;
		std::cout << "End Reached" << std::endl;
		system("../PythonScripts/plotData.py");
	}



	return 0;
}

