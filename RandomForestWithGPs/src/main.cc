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
	Data testData;
	Labels labels;
	Labels testLabels;
	std::map<std::string, Data > datas;
	DataReader::readFromFiles(datas, "../realData/");
	int labelCounter = 0;
	const int amountOfElements = datas.begin()->second.size();
	const double fac = 0.015;
	for(std::map<std::string, Data >::iterator itData = datas.begin(); itData != datas.end(); ++itData){
		for(int i = 0; i < amountOfElements; ++i){
			if(i < fac * amountOfElements){
				// train data
				data.push_back(itData->second[i]);
				labels.push_back(labelCounter);
			}else if(i < (fac) * amountOfElements + 50){
				// test data
				testData.push_back(itData->second[i]);
				testLabels.push_back(labelCounter);
			}
		}
		++labelCounter;
	}
	std::cout << "Data has dim: " << data[0].rows() << std::endl;
	std::cout << "Training size: " << data.size() << std::endl;
	// for binary case:
	if(datas.size() == 2){

		const int firstPoints = 35;
		Eigen::VectorXd y;
		Eigen::MatrixXd dataMat;
		DataConverter::toRandDataMatrix(data, labels, dataMat, y, firstPoints);

		GaussianProcessBinary gp;
		gp.init(dataMat, y);
		{
			bayesopt::Parameters par = initialize_parameters_to_default();

			par.kernel.name = "kSum(kSEISO,kConst)";
			par.kernel.hp_mean <<= 1.0;
			par.kernel.hp_std <<= 1.0;

			par.mean.name = "mConst";
			par.mean.coef_mean <<= gp.getLenMean();
			par.mean.coef_std <<= gp.getKernel().getLenVar();


			par.surr_name = "sStudentTProcessJef";
			par.noise = 1e-10;

			par.sc_type = SC_MAP;
			par.l_type = L_EMPIRICAL;

			par.n_iterations = 100;    // Number of iterations
			par.random_seed = 0;
			par.n_init_samples = 15;
			par.n_iter_relearn = 0;

			BayesOptimizer bayOpt(gp, par);
			vectord result(2);
			bayOpt.optimize(result);
			std::cout << RED << "Result: " << result[0] << ", "<< result[1] << RESET << std::endl;
			return;

		}

		//gp.train(); // train the kernel params

		const int dataPoints = data.size();
		Eigen::VectorXd y2(dataPoints);
		for(int i = 0; i < dataPoints; ++i){
			y2[i] = labels[i] != 0 ? 1 : -1; // just two classes left!
		}
		const Data& dataRef = data;
		Eigen::MatrixXd dataMat2;

		DataConverter::toDataMatrix(dataRef, dataMat2, dataRef.size());
		gp.init(dataMat2, y2);
		for(int i = 0; i < 100; ++i){
			//gp.m_kernel.newRandHyperParams();
			//gp.m_kernel.setHyperParams(gp.m_kernel.len(),gp.m_kernel.sigmaF(),1);
			gp.train();//WithoutKernelChange(dataMat2, y2); // train only the latent functions
			std::cout << "Start predicting!" << std::endl;
			int wright = 0;
			int amountOfBelow = 0;
			int amountOfAbove = 0;
			for(int j = 0; j < dataRef.size(); ++j){
				double prob = gp.predict(dataRef[j]);
				std::cout << "Prob: " << prob << ", label is: " << labels[j] << std::endl;
				if(prob > 0.5 && labels[j] == 0){
					++wright;
				}else if(prob < 0.5 && labels[j] == 1){
					++wright;
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
		}
	}else{
		printError("Implement me!");
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

int main(){


	std::cout << "Start" << std::endl;
	// read in Settings
	Settings::init("../Settings/init.json");
	std::string path;
	Settings::getValue("RealData.folderPath", path);

	executeForBinaryClass(path);
	//executeForBinaryClass(path);
	return 0;
	Data data;
	Labels labels;
	DataReader::readFromFile(data, labels, path);
	RandomForestGaussianProcess rfGp(data, labels, 4, 500, 2);

	DataWriterForVisu::generateGrid("out.txt", rfGp, 40, data, 0, 1);

	std::cout << "finish" << std::endl;
	return 0;
	/*
	Data data;
	Labels labels;
	DataReader::readFromFile(data, labels, path);
*/
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

