/*
 * binaryClassGPTest.h
 *
 *  Created on: 15.09.2016
 *      Author: Max
 */

#ifndef TESTS_BINARYCLASSGPTEST_H_
#define TESTS_BINARYCLASSGPTEST_H_

#include <Eigen/Dense>
#include "../Data/DataReader.h"
#include "../Data/DataConverter.h"
#include "../Data/DataWriterForVisu.h"
#include "../GaussianProcess/GaussianProcess.h"
#include "../GaussianProcess/GaussianProcessMultiBinary.h"
#include "../GaussianProcess/BayesOptimizer.h"
#include "../Utility/Settings.h"
#include "../Utility/ConfusionMatrixPrinter.h"


void executeForBinaryClass(const std::string& path, const bool useRealData){
	Data data;
	Labels labels;
	Data testData;
	Labels testLabels;
	std::map<std::string, Data > datas;
	const int trainAmount = 500;
	const int testAmount = 200;
	if(useRealData){
		DataReader::readFromFiles(datas, "../realTest/", trainAmount + testAmount);
		std::cout << "Amount of datas: " << datas.size() << std::endl;
	}else{
		DataReader::readFromFile(data, labels, "../testData/trainInput.txt", trainAmount + testAmount);
	}
	// for binary case:
	if(useRealData && datas.size() == 2 && false){
		srand(2);
		int labelCounter = 0;
		for(std::map<std::string, Data >::iterator itData = datas.begin(); itData != datas.end(); ++itData){
			const int amountOfElements = itData->second.size();
			std::cout << itData->first << " with: " << amountOfElements << " points"<< std::endl;
			for(int i = 0; i < amountOfElements; ++i){
				if(i < trainAmount){
					// train data
					data.push_back(itData->second[i]);
					labels.push_back(labelCounter);
				}else if(i < trainAmount + testAmount){ //  if(i < (fac) * amountOfElements + 200)
					// test data
					testData.push_back(itData->second[i]);
					testLabels.push_back(labelCounter);
				}
			}
			++labelCounter;
		}
		std::cout << "Training size: " << data.size() << std::endl;
		std::cout << "Data has dim: " << data[0].rows() << std::endl;
		const int firstPoints = 35;
		Eigen::VectorXd y;
		Eigen::MatrixXd dataMat;
		std::vector<int> classCounts(2,trainAmount);
		Eigen::VectorXd y3;
		Eigen::MatrixXd dataMat3;
		std::vector<bool> usedElements;
		std::vector<bool> blockElements(data.size(), false);
		DataConverter::toRandClassAndHalfUniformDataMatrix(data, labels, classCounts, dataMat, y, firstPoints, 0, usedElements, blockElements);
		//DataConverter::toRandDataMatrix(data, labels, dataMat, y, firstPoints);
		//std::cout << "y: " << y.transpose() << std::endl;
		GaussianProcess gp;

		gp.init(dataMat, y);

		bayesopt::Parameters par = initialize_parameters_to_default();
		par.noise = 1e-12;
		par.epsilon = 0.2;
		par.verbose_level = 3;
		par.surr_name = "sGaussianProcessML";
		BayesOptimizer bayOpt(gp, par);
		vectord result(2);
		vectord lowerBound(2);
		lowerBound[0] = 30.0;
		lowerBound[1] = 0.1;
		vectord upperBound(2);
		upperBound[0] = 150.0;
		upperBound[1] = 1.3;
		bayOpt.setBoundingBox(lowerBound, upperBound);
		bayOpt.optimize(result);
		std::cout << RED << "Result: " << result[0] << ", "<< result[1] << RESET << std::endl;
		gp.getKernel().setHyperParams(result[0], result[1], 0.95);

		Eigen::VectorXd y2;
		Eigen::MatrixXd dataMat2;
		//DataConverter::toRandDataMatrix(data, labels, dataMat2, y2, 400);
		std::vector<bool> usedElements2;
		std::vector<bool> blockElements2(data.size(), false);
		DataConverter::toRandClassAndHalfUniformDataMatrix(data, labels, classCounts, dataMat2, y2, 400, 0, usedElements2, blockElements2);
		std::cout << "Init with: " << dataMat2.cols() << std::endl;
		gp.init(dataMat2, y2);
		std::cout << "After init"<< std::endl;

		gp.trainWithoutKernelOptimize();

		/*
		GaussianProcessWriter::writeToFile("gp.bgp", gp);

		GaussianProcess testGp;
		GaussianProcessWriter::readFromFile("gp.bgp", testGp);
		 */
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
		InLinePercentageFiller::setActMax(dataRef.size());
		for(int j = 0; j < dataRef.size(); ++j){
			double prob = gp.predict(dataRef[j]);
			if(prob > 0.5 && labelsRef[j] == 0){
				++wright;
			}else if(prob < 0.5 && labelsRef[j] == 1){
				++wright;
			}else{
				//std::cout << "Prob: " << prob << ", label is: " << labelsRef[j] << std::endl;
			}
			if(prob > 0.5){
				++amountOfAbove;
			}else{
				++amountOfBelow;
			}
			InLinePercentageFiller::setActValueAndPrintLine(j);
		}
		std::cout << RED;
		std::cout << "Amount of wright: " << (double) wright / dataRef.size() * 100.0 << "%" << std::endl;
		std::cout << "Amount of above: " << (double) amountOfAbove / dataRef.size() * 100.0 << "%" << std::endl;
		std::cout << "Amount of below: " << (double) amountOfBelow / dataRef.size() * 100.0 << "%" << std::endl;
		std::cout << "len: " << gp.getKernel().len() << ", sigmaF: " << gp.getKernel().sigmaF() <<std::endl;
		std::cout << RESET;
		/*
		wright = 0;
		amountOfAbove = 0;
		amountOfBelow = 0;
		for(int j = dataRef.size() - 1; j >= 0 ; --j){
			double prob = testGp.predict(dataRef[j]);
			if(prob > 0.5 && labelsRef[j] == 1){
				++wright;
			}else if(prob < 0.5 && labelsRef[j] == 0){
				++wright;
			}else{
				//std::cout << "Prob: " << prob << ", label is: " << labelsRef[j] << std::endl;
			}
			if(prob > 0.5){
				++amountOfAbove;
			}else{
				++amountOfBelow;
			}
		}
		std::cout << RED;
		std::cout << "For loaded Gp amount of wright: " << (double) wright / dataRef.size() * 100.0 << "%" << std::endl;
		std::cout << "For loaded Gp amount of above: " << (double) amountOfAbove / dataRef.size() * 100.0 << "%" << std::endl;
		std::cout << "For loaded Gp amount of below: " << (double) amountOfBelow / dataRef.size() * 100.0 << "%" << std::endl;
		std::cout << "For loaded Gp len: " << gp.getKernel().len() << ", sigmaF: " << gp.getKernel().sigmaF() <<std::endl;
		std::cout << RESET; */
	}else if(useRealData){
		std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(6);
		GaussianProcessMultiBinary gp(datas.size());
		DataContainer container;
		container.namesOfClasses.resize(datas.size());
		int labelCounter = 0;
		//const double fac = 0.80;
		int trainAmount = 400;
		Settings::getValue("MultiBinaryGP.trainingAmount", trainAmount);
		int testAmount = 100;
		Settings::getValue("MultiBinaryGP.testingAmount", testAmount);
		for(std::map<std::string, Data >::iterator itData = datas.begin(); itData != datas.end(); ++itData){
			const int amountOfElements = itData->second.size();
			std::cout << itData->first << " with: " << amountOfElements << " points"<< std::endl;
			container.namesOfClasses[labelCounter] = itData->first;
			for(int i = 0; i < amountOfElements; ++i){
				if(i < trainAmount){
					// train data
					container.data.push_back(itData->second[i]);
					container.labels.push_back(labelCounter);
				}else if(i < trainAmount + testAmount){ //  if(i < (fac) * amountOfElements + 200)
					// test data
					testData.push_back(itData->second[i]);
					testLabels.push_back(labelCounter);
				}
			}
			++labelCounter;
		}
		container.amountOfPoints = container.data.size();
		std::cout << "Data has dim: " << container.data[0].rows() << std::endl;
		std::cout << "Training size: " << container.amountOfPoints << std::endl;
		gp.train(container);
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
		const Data& dataRef = testData;
		std::cout << "Start predicting for " << dataRef.size() << " points!" << std::endl;
		const Labels& labelsRef = testLabels;
		int wright = 0;
		std::cout << "Test" << std::endl;
		std::vector<double> prob;
		Eigen::MatrixXd confusion = Eigen::MatrixXd::Zero(datas.size(), datas.size());
		int unknownCounter = 0;
		InLinePercentageFiller::setActMax(dataRef.size());
		for(int j = 0; j < dataRef.size(); ++j){
			const int label = gp.predict(dataRef[j], prob);
			if(labelsRef[j] == label){
				++wright;
			}
			if(label != -1){
				confusion(labelsRef[j],label) += 1;
			}else{
				++unknownCounter;
			}
			InLinePercentageFiller::setActValueAndPrintLine(j);
		}
		std::cout << RED;
		std::cout << "Amount of wright:  " << (double) wright / dataRef.size() * 100.0 << "%" << std::endl;
		std::cout << "Amount of unknown: " << (double) unknownCounter / dataRef.size() * 100.0 << "%" << std::endl;
		std::cout << RESET;
		ConfusionMatrixPrinter::print(confusion, container.namesOfClasses);
		if(datas.size() == 2){
			DataWriterForVisu::writeSvg("out.svg", gp, 75, container.data);
			system("open out.svg &");
		}
	}else{

		const int firstPoints = 10000000; // all points
		Eigen::VectorXd y;
		Eigen::MatrixXd dataMat;
		DataConverter::toRandDataMatrix(data, labels, dataMat, y, firstPoints);

		GaussianProcess gp;
		gp.init(dataMat, y);
		if(false){
			double up = 3;
			double step = 0.05;
			int size = (up - 0.005) / step + 1;
			Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(size, size);
			int i = 0,j;
			double minX, minY;
			double minVal = -DBL_MAX;
			InLinePercentageFiller::setActMax(size * size);
			for(double x = 0.005; x < up; x += step){
				j = 0;
				for(double y = 0.005; y < up; y += step){
					gp.getKernel().setHyperParams(x, y, gp.getKernel().sigmaN());
					double val;
					gp.trainBayOpt(val,1);
					if(val < 10){
						if(minVal < val){
							minVal = val;
							minX = x;
							minY = y;
						}
						mat(i,j) = val;
					}else{
						mat(i,j) = -DBL_MAX;
					}
					++j;
				}
				InLinePercentageFiller::setActValueAndPrintLine(i * size + j);
				++i;
			}
			std::cout << "X: " << minX << ", y: " << minY << ", for: " << minVal << std::endl;
			DataWriterForVisu::writeSvg("out2.svg", mat);
			system("open out2.svg &");
			return;
		}

		bayesopt::Parameters par = initialize_parameters_to_default();
		std::cout << "noise: " << par.noise << std::endl;
 		//par.noise = 1-6;
		par.init_method = 300;
		//par.n_iterations = 500;
		par.noise = 1e-5;
		par.epsilon = 0.2;
		par.verbose_level = 6;
		par.surr_name = "sGaussianProcessML";

		BayesOptimizer bayOpt(gp, par);

		vectord result(2);
		vectord lowerBound(2);
		lowerBound[0] = 0.0005; // max(0.1,gp.getLenMean() - 2 * gp.getKernel().getLenVar());
		lowerBound[1] = 0.20;
		vectord upperBound(2);
		upperBound[0] = 5.75;// + gp.getKernel().getLenVar();
		upperBound[1] = 5.6;//max(0.5,gp.getKernel().getLenVar() / gp.getLenMean() * 0.5);

		bayOpt.setBoundingBox(lowerBound, upperBound);
		bayOpt.optimize(result);
		gp.getKernel().setHyperParams(result[0], result[1], 0.95);
		//gp.getKernel().setHyperParams(0.620284,0.55, 0.95);
		gp.trainWithoutKernelOptimize();
		std::cout << "Start predicting!" << std::endl;
		int wright = 0;
		int amountOfBelow = 0;
		int amountOfAbove = 0;
		for(int j = 0; j < data.size(); ++j){
			double prob = gp.predict(data[j]);
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
		std::cout << "Amount of wright: " << (double) wright / data.size() * 100.0 << "%" << std::endl;
		std::cout << "Amount of above: " << (double) amountOfAbove / data.size() * 100.0 << "%" << std::endl;
		std::cout << "Amount of below: " << (double) amountOfBelow / data.size() * 100.0 << "%" << std::endl;
		std::cout << "len: " << gp.getKernel().len() << ", sigmaF: " << gp.getKernel().sigmaF() <<std::endl;
		std::cout << RESET;
		DataWriterForVisu::writeSvg("out.svg", gp, 100, data);
		system("open out.svg &");
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



#endif /* TESTS_BINARYCLASSGPTEST_H_ */
