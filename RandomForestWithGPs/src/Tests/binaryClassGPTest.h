/*
 * binaryClassGPTest.h
 *
 *  Created on: 15.09.2016
 *      Author: Max
 */

#ifndef TESTS_BINARYCLASSGPTEST_H_
#define TESTS_BINARYCLASSGPTEST_H_

#include "../Data/DataReader.h"
#include "../Data/LabeledVectorX.h"
#include "../Data/DataConverter.h"
#include "../Data/DataWriterForVisu.h"
#include "../GaussianProcess/GaussianProcess.h"
#include "../GaussianProcess/GaussianProcessMultiBinary.h"
#include "../GaussianProcess/BayesOptimizer.h"
#include "../Base/Settings.h"
#include "../Utility/ConfusionMatrixPrinter.h"
#include "../Data/ClassKnowledge.h"
#include <chrono>
#include <thread>

void executeForBinaryClass(const bool useRealData){
	LabeledData data;
	LabeledData testData;
	DataSets datas;
	const int trainAmount = 500;
	const int testAmount = 200;
	if(useRealData){
		bool didNormalize = false;
		DataReader::readFromFiles(datas, "../realTest/", trainAmount + testAmount, false, didNormalize);
		if(!didNormalize){
			VectorX center, var;
			DataConverter::centerAndNormalizeData(datas, center, var);
		}
		std::cout << "Amount of datas: " << datas.size() << std::endl;
	}else{
		DataReader::readFromFile(data, "../testData/trainInput.txt", trainAmount);
		DataReader::readFromFile(testData, "../testData/testInput3.txt", testAmount);
	}
	// for binary case:
	if(useRealData && datas.size() == 2 && false){
		srand(2);
		int labelCounter = 0;
		for(DataSetsIterator itData = datas.begin(); itData != datas.end(); ++itData){
			const int amountOfElements = itData->second.size();
			std::cout << itData->first << " with: " << amountOfElements << " points"<< std::endl;
			for(int i = 0; i < amountOfElements; ++i){
				if(i < trainAmount){
					// train data
					data.emplace_back(itData->second[i]);
				}else if(i < trainAmount + testAmount){ //  if(i < (fac) * amountOfElements + 200)
					// test data
					testData.emplace_back(itData->second[i]);
				}
			}
			++labelCounter;
		}
		std::cout << "Training size: " << data.size() << std::endl;
		std::cout << "Data has dim: " << data[0]->rows() << std::endl;
		const int firstPoints = 35;
		VectorX y;
		Matrix dataMat;
		std::vector<int> classCounts(2,trainAmount);
		VectorX y3;
		Matrix dataMat3;
		std::vector<bool> usedElements;
		std::vector<bool> blockElements(data.size(), false);
		DataConverter::toRandClassAndHalfUniformDataMatrix(data, classCounts, dataMat, y, firstPoints, 0, usedElements, blockElements);
		//DataConverter::toRandDataMatrix(data, labels, dataMat, y, firstPoints);
		//std::cout << "y: " << y.transpose() << std::endl;
		GaussianProcess gp;

		gp.init(dataMat, y);

//		bayesopt::Parameters par = initialize_parameters_to_default();
//		par.noise = 1e-12;
//		par.epsilon = 0.2;
//		par.verbose_level = 3;
//		par.surr_name = "sGaussianProcessML";
//		BayesOptimizer bayOpt(gp, par);
//		vectord result(2);
//		vectord lowerBound(2);
//		lowerBound[0] = 30.0;
//		lowerBound[1] = 0.1;
//		vectord upperBound(2);
//		upperBound[0] = 150.0;
//		upperBound[1] = 1.3;
//		bayOpt.setBoundingBox(lowerBound, upperBound);
//		bayOpt.optimize(result);
//		std::cout << RED << "Result: " << result[0] << ", "<< result[1] << RESET << std::endl;
//		gp.getKernel().setHyperParams(result[0], result[1], 0.95);

		VectorX y2;
		Matrix dataMat2;
		//DataConverter::toRandDataMatrix(data, labels, dataMat2, y2, 400);
		std::vector<bool> usedElements2;
		std::vector<bool> blockElements2(data.size(), false);
		DataConverter::toRandClassAndHalfUniformDataMatrix(data, classCounts, dataMat2, y2, 400, 0, usedElements2, blockElements2);
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
		VectorX y2(dataPoints);
		for(int i = 0; i < dataPoints; ++i){
			y2[i] = labels[i] != 0 ? 1 : -1; // just two classes left!
		}
		Matrix dataMat2;

		DataConverter::toDataMatrix(dataRef, dataMat2, dataRef.size());
		gp.init(dataMat2, y2);
 */
		//gp.m_kernel.newRandHyperParams();
		//gp.m_kernel.setHyperParams(gp.m_kernel.len(),gp.m_kernel.sigmaF(),1);
		//gp.train();//WithoutKernelChange(dataMat2, y2); // train only the latent functions
		std::cout << "Start predicting for " << testData.size() << " points!" << std::endl;
		const LabeledData& dataRef = testData;
		int right = 0;
		int amountOfBelow = 0;
		int amountOfAbove = 0;
		std::cout << "Test" << std::endl;
		InLinePercentageFiller::instance().setActMax(dataRef.size());
		for(int j = 0; j < dataRef.size(); ++j){
			LabeledVectorX& ele = *dataRef[j];
			Real prob = gp.predict(ele);
			if(prob > 0.5 && ele.getLabel() == 0){
				++right;
			}else if(prob < 0.5 && ele.getLabel() == 1){
				++right;
			}else{
				//std::cout << "Prob: " << prob << ", label is: " << labelsRef[j] << std::endl;
			}
			if(prob > 0.5){
				++amountOfAbove;
			}else{
				++amountOfBelow;
			}
			InLinePercentageFiller::instance().setActValueAndPrintLine(j);
		}
		std::cout << RED;
		std::cout << "Amount of right: " << (Real) right / dataRef.size() * 100.0 << "%" << std::endl;
		std::cout << "Amount of above: " << (Real) amountOfAbove / dataRef.size() * 100.0 << "%" << std::endl;
		std::cout << "Amount of below: " << (Real) amountOfBelow / dataRef.size() * 100.0 << "%" << std::endl;
		std::cout << gp.getKernel().prettyString() << std::endl;
		std::cout << RESET;
		/*
		right = 0;
		amountOfAbove = 0;
		amountOfBelow = 0;
		for(int j = dataRef.size() - 1; j >= 0 ; --j){
			Real prob = testGp.predict(dataRef[j]);
			if(prob > 0.5 && labelsRef[j] == 1){
				++right;
			}else if(prob < 0.5 && labelsRef[j] == 0){
				++right;
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
		std::cout << "For loaded Gp amount of right: " << (Real) right / dataRef.size() * 100.0 << "%" << std::endl;
		std::cout << "For loaded Gp amount of above: " << (Real) amountOfAbove / dataRef.size() * 100.0 << "%" << std::endl;
		std::cout << "For loaded Gp amount of below: " << (Real) amountOfBelow / dataRef.size() * 100.0 << "%" << std::endl;
		std::cout << "For loaded Gp len: " << gp.getKernel().len() << ", sigmaF: " << gp.getKernel().sigmaF() <<std::endl;
		std::cout << RESET; */
	}else if(useRealData){
		std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(6);
		GaussianProcessMultiBinary gp(datas.size());
		LabeledData data;
		int labelCounter = 0;
		//const Real fac = 0.80;
		int trainAmount = 400;
		Settings::instance().getValue("MultiBinaryGP.trainingAmount", trainAmount);
		int testAmount = 100;
		Settings::instance().getValue("MultiBinaryGP.testingAmount", testAmount);
		for(DataSetsIterator itData = datas.begin(); itData != datas.end(); ++itData){
			const int amountOfElements = itData->second.size();
			std::cout << itData->first << " with: " << amountOfElements << " points"<< std::endl;
			ClassKnowledge::instance().setNameFor(itData->first, labelCounter);
			for(int i = 0; i < amountOfElements; ++i){
				if(i < trainAmount){
					// train data
					data.emplace_back(itData->second[i]);
				}else if(i < trainAmount + testAmount){ //  if(i < (fac) * amountOfElements + 200)
					// test data
					testData.emplace_back(itData->second[i]);
				}
			}
			++labelCounter;
		}
		std::cout << "Data has dim: " << data[0]->rows() << std::endl;
		std::cout << "Training size: " << data.size() << std::endl;
		gp.train(data);
		/*
				const int dataPoints = data.size();
				VectorX y2(dataPoints);
				for(int i = 0; i < dataPoints; ++i){
					y2[i] = labels[i] != 0 ? 1 : -1; // just two classes left!
				}
				Matrix dataMat2;
				DataConverter::toDataMatrix(dataRef, dataMat2, dataRef.size());
				gp.init(dataMat2, y2);
		 */
		//gp.m_kernel.newRandHyperParams();
		//gp.m_kernel.setHyperParams(gp.m_kernel.len(),gp.m_kernel.sigmaF(),1);
		//gp.train();//WithoutKernelChange(dataMat2, y2); // train only the latent functions
		const LabeledData& dataRef = testData;
		std::cout << "Start predicting for " << dataRef.size() << " points!" << std::endl;
		int right = 0;
		std::cout << "Test" << std::endl;
		std::vector<Real> prob;
		Matrix confusion = Matrix::Zero(datas.size(), datas.size());
		int unknownCounter = 0;
		InLinePercentageFiller::instance().setActMax(dataRef.size());
		for(int j = 0; j < (int) dataRef.size(); ++j){
			LabeledVectorX& ele = *dataRef[j];
			const unsigned int label = gp.predict(ele, prob);
			if(ele.getLabel() == label){
				++right;
			}
			if(label != UNDEF_CLASS_LABEL){
				confusion(ele.getLabel() ,label) += 1;
			}else{
				++unknownCounter;
			}
			InLinePercentageFiller::instance().setActValueAndPrintLine(j);
		}
		std::cout << RED;
		std::cout << "Amount of right:  " << (Real) right / dataRef.size() * 100.0 << "%" << std::endl;
		std::cout << "Amount of unknown: " << (Real) unknownCounter / dataRef.size() * 100.0 << "%" << std::endl;
		std::cout << RESET;
		ConfusionMatrixPrinter::print(confusion);
		if(datas.size() == 2){
			DataWriterForVisu::writeSvg("out.svg", &gp, data);
			openFileInViewer("out.svg");
		}
	}else{
		const int firstPoints = 10000000; // all points
		VectorX y;
		Matrix dataMat;
		DataConverter::toDataMatrix(data, dataMat, y, firstPoints);
		GaussianProcess gp;
		gp.init(dataMat, y);
		if(false){
			Real up1 = 1.8;
			Real up2 = 1.8;
			Real step = 0.025;
			int size1 = (up1 - 0.005) / step + 1;
			int size2 = (up2 - 0.) / step + 1;
			Matrix mat = Matrix::Zero(size1, size2);
			int i = 0,j,k = 0;
			Real minX, minY, minZ;
			Real minVal = NEG_REAL_MAX;
			InLinePercentageFiller::instance().setActMax(size1 * size2);
			//for(Real z = -1.; z < 1.0; z += 0.01){
			{	Real z = 0.1;
				i = 0;
				for(Real x = 0.005; x < up1; x += step){
				//{ Real x = 0.455;
					j = 0;
					for(Real y = 0.; y < up2; y += step){
					//{
						gp.getKernel().setHyperParams(x, y, z);
						Real val;
						gp.trainBayOpt(val,1);
						if(val < 10){
							if(minVal < val){
								minVal = val;
								minX = x;
								minY = y;
								minZ = z;
							}
							mat(i,j) = val;
						}else{
							mat(i,j) = NEG_REAL_MAX;
						}
					//	std::cout << "z: " << z << ", val: " << val << std::endl;
						std::cout << "x: " << x << ", y: " << y << std::endl;
						++j;
					}
					//InLinePercentageFiller::instance().setActValueAndPrintLine(k * size * size + i * size + j);
					++i;
				}
				++k;
			}
			std::cout << "X: " << minX << ", y: " << minY << ", z: " << minZ << ", for: " << minVal << std::endl;
			DataWriterForVisu::writeSvg("out2.svg", mat);
			openFileInViewer("out2.svg");
			return;
		}

		if(false){
//			bayesopt::Parameters par = initialize_parameters_to_default();
//			std::cout << "noise: " << par.noise << std::endl;
//			//par.noise = 1-6;
//			par.init_method = 300;
//			//par.n_iterations = 500;
//			par.noise = 1e-5;
//			par.epsilon = 0.2;
//			par.verbose_level = 6;
//			par.surr_name = "sGaussianProcessML";
//
//			BayesOptimizer bayOpt(gp, par);
//
//			vectord result(2);
//			vectord lowerBound(2);
//			lowerBound[0] = 0.0005; // max(0.1,gp.getLenMean() - 2 * gp.getKernel().getLenVar());
//			lowerBound[1] = 0.20;
//			vectord upperBound(2);
//			upperBound[0] = 1.75;// + gp.getKernel().getLenVar();
//			upperBound[1] = 1.6;//max(0.5,gp.getKernel().getLenVar() / gp.getLenMean() * 0.5);
//
//			bayOpt.setBoundingBox(lowerBound, upperBound);
//			bayOpt.optimize(result);
//			gp.getKernel().setHyperParams(result[0], result[1], EPSILON);
//			std::cout << gp.getKernel().prettyString() << std::endl;
		}
		gp.getKernel().setHyperParams(0.6,0.4, 0.1);
		StopWatch sw;
		gp.trainWithoutKernelOptimize();
		std::cout << "For GP training: " << sw.elapsedAsTimeFrame() << std::endl;
		std::cout << "Start predicting!" << std::endl;
		int right = 0;
		int amountOfBelow = 0;
		int amountOfAbove = 0;
		for(int j = 0; j < (int) data.size(); ++j){
			gp.resetFastPredict();
			LabeledVectorX& ele = *data[j];
			Real prob = gp.predict(ele, 100000);
			std::cout << "Prob: " << prob << ", label is: " << ele.getLabel() << std::endl;
			if(prob > 0.5 && ele.getLabel() == 0){
				++right;
			}else if(prob < 0.5 && ele.getLabel() == 1){
				++right;
			}
			if(prob > 0.5){
				++amountOfAbove;
			}else{
				++amountOfBelow;
			}
		}
		std::cout << RED;
		std::cout << "Amount of right: " << (Real) right / data.size() * 100.0 << "%" << std::endl;
		std::cout << "Amount of above: " << (Real) amountOfAbove / data.size() * 100.0 << "%" << std::endl;
		std::cout << "Amount of below: " << (Real) amountOfBelow / data.size() * 100.0 << "%" << std::endl;
		std::cout << gp.getKernel().prettyString() << std::endl;
		std::cout << RESET;
		DataWriterForVisu::writeSvg("out.svg", &gp, data);
		openFileInViewer("out.svg");
	}
	//DataReader::readFromFile(data, labels, path);

	/*gp.setValues(dataMat, y);
	VectorX x(3);
	x(0) = 0.5; // length
	x(1) = 0.5; // sigmaF
	x(2) = 0.5; // sigmaN
	std::cout << "x: " << x << std::endl;

	OptimizeFunctor functor(&gp);
	Eigen::LevenbergMarquardt<OptimizeFunctor, Real> lm(functor);
	printLine();
	lm.parameters.ftol = EPSILON;
	lm.parameters.xtol = EPSILON;
	lm.parameters.maxfev = 1000; // Max iterations
	int status = lm.minimize(x);
	std::cout << "LM status: " << status << std::endl;
	printLine();
	getchar();*/
/*
	const int dim = data[0].rows();
	Vector2d dimVec;
	dimVec << 0,1;
	Real min = 1000000;
	Real max = -1000000;
	for(LabeledDataConstIterator it = data.cbegin(); it != data.cend(); ++it){
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
	const Real diff = max - min;
	min -= diff * 0.2;
	max += diff * 0.2;
	std::cout << "min: " << min << std::endl;
	std::cout << "max: " << max << std::endl;
	const int amountOfPointsOnOneAxis = 50;
	const Real stepSize = (1. / amountOfPointsOnOneAxis) * (max - min);
	std::ofstream file;
	file.open("visu.txt");
	Data points;
	points.reserve(amountOfPointsOnOneAxis * (amountOfPointsOnOneAxis + 1));
	int amount = 0;
	DoubleLabels dlabels;
	int k = 0;
	for(Real xVal = max; xVal >= min; xVal -= stepSize){
		std::cout << "\rDone: " << k++ / (Real) amountOfPointsOnOneAxis * 100.0 << "%           ";
		flush(std::cout);
		for(Real yVal = min; yVal <= max; yVal+= stepSize){
			DataElement ele(dim);
			ele << xVal, yVal;
			points.emplace_back(ele);
			const Real label = gp.predict(ele);
			dlabels.emplace_back(label);
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
