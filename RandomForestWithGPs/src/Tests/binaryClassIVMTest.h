/*
 * binaryClassIVMTest.h
 *
 *  Created on: 04.10.2016
 *      Author: Max
 */

#ifndef TESTS_BINARYCLASSIVMTEST_H_
#define TESTS_BINARYCLASSIVMTEST_H_

#include <Eigen/Dense>
#include "../Data/DataReader.h"
#include "../Data/DataConverter.h"
#include "../Data/DataWriterForVisu.h"
#include "../GaussianProcess/IVM.h"
#include "../GaussianProcess/BayesOptimizerIVM.h"
#include "../Base/Settings.h"
#include "../Utility/ConfusionMatrixPrinter.h"
#include <chrono>
#include <thread>

void testIvm(IVM& ivm, const Eigen::MatrixXd& data, const Eigen::VectorXd& y){
	if(y.rows() != data.cols()){
		printError("The data does not correspond to the labels!");
		return;
	}
	int right = 0;
	int amountOfBelow = 0;
	int amountOfAbove = 0;
	const int amountOfTestPoints = data.cols();
	std::list<double> probs;
	for(int i = 0; i < amountOfTestPoints; ++i){
		double prob = ivm.predict(data.col(i));
		if(prob > 0.5 && y[i] == 1){
			++right;
		}else if(prob < 0.5 && y[i] == -1){
			++right;
		}
		if(prob > 0.5){
			++amountOfAbove;
		}else if(prob < 0.5){
			++amountOfBelow;
		}
		probs.push_back(prob);
	}
	if(amountOfTestPoints > 0){
		DataWriterForVisu::writeHisto("histo.svg", probs, 14);
		openFileInViewer("histo.svg");
	}
	std::cout << RED;
	std::cout << "Amount of right: " << (double) right / amountOfTestPoints * 100.0 << "%" << std::endl;
	std::cout << "Amount of above: " << (double) amountOfAbove / amountOfTestPoints * 100.0 << "%" << std::endl;
	std::cout << "Amount of below: " << (double) amountOfBelow / amountOfTestPoints * 100.0 << "%" << std::endl;
	std::cout << "len: " << ivm.getKernel().len() << ", sigmaF: " << ivm.getKernel().sigmaF() <<std::endl;
	std::cout << RESET;
}

void executeForBinaryClassIVM(const std::string& path, const bool useRealData, const bool visu){
	ClassData data;
	ClassData testData;
	DataSets datas;
	const int trainAmount = 500;
	const int testAmount = 200;
	const int firstPoints = 10000000; // all points
	Eigen::VectorXd y;
	Eigen::MatrixXd dataMat;
	Eigen::VectorXd yTest;
	Eigen::MatrixXd dataMatTest;
	bool doEpUpdate;
	Settings::getValue("IVM.doEpUpdate", doEpUpdate);
	const double sNoise = 0.0;
	if(useRealData){
		DataReader::readFromFiles(datas, "../realTest/", trainAmount + testAmount);
		std::cout << "Amount of datas: " << datas.size() << std::endl;
		DataPoint center, var;
		DataConverter::centerAndNormalizeData(datas, center, var);
		DataConverter::toDataMatrix(datas, dataMat, y, dataMatTest, yTest, trainAmount);
	}else{
		DataReader::readFromFile(data, "../testData/trainInput.txt", 1000000);
		DataReader::readFromFile(testData, "../testData/testInput3.txt", 1000000);
		DataPoint center, var;
		DataConverter::centerAndNormalizeData(data, center, var);
		DataConverter::centerAndNormalizeData(testData, center, var);
		DataConverter::toDataMatrix(data, dataMat, y, firstPoints);
		DataConverter::toDataMatrix(testData, dataMatTest, yTest, firstPoints);
	}
	std::cout << "Finish reading " << std::endl;
	if(false){
		double length = 23, noise = 4, number = 30;
		if(!useRealData){
			length = 0.8;
			noise = 1;
			number = 14;
		}
		if(false){
			IVM ivm;
			ivm.init(dataMat, y, number, doEpUpdate);
			ivm.setDerivAndLogZFlag(true, false);
			std::list<double> list;
			Eigen::VectorXd hyperparams(2);
			Eigen::VectorXd bestHyperparams(2);
			const int randStartGen = StopWatch::getActTime();
			srand(randStartGen);
			hyperparams[1] = 2.0;
			double bestLogZ = -DBL_MAX;
			StopWatch swTry;
			for(int i = 0; i < 500; ++i){
				hyperparams[0] = rand() / (double) RAND_MAX * length + 0.03;
				hyperparams[1] = rand() / (double) RAND_MAX * noise + 0.01;
				std::cout << "Try " << hyperparams[0] << ", " << hyperparams[1] << ", logZ: " << ivm.m_logZ
						<< ", best: " << bestLogZ << ", for: " << bestHyperparams[0] << ", " << bestHyperparams[1] << std::endl;
				ivm.getKernel().setHyperParams((double)hyperparams[0], (double) hyperparams[1], sNoise);
				StopWatch sw;
				const bool train = ivm.train();
				std::cout << "Time for training: " << sw.elapsedAsPrettyTime() << std::endl;
				/*if(!train){
					getchar();
				}*/
				if(bestLogZ < ivm.m_logZ && train){
					bestLogZ = ivm.m_logZ;
					bestHyperparams[0] = hyperparams[0];
					bestHyperparams[1] = hyperparams[1];
				}
			}
			const TimeFrame timeTry = swTry.elapsedAsTimeFrame();
			StopWatch swGrad;
			hyperparams[0] = bestHyperparams[0];
			hyperparams[1] = bestHyperparams[1];
			ivm.setDerivAndLogZFlag(true, true);
			ivm.getKernel().setHyperParams((double)bestHyperparams[0], (double) bestHyperparams[1], sNoise);
			bool t = ivm.train();
			if(!useRealData && t){
				DataWriterForVisu::writeSvg("before.svg", ivm, ivm.getSelectedInducingPoints(), y, 50, data);
				system("open before.svg");
			}
			double fac = 0.0001;
			double smallestLog = -DBL_MAX;
			for(int i = 0; i < 10000; ++i){
				std::cout << "Best " << hyperparams[0] << ", " << hyperparams[1] << ", logZ: " << ivm.m_logZ << ", deriv: "
						<< ivm.m_derivLogZ[0] << ", " << ivm.m_derivLogZ[1] << ", fac: " << fac << std::endl;
				//DataWriterForVisu::writeSvg("logZValues.svg", bayOpt.m_logZValues, true);
				//system("open logZValues.svg");
				ivm.getKernel().setHyperParams((double)hyperparams[0], (double) hyperparams[1], sNoise);
				StopWatch sw;
				bool train = ivm.train(i % 10 == 0, 1);
				if(train && !isnan(ivm.m_logZ)){
					std::cout << "Time for training: " << sw.elapsedAsPrettyTime() << std::endl;
					/*int right = 0;
				int amountOfBelow = 0;
				int amountOfAbove = 0;
				for(int i = 0; i < dataMatTest.cols(); ++i){
					double prob = ivm.predict(dataMatTest.col(i));
					if(prob < 0.5 && yTest[i] == 1){
						++right;
					}else if(prob > 0.5 && yTest[i] == -1){
						++right;
					}
					if(prob > 0.5){
						++amountOfAbove;
					}else if(prob < 0.5){
						++amountOfBelow;
					}
				}
				list.push_back(right / (double) dataMatTest.cols() * 100.0);*/
					list.push_back(ivm.m_logZ);
					if(smallestLog < ivm.m_logZ){
						smallestLog = ivm.m_logZ;
						bestHyperparams[0] = hyperparams[0];
						bestHyperparams[1] = hyperparams[1];
					}
					hyperparams[0] += ivm.m_derivLogZ[0] * fac; // try to maximize the value from -DBL_MAX to 0 -> follow positiv gradient!
					hyperparams[1] += ivm.m_derivLogZ[1] * fac;
					if(fabs(ivm.m_derivLogZ[0]) + fabs(ivm.m_derivLogZ[1]) <= 0.1){
						break;
					}
				}else{
					std::cout << RED << "Training failed: " << train << ", " << ivm.m_logZ << RESET << std::endl;
					break;
				}
			}
			ivm.getKernel().setHyperParams((double)bestHyperparams[0], (double) bestHyperparams[1], sNoise);
			t = ivm.train(true);
			std::cout << "T: " << t << std::endl;
			const TimeFrame timeGrad = swGrad.elapsedAsTimeFrame();
			if(!useRealData){
				DataWriterForVisu::writeSvg("after.svg", ivm, ivm.getSelectedInducingPoints(), y, 50, data);
				system("open after.svg");
			}
			std::cout << "On trainings data:" << std::endl;
			testIvm(ivm, dataMat, y);
			std::cout << "On real test data:" << std::endl;
			testIvm(ivm, dataMatTest, yTest);
			std::cout << RED;
			std::cout << "Time try: " << timeTry << std::endl;
			std::cout << "Time gradient: " << timeGrad << std::endl;
			std::cout << "Rand generator: " << randStartGen << std::endl;
			std::cout << RESET;
			if(!useRealData){
				std::list<int> emptyList;
				DataWriterForVisu::writeSvg("withTest.svg", ivm, emptyList, yTest, 50, testData);
				system("open withTest.svg");
			}
			DataWriterForVisu::writeSvg("result.svg", list, true);
			system("open result.svg");

		}else{
			bool doEpUpdate;
			Settings::getValue("IVM.doEpUpdate", doEpUpdate);
			/*StopWatch total;
		Eigen::MatrixXd m_L = Eigen::MatrixXd::Zero(1,1);
		StopWatch sw;
		const int nr = 50;
		for(unsigned int i = 0; i < nr; ++i){
			Eigen::VectorXd a_nk = Eigen::VectorXd::Zero(m_L.rows());
			const double sqrtNu = 1;
			sw.startTime();
			Eigen::MatrixXd D2(m_L.rows() + 1, m_L.cols() + 1);
			D2 << m_L, Eigen::VectorXd::Zero(m_L.cols()),
					a_nk.transpose(), 1. / sqrtNu;
			m_L = D2;
			sw.recordActTime();
		}
		std::cout << "Time: " << sw.elapsedAvgAsPrettyTime() << ", total: " << total.elapsedAsPrettyTime() << std::endl;
		total.startTime();
		Eigen::MatrixXd m_L2 = Eigen::MatrixXd::Zero(1,1);
		StopWatch sw2;
		std::list<Eigen::VectorXd> m_list;
		std::list<double> m_diag;
		for(unsigned int i = 0; i < nr; ++i){
			Eigen::VectorXd a_nk = Eigen::VectorXd::Zero(m_L2.rows());
			const double sqrtNu = 1;
			sw2.startTime();
			m_list.push_back(a_nk);
			m_diag.push_back(1 / sqrtNu);
			sw2.recordActTime();
		}
		sw2.startTime();
		unsigned int k = 1;
		m_L2 = Eigen::MatrixXd::Zero(nr + 1, nr + 1);
		std::list<double>::iterator diag = m_diag.begin();
		for(std::list<Eigen::VectorXd>::iterator it = m_list.begin(); it != m_list.end(); ++it, ++k){
			for(unsigned int i = 0; i < it->rows(); ++i){
				m_L2(k,i) = (*it)[i];
			}
			m_L2(k,k) = *diag;
			++diag;
		}
		sw2.recordActTime();
		std::cout << "Time new: " << sw2.elapsedAvgAsPrettyTime() << ", total: " << total.elapsedAsPrettyTime()  << std::endl;
		for(int i = 0; i < m_L.rows(); ++i){
			for(int j = 0; j < m_L.cols(); ++j){
				if(m_L2(i,j) != m_L(i,j)){
					std::cout << "Result is not the same!" << std::endl;
				}
			}
		}

		return;*/
			IVM ivm;
			ivm.init(dataMat, y, number, doEpUpdate);
			ivm.setDerivAndLogZFlag(true, false);
			ivm.getKernel().setHyperParams(0.5, 0.8, 0.1);
			bayesopt::Parameters par = initialize_parameters_to_default();
			std::cout << "noise: " << par.noise << std::endl;
			//par.noise = 1-6;
			par.init_method = 500;
			par.n_iterations = 600;
			par.noise = 1e-5;
			par.epsilon = 0.2;
			par.verbose_level = 6;
			par.surr_name = "sGaussianProcessML";

			BayesOptimizerIVM bayOpt(ivm, par);
			const int paramsAmount = 2;
			vectord result(paramsAmount);
			vectord lowerBound(paramsAmount);
			vectord upperBound(paramsAmount);
			lowerBound[0] = 0.0005; // max(0.1,gp.getLenMean() - 2 * gp.getKernel().getLenVar());
			lowerBound[1] = 0.001;
			upperBound[0] = length;// + gp.getKernel().getLenVar();
			upperBound[1] = noise;//max(0.5,gp.getKernel().getLenVar() / gp.getLenMean() * 0.5);
			bayOpt.setBoundingBox(lowerBound, upperBound);
			try{
				bayOpt.optimize(result);
			}catch(std::runtime_error& e){
				printError(e.what()); return;
			}
			const int nr = 50;
			ivm.setNumberOfInducingPoints(nr);
			std::cout << "Best " << result[0] << ", " << result[1] << ", nr: " << nr << std::endl;
			//DataWriterForVisu::writeSvg("logZValues.svg", bayOpt.m_logZValues, true);
			//system("open logZValues.svg");
			ivm.getKernel().setHyperParams(result[0], result[1], ivm.getKernel().sigmaN());
			StopWatch sw;
			ivm.train();
			std::cout << "Time for training: " << sw.elapsedAsPrettyTime() << std::endl;
			if(!useRealData){
				DataWriterForVisu::writeSvg("new.svg", ivm, ivm.getSelectedInducingPoints(), y, 50, data);
				system("open new.svg");
			}
			std::cout << "On trainings data:" << std::endl;
			testIvm(ivm, dataMat, y);
			std::cout << "On real test data:" << std::endl;
			testIvm(ivm, dataMatTest, yTest);
			return;
		}
	}else{
		Eigen::MatrixXd finalMat;
		double bestResult = 0;
		double bestX = 0.988651, bestY = 1.6983;
		if(useRealData){
			bestX = 14.9957;
			bestY = 0.0357228;
		}
		double worstX = 0.57, worstY = 0.84;
		double worstResult = DBL_MAX;
		//for(unsigned int i = 0; i < 100000; ++i){
		const int size = ((1. - 0.9) / 0.005 + 1);
		InLinePercentageFiller::setActMax(size * size);
		int i = 0;
		double bestLogZ = -DBL_MAX;
		if(false){
			Eigen::VectorXd correctVec = Eigen::VectorXd::Zero(size * size);
			for(double x = 0.5; x < 0.6; x += 0.005){
				int j = 0;
				for(double y2 = 0.8; y2 < 0.9; y2 += 0.005)
				{	//double x = 955; double y2 = 0.855;
					IVM ivm;
					ivm.init(dataMat, y, 0.33333 * dataMat.cols(), doEpUpdate);
					ivm.setDerivAndLogZFlag(true, false);
					ivm.getKernel().setHyperParams(x, y2, sNoise);
					bool trained = ivm.train(2);
					if(trained){
						int right = 0;
						int amountOfBelow = 0;
						int amountOfAbove = 0;
						for(int j = 0; j < testData.size(); ++j){
							ClassPoint& ele = *testData[j];
							double prob = ivm.predict(ele);
							//std::cout << "Prob: " << prob << ", label is: " << labels[j] << std::endl;
							if(prob < 0.5 && ele.getLabel() == 0){
								++right;
							}else if(prob > 0.5 && ele.getLabel() == 1){
								++right;
							}
							if(prob > 0.5){
								++amountOfAbove;
							}else{
								++amountOfBelow;
							}
						}
						if(ivm.m_logZ < bestLogZ){
							bestLogZ = ivm.m_logZ;
							bestX = x; bestY = y2;
						}
						if((double) right / testData.size() * 100.0  <= 90. && false){
							bestResult = (double) right / testData.size() * 100.0;
							DataWriterForVisu::writeSvg("out3.svg", ivm, ivm.getSelectedInducingPoints(), y, 50, data);
							std::this_thread::sleep_for(std::chrono::milliseconds((int)(100)));
							openFileInViewer("out3.svg");
							//break;
							std::this_thread::sleep_for(std::chrono::milliseconds((int)(500)));
							std::cout << RED;
							std::cout << "Amount of right: " << (double) right / testData.size() * 100.0 << "%" << std::endl;
							std::cout << "Amount of above: " << (double) amountOfAbove / testData.size() * 100.0 << "%" << std::endl;
							std::cout << "Amount of below: " << (double) amountOfBelow / testData.size() * 100.0 << "%" << std::endl;
							std::cout << "len: " << ivm.getKernel().len() << ", sigmaF: " << ivm.getKernel().sigmaF() <<std::endl;
							std::cout << RESET;
						}
						if(worstResult > (double) right / testData.size() * 100.0){
							worstResult = (double) right / testData.size() * 100.0;
							worstX = x; worstY = y2;
						}
						correctVec[size * i + j] = (double) right / testData.size() * 100.0;
					}else{
						correctVec[size * i + j] = -1;
						//std::cout << "len: " << ivm.getKernel().len() << ", sigmaF: " << ivm.getKernel().sigmaF() <<std::endl;
					}
					InLinePercentageFiller::setActValueAndPrintLine(size * i + j);
					++j;
				}
				++i;
			}
			std::cout << "Best log z: " << bestLogZ << ", for: " << bestX << ", " << bestY << std::endl;
			std::cout << "Worst for: " << worstX << ", " << worstY << std::endl;
			unsigned int amountOfCorrect = 0, amountOfUntrained = 0;
			std::cout << std::endl;
			for(unsigned int i = 0; i < correctVec.size(); ++i){
				if(correctVec[i] > 90.0){
					++amountOfCorrect;
				}
				if(correctVec[i] == -1){
					++amountOfUntrained;
				}
				std::cout << correctVec[i] << " ";
			}
			std::cout << std::endl;
			std::cout << RED << "Amount of correct: " << amountOfCorrect / (double) correctVec.size() * 100.0 << " %" << RESET << std::endl;
			std::cout << RED << "Amount of untrained: " << amountOfUntrained / (double) correctVec.size() * 100.0 << " %" << RESET << std::endl;
			DataWriterForVisu::writeSvg("vec2.svg", correctVec);
			openFileInViewer("vec2.svg");
		}
		IVM ivm;
		std::cout << "Size: " << dataMat.cols() << std::endl;
		int nrOfInducingPoints;
		Settings::getValue("IVM.nrOfInducingPoints", nrOfInducingPoints);
		ivm.init(dataMat, y, nrOfInducingPoints, doEpUpdate);
		ivm.setDerivAndLogZFlag(true, true);
		ivm.getKernel().setHyperParams(bestX, bestY, sNoise);
		std::cout << "Start training" << std::endl;
		std::list<double> times;
		for(unsigned int k = 2; k < 350; k += 2){
			StopWatch sw;
			ivm.setNumberOfInducingPoints(k);
			ivm.train(true, 1);
			times.push_back(sw.elapsedSeconds());
			std::cout << "For IVM " << k << " training: " << sw.elapsedAsTimeFrame() << std::endl;
		}
		DataWriterForVisu::writeSvg("timeLine.svg", times);
		openFileInViewer("timeLine.svg");
		DataWriterForVisu::writeHisto("timeLineHisto.svg", times);
		openFileInViewer("timeLineHisto.svg");
		if(!useRealData && visu){
			DataWriterForVisu::writeSvg("out3.svg", ivm, ivm.getSelectedInducingPoints(), y, 100, data);
			openFileInViewer("out3.svg");
		}
		std::cout << "On trainings data:" << std::endl;
		testIvm(ivm, dataMat, y);
		std::cout << "On real test data:" << std::endl;
		testIvm(ivm, dataMatTest, yTest);
		return;
	}
}



#endif /* TESTS_BINARYCLASSIVMTEST_H_ */
