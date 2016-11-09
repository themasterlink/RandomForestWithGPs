/*
 * binaryClassIVMTest.h
 *
 *  Created on: 04.10.2016
 *      Author: Max
 */

#ifndef TESTS_BINARYCLASSIVMTEST_H_
#define TESTS_BINARYCLASSIVMTEST_H_

#include <Eigen/Dense>
#include "../Data/TotalStorage.h"
#include "../Data/DataConverter.h"
#include "../Data/DataWriterForVisu.h"
#include "../GaussianProcess/IVM.h"
#include "../GaussianProcess/BayesOptimizerIVM.h"
#include "../Base/Settings.h"
#include "../Utility/ConfusionMatrixPrinter.h"
#include "../Base/CommandSettings.h"
#include <chrono>
#include <thread>

void testIvm(IVM& ivm, const ClassData& data){
	int right = 0;
	int amountOfBelow = 0;
	int amountOfAbove = 0;
	const int amountOfTestPoints = data.size();
	std::list<double> probs;
	for(int i = 0; i < amountOfTestPoints; ++i){
		double prob = ivm.predict(*data[i]);
		if(prob > 0.5 && data[i]->getLabel() == ivm.getLabelForOne()){
			++right;
		}else if(prob < 0.5 && data[i]->getLabel() == ivm.getLabelForMinusOne()){
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
		DataWriterForVisu::writeHisto("histo.svg", probs, 14, 0, 1);
		openFileInViewer("histo.svg");
	}
	std::cout << RED;
	std::cout << "Amount of right: " << (double) right / amountOfTestPoints * 100.0 << "%" << std::endl;
	std::cout << "Amount of above: " << (double) amountOfAbove / amountOfTestPoints * 100.0 << "%" << std::endl;
	std::cout << "Amount of below: " << (double) amountOfBelow / amountOfTestPoints * 100.0 << "%" << std::endl;
	std::cout << ivm.getKernel().prettyString() << std::endl;
	std::cout << RESET;
}

void executeForBinaryClassIVM(){
	const int firstPoints = 10000000; // all points
	TotalStorage::readData(firstPoints);
	DataSets datas;
	std::cout << "TotalStorage::getSmallestClassSize(): " << TotalStorage::getSmallestClassSize() << std::endl;
	const int trainAmount = 0.75 * TotalStorage::getSmallestClassSize() * TotalStorage::getAmountOfClass();
	OnlineStorage<ClassPoint*> train;
	OnlineStorage<ClassPoint*> test;
	// starts the training by its own
	TotalStorage::getOnlineStorageCopyWithTest(train, test, trainAmount);
	ClassData& data = train.storage();
	ClassData& testData = test.storage();
	std::cout << "Finish reading " << std::endl;
	bool doEpUpdate;
	Settings::getValue("IVM.doEpUpdate", doEpUpdate);
	Eigen::Vector2i usedClasses;
	usedClasses[0] = 0;
	usedClasses[1] = 1;
	const double sNoise = 0.2;
	if(true){
		std::vector<double> means = {10, 1.7, 1};
		std::vector<double> sds = {8, 0.2, 1};
		int number = 30;
		if(CommandSettings::get_useFakeData()){
			means[0] = 1.;
			sds[0] = 0.1;
			number = 50;
		}
		bool hasMoreThanOneLengthValue = Settings::getDirectBoolValue("IVM.hasLengthMoreThanParam");
		if(true){
			IVM ivm;
			ivm.init(data, number, usedClasses, doEpUpdate);
			ivm.getKernel().changeKernelConfig(hasMoreThanOneLengthValue);
			ivm.getKernel().setGaussianRandomVariables(means, sds);
			ivm.setDerivAndLogZFlag(true, false);
			std::list<double> list;
			GaussianKernelParams bestParams(!hasMoreThanOneLengthValue);
			const int randStartGen = StopWatch::getActTime();
			srand(randStartGen);
			double bestLogZ = -DBL_MAX;
			StopWatch swTry;
			std::string folderLocation;
			if(CommandSettings::get_useFakeData()){
				Settings::getValue("TotalStorage.folderLocFake", folderLocation);
			}else{
				Settings::getValue("TotalStorage.folderLocReal", folderLocation);
			}
			const std::string kernelFilePath = folderLocation + "bestKernelParams.binary";
			if(boost::filesystem::exists(kernelFilePath) && Settings::getDirectBoolValue("IVM.Training.useSavedHyperParams")){
				bestParams.readFromFile(kernelFilePath);
			}else{
				for(int i = 0; i < 2000; ++i){
					ivm.getKernel().newRandHyperParams();
					//ivm.getKernel().getHyperParams().m_sNoise.setAllValuesTo(sNoise);
					std::cout << "Try " << ivm.getKernel().getHyperParams() << ", logZ: " << ivm.m_logZ
							<< ", best value: " << bestParams << ", bestLogZ: " << bestLogZ << std::endl;
					StopWatch sw;
					const bool train = ivm.train();
					std::cout << "Time for training: " << sw.elapsedAsPrettyTime() << std::endl;
					/*if(!train){
					getchar();
				}*/
					if(bestLogZ < ivm.m_logZ && train){
						bestLogZ = ivm.m_logZ;
						ivm.getKernel().getCopyOfParams(bestParams);
					}
				}
			}
			if(Settings::getDirectBoolValue("IVM.Training.overwriteExistingHyperParams")){
				bestParams.writeToFile(kernelFilePath);
			}
			const TimeFrame timeTry = swTry.elapsedAsTimeFrame();
			StopWatch swGrad;
			ivm.setDerivAndLogZFlag(true, true);
			ivm.getKernel().setHyperParamsWith(bestParams);

			bool t = ivm.train();
			if(CommandSettings::get_useFakeData() && (CommandSettings::get_visuRes() > 0 || CommandSettings::get_visuResSimple() > 0) && t){
				DataWriterForVisu::writeSvg("before.svg", ivm, ivm.getSelectedInducingPoints(), data);
				system("open before.svg");
			}
			double fac = 0.0001;
			double smallestLog = -DBL_MAX;
			for(int i = 0; i < 10000; ++i){
				std::cout << "Act " << ivm.getKernel().getHyperParams() << ", logZ: " << ivm.m_logZ << ", deriv: "
						<< ivm.m_derivLogZ << ", fac: " << fac << std::endl;
				//DataWriterForVisu::writeSvg("logZValues.svg", bayOpt.m_logZValues, true);
				//system("open logZValues.svg");
				//ivm.getKernel().setHyperParamsWith(bestParams);
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
						ivm.getKernel().getCopyOfParams(bestParams);
					}
					ivm.getKernel().addToHyperParams(ivm.m_derivLogZ, -fac);
					double sum = 0;
					if(ivm.getKernel().hasLengthMoreThanOneDim()){
						for(unsigned int i = 0; i < ClassKnowledge::amountOfDims(); ++i){
							sum += fabs(ivm.m_derivLogZ.m_length.getValues()[i]);
						}
						sum /= ClassKnowledge::amountOfDims();
					}else{
						sum = fabs(ivm.m_derivLogZ.m_length.getValue());
					}
					if(fabs(ivm.m_derivLogZ.m_fNoise.getValue()) + sum <= 0.1){
						printDebug("Reached end: " << fabs(ivm.m_derivLogZ.m_fNoise.getValue()) + sum);
						break;
					}
				}else{
					std::cout << RED << "Training failed: " << train << ", " << ivm.m_logZ << RESET << std::endl;
					break;
				}
			}
			ivm.getKernel().setHyperParamsWith(bestParams);
			t = ivm.train(true);
			std::cout << "T: " << t << std::endl;
			const TimeFrame timeGrad = swGrad.elapsedAsTimeFrame();
			if(CommandSettings::get_useFakeData() && (CommandSettings::get_visuRes() > 0 || CommandSettings::get_visuResSimple() > 0)){
				DataWriterForVisu::writeSvg("after.svg", ivm, ivm.getSelectedInducingPoints(), data);
				system("open after.svg");
			}
			std::cout << "On trainings data:" << std::endl;
			testIvm(ivm, data);
			std::cout << "On real test data:" << std::endl;
			testIvm(ivm, testData);
			std::cout << RED;
			std::cout << "Time try: " << timeTry << std::endl;
			std::cout << "Time gradient: " << timeGrad << std::endl;
			std::cout << "Rand generator: " << randStartGen << std::endl;
			std::cout << RESET;
			if(CommandSettings::get_useFakeData() && (CommandSettings::get_visuRes() > 0 || CommandSettings::get_visuResSimple() > 0)){
				std::list<int> emptyList;
				DataWriterForVisu::writeSvg("withTest.svg", ivm, emptyList, testData);
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
			ivm.init(data, number, usedClasses, doEpUpdate);
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
			upperBound[0] = means[0] + sds[0];// + gp.getKernel().getLenVar();
			upperBound[1] = means[1] + sds[1];//max(0.5,gp.getKernel().getLenVar() / gp.getLenMean() * 0.5);
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
			ivm.getKernel().setHyperParams(result[0], result[1]);
			StopWatch sw;
			ivm.train();
			std::cout << "Time for training: " << sw.elapsedAsPrettyTime() << std::endl;
			if(CommandSettings::get_useFakeData() && (CommandSettings::get_visuRes() > 0 || CommandSettings::get_visuResSimple() > 0)){
				DataWriterForVisu::writeSvg("new.svg", ivm, ivm.getSelectedInducingPoints(), data);
				system("open new.svg");
			}
			std::cout << "On trainings data:" << std::endl;
			testIvm(ivm, data);
			std::cout << "On real test data:" << std::endl;
			testIvm(ivm, testData);
			return;
		}
	}else{
		Eigen::MatrixXd finalMat;
		double bestResult = 0;
		double bestX = 0.988651, bestY = 1.6983;
		if(!CommandSettings::get_useFakeData()){
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
					ivm.init(data, 0.33333 * data.size(), usedClasses, doEpUpdate);
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
							DataWriterForVisu::writeSvg("out3.svg", ivm, ivm.getSelectedInducingPoints(), data);
							std::this_thread::sleep_for(std::chrono::milliseconds((int)(100)));
							openFileInViewer("out3.svg");
							//break;
							std::this_thread::sleep_for(std::chrono::milliseconds((int)(500)));
							std::cout << RED;
							std::cout << "Amount of right: " << (double) right / testData.size() * 100.0 << "%" << std::endl;
							std::cout << "Amount of above: " << (double) amountOfAbove / testData.size() * 100.0 << "%" << std::endl;
							std::cout << "Amount of below: " << (double) amountOfBelow / testData.size() * 100.0 << "%" << std::endl;
							std::cout << ivm.getKernel().prettyString() << std::endl;
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
			for(unsigned int i = 0; i < correctVec.rows(); ++i){
				if(correctVec[i] > 90.0){
					++amountOfCorrect;
				}
				if(correctVec[i] == -1){
					++amountOfUntrained;
				}
				std::cout << correctVec[i] << " ";
			}
			std::cout << std::endl;
			std::cout << RED << "Amount of correct: " << amountOfCorrect / (double) correctVec.rows() * 100.0 << " %" << RESET << std::endl;
			std::cout << RED << "Amount of untrained: " << amountOfUntrained / (double) correctVec.rows() * 100.0 << " %" << RESET << std::endl;
			DataWriterForVisu::writeSvg("vec2.svg", correctVec);
			openFileInViewer("vec2.svg");
		}
		IVM ivm;
		std::cout << "Size: " << data.size() << std::endl;
		int nrOfInducingPoints;
		Settings::getValue("IVM.nrOfInducingPoints", nrOfInducingPoints);
		ivm.init(data, nrOfInducingPoints, usedClasses, doEpUpdate);
		ivm.setDerivAndLogZFlag(true, true);
		ivm.getKernel().setHyperParams(bestX, bestY, sNoise);
		std::cout << "Start training" << std::endl;

		ivm.setNumberOfInducingPoints(std::min((int)(data.size() * 0.25), 100));
		ivm.train(true, 1);
		/*
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
		*/
		if(CommandSettings::get_useFakeData() && (CommandSettings::get_visuRes() > 0 || CommandSettings::get_visuResSimple() > 0)){
			DataWriterForVisu::writeSvg("out3.svg", ivm, ivm.getSelectedInducingPoints(), data);
			openFileInViewer("out3.svg");
		}
		std::cout << "On trainings data:" << std::endl;
		testIvm(ivm, data);
		std::cout << "On real test data:" << std::endl;
		testIvm(ivm, testData);
		return;
	}
}



#endif /* TESTS_BINARYCLASSIVMTEST_H_ */
