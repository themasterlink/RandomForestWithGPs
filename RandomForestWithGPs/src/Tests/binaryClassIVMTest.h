/*
 * binaryClassIVMTest.h
 *
 *  Created on: 04.10.2016
 *      Author: Max
 */

#ifndef TESTS_BINARYCLASSIVMTEST_H_
#define TESTS_BINARYCLASSIVMTEST_H_

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

void testIvm(IVM& ivm, const OnlineStorage<LabeledVectorX*>& data){
	int right = 0;
	Vector2i rightPerClass;
	rightPerClass[0] = rightPerClass[1] = 0;
	int amountOfBelow = 0;
	int amountOfAbove = 0;
	const int amountOfTestPoints = data.size();
	std::list<Real> probs;
	Vector2i amountPerClass;
	amountPerClass[0] = amountPerClass[1] = 0;
	AvgNumber neg, pos;
	AvgNumber oc, uc, ocBVS, ucBVS;
	StopWatch sw;
	for(int i = 0; i < amountOfTestPoints; ++i){
		Real prob = ivm.predict(*data[i]);
		if(data[i]->getLabel() == ivm.getLabelForOne()){
			++amountPerClass[0];
			pos.addNew(prob);
		}else if(data[i]->getLabel() != ivm.getLabelForOne()){
			++amountPerClass[1];
			neg.addNew(prob);
		}
		Real entropy = 0;
		std::vector<Real> probs = {prob, 1 - prob};
		for(unsigned int j = 0; j < 2; ++j){
			if(probs[j] > 0){
				entropy -= probs[j] * log(probs[j]) / log(2);
			}
		}
		Real max1 = 0, max2 = 0;
		for(unsigned int j = 0; j < 2; ++j){
			if(probs[j] > max1){
				max2 = max1;
				max1 = probs[j];
			}
		}
		Real entropyBVS = max2 / max1;
		if(prob >= 0.5 && data[i]->getLabel() == ivm.getLabelForOne()){
			++right; ++rightPerClass[0];
			uc.addNew(entropy);
			ucBVS.addNew(entropyBVS);
		}else if(prob < 0.5 && data[i]->getLabel() != ivm.getLabelForOne()){
			++right; ++rightPerClass[1];
			uc.addNew(entropy);
			ucBVS.addNew(entropyBVS);
		}else{
			oc.addNew(1.-entropy);
			ocBVS.addNew(1.-entropyBVS);
			printOnScreen("Class: " << ClassKnowledge::getNameFor(data[i]->getLabel()) << ", for 0: " << probs[0] << ", for 1: " << probs[1]);
		}
		if(prob > 0.5){
			++amountOfAbove;
		}else if(prob < 0.5){
			++amountOfBelow;
		}
		probs.push_back(prob);
	}
	printOnScreen("Prediction took: " << sw.elapsedAsPrettyTime());
	if(amountOfTestPoints > 0 && CommandSettings::get_plotHistos()){
		DataWriterForVisu::writeHisto("histo.svg", probs, 14, 0, 1);
		openFileInViewer("histo.svg");
	}
	printOnScreen("Amount of 1: " << amountPerClass[0] << ", amount of -1: " << amountPerClass[1]);
	printOnScreen("Amount of right: " << (Real) right / amountOfTestPoints * 100.0 << " %%");
	printOnScreen("Amount of above: " << (Real) amountOfAbove / amountOfTestPoints * 100.0 << " %%");
	printOnScreen("Amount of below: " << (Real) amountOfBelow / amountOfTestPoints * 100.0 << " %%");
	printOnScreen("Recall for  1: " << (Real) rightPerClass[0] / (Real) amountPerClass[0] * 100.0 << " %%");
	printOnScreen("Recall for -1: " << (Real) rightPerClass[1] / (Real) amountPerClass[1] * 100.0 << " %%");
	printOnScreen("Precision for  1: " << (Real) rightPerClass[0] / right * 100.0 << "%%");
	printOnScreen("Precision for -1: " << (Real) rightPerClass[1] / right * 100.0 << "%%");
	printOnScreen("Avg for  1: " << (Real) pos.mean() * 100.0 << "%%");
	printOnScreen("Avg for -1: " << (Real) neg.mean() * 100.0 << "%%");
	printOnScreen("Overconf:  " << (Real) oc.mean() * 100.0 << "%%");
	printOnScreen("Underconf: " << (Real) uc.mean() * 100.0 << "%%");
	printOnScreen("Overconf BVS:  " << (Real) ocBVS.mean() * 100.0 << "%%");
	printOnScreen("Underconf BVS: " << (Real) ucBVS.mean() * 100.0 << "%%");

	printOnScreen("Amount of 1 in total: " << (Real) amountPerClass[0] / amountOfTestPoints * 100.0 << " %%");
}

void sampleInParallel(IVM* ivm, GaussianKernelParams* bestParams, Real* bestLogZ, boost::mutex* mutex, const Real durationOfTraining, int* counter){
	StopWatch sw;
	while(sw.elapsedSeconds() < durationOfTraining && ivm->getGaussianKernel() != nullptr){
		ivm->getGaussianKernel()->newRandHyperParams();
		ivm->train(false, 0);
		mutex->lock();
		if(*bestLogZ < ivm->m_logZ){
			ivm->getGaussianKernel()->getCopyOfParams(*bestParams);
			*bestLogZ = ivm->m_logZ;
		}
		++(*counter);
		mutex->unlock();
	}
}

GaussianKernelParams sampleParams(OnlineStorage<LabeledVectorX*>& storage, int number, const Vector2i& usedClasses, bool doEpUpdate,
		const std::vector<Real>& means, const std::vector<Real>& sds){
	boost::thread_group group;
	boost::mutex mutex;
	const auto nrOfParallel = ThreadMaster::getAmountOfThreads();
	Real bestLogZ = NEG_REAL_MAX;
	const Real durationOfTraining = CommandSettings::get_samplingAndTraining();
	std::vector<IVM*> ivms(nrOfParallel);
	bool hasMoreThanOneLengthValue = Settings::getDirectBoolValue("IVM.hasLengthMoreThanParam");
	GaussianKernelParams bestParams(!hasMoreThanOneLengthValue);
	int counter = 0;
	for(unsigned int i = 0; i < nrOfParallel; ++i){
		ivms[i] = new IVM(storage);
		ivms[i]->getGaussianKernel()->changeKernelConfig(hasMoreThanOneLengthValue);
		ivms[i]->getGaussianKernel()->setGaussianRandomVariables(means, sds);
		ivms[i]->setDerivAndLogZFlag(true, false);
		ivms[i]->init(number, usedClasses, doEpUpdate);
		group.add_thread(new boost::thread(boost::bind(&sampleInParallel, ivms[i], &bestParams, &bestLogZ, &mutex, durationOfTraining, &counter)));
	}
	InLinePercentageFiller::setActMaxTime(durationOfTraining);
	printOnScreen("It will take: " << TimeFrame(durationOfTraining));
	StopWatch sw;
	while(sw.elapsedSeconds() < durationOfTraining){
		InLinePercentageFiller::printLineWithRestTimeBasedOnMaxTime(counter);
		sleepFor(0.1);
	}
	group.join_all();
	InLinePercentageFiller::printLineWithRestTimeBasedOnMaxTime(counter, true);
	return bestParams;
}

void trainIVM(IVM* ivm, const int verboseLevel){
	UNUSED(verboseLevel);
	ivm->train(CommandSettings::get_samplingAndTraining() > 0, 1, false);
}

void executeForBinaryClassIVM(){
	int firstPoints; // all points
	Settings::getValue("TotalStorage.amountOfPointsUsedForTraining", firstPoints);
	const Real share = Settings::getDirectRealValue("TotalStorage.shareForTraining");
	firstPoints /= share;
	printOnScreen("Read " << firstPoints << " points per class");
	TotalStorage::readData(firstPoints);
	DataSets datas;
	printOnScreen("TotalStorage::getSmallestClassSize(): " << TotalStorage::getSmallestClassSize() << " with " << TotalStorage::getAmountOfClass() << " classes");
	const int trainAmount = share * (std::min((int) TotalStorage::getSmallestClassSize(), firstPoints) * (Real) TotalStorage::getAmountOfClass());
	OnlineStorage<LabeledVectorX*> train;
	OnlineStorage<LabeledVectorX*> test;
	// starts the training by its own
	TotalStorage::getOnlineStorageCopyWithTest(train, test, trainAmount);
	printOnScreen("Finish reading ");
	Vector2i usedClasses;
	usedClasses[0] = 0;
	usedClasses[1] = 1;
	bool doEpUpdate;
	Settings::getValue("IVM.doEpUpdate", doEpUpdate);
	if(true){
		int nrOfInducingPoints;
		Settings::getValue("IVM.nrOfInducingPoints", nrOfInducingPoints);
		InformationPackage* package = new InformationPackage(InformationPackage::IVM_TRAIN, 0, train.size());
		package->setStandartInformation("Binary Ivm Training");
		IVM ivm(train);
		ivm.setInformationPackage(package);
		ivm.init(nrOfInducingPoints, usedClasses, doEpUpdate, false);
		boost::thread_group group;
		group.add_thread(new boost::thread(boost::bind(&trainIVM, &ivm, 1)));
		ThreadMaster::appendThreadToList(package);
		group.join_all();
		bool ret = ivm.isTrained();
		if(ret){
			printOnScreen("On " << train.size() << " points from trainings data:");
			testIvm(ivm, train);
			printOnScreen("On " << test.size() << " points from real test data:");
			testIvm(ivm, test);
			if(CommandSettings::get_visuRes() > 0){
			DataWriterForVisu::writeImg("test.png", &ivm, train.storage());
			openFileInViewer("test.png");
			}
//			OnlineStorage<LabeledVectorX*> removedTrain;
//			OnlineStorage<LabeledVectorX*> removedTest;
//			TotalStorage::getRemovedOnlineStorageCopyWithTest(train, test);
//			printOnScreen("On " << train.size() << " removed points from trainings data:");
//			testIvm(ivm, removedTrain);
//			printOnScreen("On " << test.size() << " removed points from real test data:");
//			testIvm(ivm, removedTest);

//			DataWriterForVisu::writeSvg("test.svg", &ivm, train.storage());
//			openFileInViewer("test.svg");
//			DataWriterForVisu::writeImg("empty.png", (PredictorBinaryClass*)nullptr, train.storage());
	//		DataWriterForVisu::writeSvg("test.svg", &ivm, test.storage());
		}else{
			printError("The ivm could not be trained!");
		}
//		openFileInViewer("empty.png");
	}else{
		Real sNoise = Settings::getDirectRealValue("KernelParam.sNoise");
		if(CommandSettings::get_samplingAndTraining() > 0.){
			std::vector<Real> means = {10, 1.2, 0.5};
			std::vector<Real> sds = {8, 0.8, 0.4};
			int number = 100;
			if(CommandSettings::get_useFakeData()){
				means[0] = 1.2;
				sds[0] = 0.8;
			}
			Settings::getValue("IVM.nrOfInducingPoints", number);
			bool hasMoreThanOneLengthValue = Settings::getDirectBoolValue("IVM.hasLengthMoreThanParam");
			if(true){
				IVM ivm(train);
				ivm.init(number, usedClasses, doEpUpdate);
				ivm.getGaussianKernel()->changeKernelConfig(hasMoreThanOneLengthValue);
				ivm.getGaussianKernel()->setGaussianRandomVariables(means, sds);
				ivm.setDerivAndLogZFlag(true, false);
				std::list<Real> list;
				GaussianKernelParams bestParams(!hasMoreThanOneLengthValue);
				const int randStartGen = StopWatch::getActTime();
				srand(randStartGen);
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
					StopWatch sw;
					bestParams = sampleParams(train, number, usedClasses, doEpUpdate, means, sds);
					printOnScreen("Time for sampling: " << sw.elapsedAsPrettyTime() << ", result: " << bestParams);
				}
				if(Settings::getDirectBoolValue("IVM.Training.overwriteExistingHyperParams")){
					bestParams.writeToFile(kernelFilePath);
				}
				const TimeFrame timeTry = swTry.elapsedAsTimeFrame();
				StopWatch swGrad;
				ivm.getGaussianKernel()->setHyperParamsWith(bestParams);
				bool t = ivm.train(false, 0);
				if(CommandSettings::get_useFakeData() && (CommandSettings::get_visuRes() > 0 || CommandSettings::get_visuResSimple() > 0) && t){
					DataWriterForVisu::writeSvg("before.svg", ivm, ivm.getSelectedInducingPoints(), train.storage());
					openFileInViewer("before.svg");
				}
				ivm.setDerivAndLogZFlag(true, true);
				Real fac = 0.0001;
				Real smallestLog = NEG_REAL_MAX;
				const int amountOfTrainingSteps = 0;
				for(int i = 0; i < amountOfTrainingSteps; ++i){
					printOnScreen("Act " << ivm.getGaussianKernel()->getHyperParams() << ", logZ: " << ivm.m_logZ << ", deriv: "
							<< ivm.m_derivLogZ << ", fac: " << fac);
					//DataWriterForVisu::writeSvg("logZValues.svg", bayOpt.m_logZValues, true);
					//openFileInViewer("logZValues.svg");
					//ivm.getGaussianKernel()->setHyperParamsWith(bestParams);
					StopWatch sw;
					bool train = ivm.train(false, 1);
					if(train && !isnan(ivm.m_logZ)){
						printOnScreen("Time for training: " << sw.elapsedAsPrettyTime());
						/*int right = 0;
				int amountOfBelow = 0;
				int amountOfAbove = 0;
				for(int i = 0; i < dataMatTest.cols(); ++i){
					Real prob = ivm.predict(dataMatTest.col(i));
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
				list.push_back(right / (Real) dataMatTest.cols() * 100.0);*/
						list.push_back(ivm.m_logZ);
						if(smallestLog < ivm.m_logZ){
							smallestLog = ivm.m_logZ;
							ivm.getGaussianKernel()->getCopyOfParams(bestParams);
						}
						ivm.getGaussianKernel()->addToHyperParams(ivm.m_derivLogZ, -fac);
						Real sum = 0;
						if(ivm.getGaussianKernel()->hasLengthMoreThanOneDim()){
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
						printOnScreen("Training failed: " << train << ", " << ivm.m_logZ);
						break;
					}
				}
				ivm.getGaussianKernel()->setHyperParamsWith(bestParams);
				t = ivm.train(false, 0);
				printOnScreen("T: " << t);
				const TimeFrame timeGrad = swGrad.elapsedAsTimeFrame();
				if(CommandSettings::get_useFakeData() && (CommandSettings::get_visuRes() > 0 || CommandSettings::get_visuResSimple() > 0)){
					if(amountOfTrainingSteps > 0){
						DataWriterForVisu::writeSvg("after.svg", ivm, ivm.getSelectedInducingPoints(), train.storage());
						openFileInViewer("after.svg");
					}
				}
				printOnScreen("On trainings data:");
				testIvm(ivm, train);
				printOnScreen("On real test data:");
				testIvm(ivm, test);
				printOnScreen("Time try: " << timeTry);
				printOnScreen("Time gradient: " << timeGrad);
				printOnScreen("Rand generator: " << randStartGen);
				if(CommandSettings::get_useFakeData() && (CommandSettings::get_visuRes() > 0 || CommandSettings::get_visuResSimple() > 0)){
					std::list<unsigned int> emptyList;
					DataWriterForVisu::writeSvg("withTest.svg", ivm, emptyList, test.storage());
					openFileInViewer("withTest.svg");
				}
				if(amountOfTrainingSteps > 0){
					DataWriterForVisu::writeSvg("result.svg", list, true);
					openFileInViewer("result.svg");
				}

			}else{
				bool doEpUpdate;
				Settings::getValue("IVM.doEpUpdate", doEpUpdate);
				/*StopWatch total;
		Matrix m_L = Matrix::Zero(1,1);
		StopWatch sw;
		const int nr = 50;
		for(unsigned int i = 0; i < nr; ++i){
			VectorX a_nk = VectorX::Zero(m_L.rows());
			const Real sqrtNu = 1;
			sw.startTime();
			Matrix D2(m_L.rows() + 1, m_L.cols() + 1);
			D2 << m_L, VectorX::Zero(m_L.cols()),
					a_nk.transpose(), 1. / sqrtNu;
			m_L = D2;
			sw.recordActTime();
		}
		printOnScreen("Time: " << sw.elapsedAvgAsPrettyTime() << ", total: " << total.elapsedAsPrettyTime());
		total.startTime();
		Matrix m_L2 = Matrix::Zero(1,1);
		StopWatch sw2;
		std::list<VectorX> m_list;
		std::list<Real> m_diag;
		for(unsigned int i = 0; i < nr; ++i){
			VectorX a_nk = VectorX::Zero(m_L2.rows());
			const Real sqrtNu = 1;
			sw2.startTime();
			m_list.push_back(a_nk);
			m_diag.push_back(1 / sqrtNu);
			sw2.recordActTime();
		}
		sw2.startTime();
		unsigned int k = 1;
		m_L2 = Matrix::Zero(nr + 1, nr + 1);
		auto diag = m_diag.begin();
		for(auto it = m_list.begin(); it != m_list.end(); ++it, ++k){
			for(unsigned int i = 0; i < it->rows(); ++i){
				m_L2(k,i) = (*it)[i];
			}
			m_L2(k,k) = *diag;
			++diag;
		}
		sw2.recordActTime();
		printOnScreen("Time new: " << sw2.elapsedAvgAsPrettyTime() << ", total: " << total.elapsedAsPrettyTime() );
		for(int i = 0; i < m_L.rows(); ++i){
			for(int j = 0; j < m_L.cols(); ++j){
				if(m_L2(i,j) != m_L(i,j)){
					printOnScreen("Result is not the same!");
				}
			}
		}

		return;*/
//				IVM ivm(train);
//				ivm.init(number, usedClasses, doEpUpdate);
//				ivm.setDerivAndLogZFlag(true, false);
//				ivm.getGaussianKernel()->setHyperParams(0.5, 0.8, 0.1);
//				bayesopt::Parameters par = initialize_parameters_to_default();
//				printOnScreen("noise: " << par.noise);
//				//par.noise = 1-6;
//				par.init_method = 500;
//				par.n_iterations = 600;
//				par.noise = 1e-5;
//				par.epsilon = 0.2;
//				par.verbose_level = 6;
//				par.surr_name = "sGaussianProcessML";

//				BayesOptimizerIVM bayOpt(ivm, par);
//				const int paramsAmount = 2;
//				vectord result(paramsAmount);
//				vectord lowerBound(paramsAmount);
//				vectord upperBound(paramsAmount);
//				lowerBound[0] = 0.0005; // max(0.1,gp.getLenMean() - 2 * gp.getGaussianKernel()->getLenVar());
//				lowerBound[1] = 0.001;
//				upperBound[0] = means[0] + sds[0];// + gp.getGaussianKernel()->getLenVar();
//				upperBound[1] = means[1] + sds[1];//max(0.5,gp.getGaussianKernel()->getLenVar() / gp.getLenMean() * 0.5);
//				bayOpt.setBoundingBox(lowerBound, upperBound);
//				try{
//					bayOpt.optimize(result);
//				}catch(std::runtime_error& e){
//					printError(e.what()); return;
//				}
//				const int nr = 50;
//				ivm.setNumberOfInducingPoints(nr);
//				printOnScreen("Best " << result[0] << ", " << result[1] << ", nr: " << nr);
				//DataWriterForVisu::writeSvg("logZValues.svg", bayOpt.m_logZValues, true);
				//openFileInViewer("logZValues.svg");
//				ivm.getGaussianKernel()->setHyperParams(result[0], result[1]);
//				StopWatch sw;
//				ivm.train(false, 0);
//				printOnScreen("Time for training: " << sw.elapsedAsPrettyTime());
//				if(CommandSettings::get_useFakeData() && (CommandSettings::get_visuRes() > 0 || CommandSettings::get_visuResSimple() > 0)){
//					DataWriterForVisu::writeSvg("new.svg", ivm, ivm.getSelectedInducingPoints(), train.storage());
//					openFileInViewer("new.svg");
//				}
//				printOnScreen("On trainings data:");
//				testIvm(ivm, train);
//				printOnScreen("On Real test data:");
//				testIvm(ivm, test);
				return;
			}
		}else{
			Matrix finalMat;
			Real bestResult = 0;
			Real bestX = Settings::getDirectRealValue("KernelParam.len");
			Real bestY = Settings::getDirectRealValue("KernelParam.fNoise");
			if(!CommandSettings::get_useFakeData()){
				bestX = 32.5579;
				bestY = 1.24258;
				sNoise = 1.33968;
			}
			Real worstX = 0.57, worstY = 0.84;
			Real worstResult = REAL_MAX;
			//for(unsigned int i = 0; i < 100000; ++i){
			const int size = ((1. - 0.9) / 0.005 + 1);
			InLinePercentageFiller::setActMax(size * size);
			int i = 0;
			Real bestLogZ = NEG_REAL_MAX;
			if(false){
				VectorX correctVec = VectorX::Zero(size * size);
				for(Real x = 0.5; x < 0.6; x += 0.005){
					int j = 0;
					for(Real y2 = 0.8; y2 < 0.9; y2 += 0.005)
					{	//Real x = 955; Real y2 = 0.855;
						IVM ivm(train);
						ivm.init(0.33333 * train.size(), usedClasses, doEpUpdate);
						ivm.setDerivAndLogZFlag(true, false);
						ivm.getGaussianKernel()->setHyperParams(x, y2, sNoise);
						bool trained = ivm.train(false, 0);
						if(trained){
							int right = 0;
							int amountOfBelow = 0;
							int amountOfAbove = 0;
							for(int j = 0; j < test.size(); ++j){
								LabeledVectorX& ele = *test[j];
								Real prob = ivm.predict(ele);
								//printOnScreen("Prob: " << prob << ", label is: " << labels[j]);
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
							if((Real) right / test.size() * 100.0  <= 90. && false){
								bestResult = (Real) right / test.size() * 100.0;
								DataWriterForVisu::writeSvg("out3.svg", ivm, ivm.getSelectedInducingPoints(), train.storage());
								std::this_thread::sleep_for(std::chrono::milliseconds((int)(100)));
								openFileInViewer("out3.svg");
								//break;
								std::this_thread::sleep_for(std::chrono::milliseconds((int)(500)));
								printOnScreen("Amount of right: " << (Real) right / test.size() * 100.0 << "%");
								printOnScreen("Amount of above: " << (Real) amountOfAbove / test.size() * 100.0 << "%");
								printOnScreen("Amount of below: " << (Real) amountOfBelow / test.size() * 100.0 << "%");
								printOnScreen(ivm.getGaussianKernel()->prettyString());
							}
							if(worstResult > (Real) right / test.size() * 100.0){
								worstResult = (Real) right / test.size() * 100.0;
								worstX = x; worstY = y2;
							}
							correctVec[size * i + j] = (Real) right / test.size() * 100.0;
						}else{
							correctVec[size * i + j] = -1;
							//printOnScreen("len: " << ivm.getGaussianKernel()->len() << ", sigmaF: " << ivm.getGaussianKernel()->sigmaF());
						}
						InLinePercentageFiller::setActValueAndPrintLine(size * i + j);
						++j;
					}
					++i;
				}
				printOnScreen("Best log z: " << bestLogZ << ", for: " << bestX << ", " << bestY);
				printOnScreen("Worst for: " << worstX << ", " << worstY);
				unsigned int amountOfCorrect = 0, amountOfUntrained = 0;
				for(unsigned int i = 0; i < correctVec.rows(); ++i){
					if(correctVec[i] > 90.0){
						++amountOfCorrect;
					}
					if(correctVec[i] == -1){
						++amountOfUntrained;
					}
					printOnScreen(correctVec[i] << " ");
				}
				printOnScreen("Amount of correct: " << amountOfCorrect / (Real) correctVec.rows() * 100.0 << " %");
				printOnScreen("Amount of untrained: " << amountOfUntrained / (Real) correctVec.rows() * 100.0 << " %");
				DataWriterForVisu::writeSvg("vec2.svg", correctVec);
				openFileInViewer("vec2.svg");
			}
			IVM ivm(train);
			printOnScreen("Size: " << train.size());
			int nrOfInducingPoints;
			Settings::getValue("IVM.nrOfInducingPoints", nrOfInducingPoints);
			ivm.init(nrOfInducingPoints, usedClasses, doEpUpdate);
			ivm.setDerivAndLogZFlag(true, true);
			if(Settings::getDirectBoolValue("IVM.hasLengthMoreThanParam")){
				bestY = 1.24258; //1.67215;
				std::vector<Real> bestXs = {1.72188, 0.209048};//{1.03035, -0.280561};
				if(!CommandSettings::get_useFakeData()){
					bestXs = {13.469127, 12.469127};
					bestY = 0.0357228;
					sNoise = 0.979355;
				}
				ivm.getGaussianKernel()->setHyperParams(bestXs, bestY, sNoise);
			}else{
				ivm.getGaussianKernel()->setHyperParams(bestX, bestY, sNoise);
			}
			printOnScreen("Start training");
			StopWatch swTraining;
			//	ivm.setNumberOfInducingPoints(std::min((int)(data.size() * 0.25), 100));
			ivm.train(false, 1);
			printOnScreen("Needed for training: " << swTraining.elapsedAsTimeFrame());
			/*
		std::list<Real> times;
		for(unsigned int k = 2; k < 350; k += 2){
			StopWatch sw;
			ivm.setNumberOfInducingPoints(k);
			ivm.train(true, 1);
			times.push_back(sw.elapsedSeconds());
			printOnScreen("For IVM " << k << " training: " << sw.elapsedAsTimeFrame());
		}
		DataWriterForVisu::writeSvg("timeLine.svg", times);
		openFileInViewer("timeLine.svg");
		DataWriterForVisu::writeHisto("timeLineHisto.svg", times);
		openFileInViewer("timeLineHisto.svg");
			 */
			if((CommandSettings::get_visuRes() > 0 || CommandSettings::get_visuResSimple() > 0)){
				int x = 0, y = 1;
				if(!CommandSettings::get_useFakeData() && ivm.getGaussianKernel()->hasLengthMoreThanOneDim()){
					Real highestVal = NEG_REAL_MAX, secondHighestVal = NEG_REAL_MAX;
					for(unsigned int i = 0; i < ClassKnowledge::amountOfDims(); ++i){
						const Real len = ivm.getGaussianKernel()->getHyperParams().m_length.getValues()[i];
						if(len > highestVal){
							secondHighestVal = highestVal;
							y = x;
							highestVal = len;
							x = i;
						}else if(len > secondHighestVal){
							secondHighestVal = len;
							y = i;
						}
					}
				}else if(CommandSettings::get_useFakeData()){
					DataWriterForVisu::writeSvg("out3.svg", ivm, ivm.getSelectedInducingPoints(), train.storage(), x, y);
					openFileInViewer("out3.svg");
					//				DataWriterForVisu::writeSvg("mu.svg", ivm, ivm.getSelectedInducingPoints(), data, x, y, 1);
					//				openFileInViewer("mu.svg");
					//				DataWriterForVisu::writeSvg("sigma.svg", ivm, ivm.getSelectedInducingPoints(), data, x, y, 2);
					//				openFileInViewer("sigma.svg");
				}
			}
			printOnScreen("On " << train.size() << " points from trainings data:");
			testIvm(ivm, train);
			printOnScreen("On " << test.size() << " points from real test data:");
			testIvm(ivm, test);
			return;
		}

	}
}



#endif /* TESTS_BINARYCLASSIVMTEST_H_ */
