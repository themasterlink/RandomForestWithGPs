/*
 * IVMMultiBinary.cc
 *
 *  Created on: 17.11.2016
 *      Author: Max
 */

#include "IVMMultiBinary.h"
#include "../Base/CommandSettings.h"

IVMMultiBinary::IVMMultiBinary(OnlineStorage<ClassPoint*>& storage,
		const unsigned int numberOfInducingPointsPerIVM,
		const bool doEPUpdate):
		m_storage(storage),
		m_numberOfInducingPointsPerIVM(numberOfInducingPointsPerIVM),
		m_doEpUpdate(doEPUpdate),
		m_init(false),
		m_firstTraining(true),
		m_correctAmountForTrainingData(0),
		m_orfForKernel(nullptr){
	m_storage.attach(this);
}

void IVMMultiBinary::update(Subject* caller, unsigned int event){
	if(caller->classType() == m_storage.classType()){
		if(event == OnlineStorage<ClassPoint*>::APPENDBLOCK){
			// assumption that the correct OnlineStorage is callig and not a false one -> no check
			const unsigned int lastUpdateIndex = m_storage.getLastUpdateIndex();
			if(!m_init && lastUpdateIndex == 0){
				// to find out the amount of used classes in this ivm look at the data
				std::map<unsigned int, unsigned int> classCounter;
				for(ClassData::const_iterator it = m_storage.begin(); it != m_storage.end(); ++it){
					const unsigned int label = (**it).getLabel();
					std::map<unsigned int, unsigned int>::iterator itClass = classCounter.find(label);
					if(itClass == classCounter.end()){
						classCounter.insert(std::pair<unsigned int, unsigned int>(label, 1));
						m_classOfIVMs.push_back(label);
					}else{
						itClass->second += 1;
					}
				}
				int amountOfUsedClasses = 0;
				m_isClassUsed.resize(classCounter.size());
				const int minimumNeededAmountOfElements = 0.75 * m_numberOfInducingPointsPerIVM;
				int t = 0;
				for (std::map<unsigned int, unsigned int>::const_iterator it = classCounter.begin(); it != classCounter.end(); ++it, ++t){
					m_isClassUsed[t] = it->second > minimumNeededAmountOfElements;
					if(m_isClassUsed[t]){
						++amountOfUsedClasses;
					}
				}
				m_ivms.resize(amountOfUsedClasses);
				m_correctAmountForTrainingDataForClasses.resize(amountOfClasses());
				std::fill(m_correctAmountForTrainingDataForClasses.begin(), m_correctAmountForTrainingDataForClasses.end(), 0);
				if(amountOfUsedClasses > 0){
					const bool calcDifferenceMatrixAlone = false;
					int kernelType = 0;
					Settings::getValue("IVM.kernelType", kernelType);
					if(kernelType == 1){
						int maxDepth, samplingAmount;
						Settings::getValue("RandomForestKernel.maxDepth", maxDepth);
						Settings::getValue("RandomForestKernel.samplingAmount", samplingAmount);
						m_orfForKernel = new OnlineRandomForest(m_storage, maxDepth, amountOfClasses());
						// train the trees before the ivms are used
						m_orfForKernel->setDesiredAmountOfTrees(samplingAmount);
						m_orfForKernel->update(&m_storage, OnlineStorage<ClassPoint*>::APPENDBLOCK);
					}
					std::list<IVM*> usedIvms;
					for(unsigned int i = 0; i < m_ivms.size(); ++i){
						if(m_isClassUsed[i]){
							m_ivms[i] = new IVM(m_storage, true);
							usedIvms.push_back(m_ivms[i]);
							if(m_ivms[i]->getKernelType() == IVM::RF){
								m_ivms[i]->setOnlineRandomForest(m_orfForKernel);
							}
							Eigen::Vector2i usedClasses;
							usedClasses << m_classOfIVMs[i], -1;
							m_ivms[i]->init(m_numberOfInducingPointsPerIVM, usedClasses,
									m_doEpUpdate, calcDifferenceMatrixAlone);
						}
					}
					const int nrOfParallel = (int) boost::thread::hardware_concurrency();
					const int size = (m_storage.size() * m_storage.size() + m_storage.size()) / 2;
					const int sizeOfPart =  size / nrOfParallel;
					if(kernelType == 0 && usedIvms.size() > 0){ // GAUSS, calc the kernel matrix
						Eigen::MatrixXd* differenceMatrix = new Eigen::MatrixXd(m_storage.size(), m_storage.size());
						boost::thread_group group;
						for(unsigned int i = 0; i < nrOfParallel; ++i){
							int start = sizeOfPart * i;
							int end = (i + 1 != nrOfParallel) ? sizeOfPart * (i + 1) : size;
							group.add_thread(new boost::thread(boost::bind(&IVMMultiBinary::initInParallel, this, start, end, differenceMatrix)));
						}
						group.join_all();
						for(std::list<IVM*>::iterator itIvm = usedIvms.begin(); itIvm != usedIvms.end(); ++itIvm){
							(*itIvm)->getGaussianKernel()->setDifferenceMatrix(differenceMatrix);
						}
					}
					m_init = true;
					train();
				}
			}else if(!m_init && lastUpdateIndex != 0){
				printError("The IVMMultiBinary was not constructed with an empty online storage!");
			}else if(m_init){
				train();
			}
		}else{
			printError("This event type is not handled here!");
		}
	}else{
		printError("This caller is unknown!");
	}
}

IVMMultiBinary::~IVMMultiBinary() {
	delete m_orfForKernel;
}

void IVMMultiBinary::train(){
	if(!m_firstTraining){
		if(m_packages.size() > 0 ){
			// abort all running ivm retrainings
			for(std::list<InformationPackage*>::iterator it = m_packages.begin(); it != m_packages.end(); ++it){
				(*it)->abortTraing();
			}
			// wait until all old runnings ivm retrainings are finished
			bool taskFinish = true;
			while(taskFinish){
				taskFinish = false;
				for(std::list<InformationPackage*>::iterator it = m_packages.begin(); it != m_packages.end(); ++it){
					if(!(*it)->isTaskFinished()){
						taskFinish = true;
						break;
					}
				}
				usleep(0.05 * 1e6);
			}
			// remove all running threads
			for(std::list<InformationPackage*>::iterator it = m_packages.begin(); it != m_packages.end(); ++it){
				delete *it;
			}
			m_packages.clear();
		}
		InformationPackage* wholePackage = new InformationPackage(InformationPackage::IVM_MULTI_UPDATE, m_correctAmountForTrainingData, m_storage.size());
		m_group.add_thread(new boost::thread(boost::bind(&IVMMultiBinary::retrainAllIvmsIfNeeded, this, wholePackage)));
	}else{
		if(!Settings::getDirectBoolValue("IVM.hasLengthMoreThanParam")){
			std::vector<double> means = {Settings::getDirectDoubleValue("KernelParam.lenMean"),
					Settings::getDirectDoubleValue("KernelParam.fNoiseMean"),
					Settings::getDirectDoubleValue("KernelParam.sNoiseMean")};
			std::vector<double> sds = {Settings::getDirectDoubleValue("KernelParam.lenVar"),
					Settings::getDirectDoubleValue("KernelParam.fNoiseVar"),
					Settings::getDirectDoubleValue("KernelParam.sNoiseVar")};
			std::stringstream stringStream;
			stringStream << "Used means: ";
			for(unsigned int i = 0; i < 3; ++i){
				stringStream << means[i] << ", ";
			}
			stringStream << "used sds:";
			for(unsigned int i = 0; i < 3; ++i){
				stringStream << sds[i];
				if(i != 2)
					stringStream << ", ";
			}
			printOnScreen(stringStream.str());
		}
		double durationOfTraining = CommandSettings::get_samplingAndTraining();
		boost::thread_group group;
		boost::thread_group groupForRetraining;
		const int nrOfParallel = boost::thread::hardware_concurrency();
		StopWatch sw;
		std::vector<int> counterRes(amountOfClasses(), 0);
		std::vector<bool> stillWorking(amountOfClasses(), true);
		std::vector<InformationPackage*> packages(amountOfClasses());
		for(unsigned int i = 0; i < amountOfClasses(); ++i){
			if(m_isClassUsed[i]){
				packages[i] = new InformationPackage(InformationPackage::IVM_TRAIN, m_correctAmountForTrainingDataForClasses[i], m_storage.size());
			}
		}
		double durationOfWholeTraining = durationOfTraining;
		if(amountOfClasses() > nrOfParallel){
			durationOfWholeTraining *= ceil(amountOfClasses() / (double) nrOfParallel);
		}
//		if(amountOfClasses() <= nrOfParallel){
			const bool fitParams = CommandSettings::get_samplingAndTraining();
			if(fitParams){
				InLinePercentageFiller::setActMaxTime(durationOfWholeTraining);
			}else{
				InLinePercentageFiller::setActMaxTime(5); // max time for sampling
			}
			std::list<InformationPackage*> packagesForRetrain;
			std::vector<unsigned char> stateOfIvms(amountOfClasses(), 0);
			for(unsigned int i = 0; i < amountOfClasses(); ++i){
				if(m_isClassUsed[i]){
					group.add_thread(new boost::thread(boost::bind(&IVMMultiBinary::trainInParallel, this, i, durationOfTraining, packages[i])));
					stateOfIvms[i] = 1;
				}
			}

			unsigned int counter = 0;
			bool stillOneRunning = true;
			while(stillOneRunning){
				stillOneRunning = false;
				counter = 0;
				for(unsigned int i = 0; i < amountOfClasses(); ++i){
					if(m_isClassUsed[i]){
						if(!packages[i]->isTaskFinished()){
							stillOneRunning = true;
//							printOnScreen("Package: " << i << ", is still running");
						}else{ // task is finished
							if(stateOfIvms[i] == 1 && m_ivms[i]->getKernelType() == IVM::GAUSS){
//								printOnScreen("Add " << i << " to retrain");
								// add to retrain
//								packagesForRetrain.push_back(new InformationPackage(InformationPackage::IVM_RETRAIN, packages[i]->correctlyClassified(), m_storage.size()));
//								groupForRetraining.add_thread(new boost::thread(boost::bind(&IVMMultiBinary::retrainIvmIfNeeded, this, packagesForRetrain.back(), i)));
								stateOfIvms[i] = 2;
							}
						}
						const int c = packages[i]->amountOfTrainingStepsPerformed();
						counter += c;
					}
				}
				InLinePercentageFiller::printLineWithRestTimeBasedOnMaxTime(counter, false);
				usleep(0.15 * 1e6);
			}
			counter = 0;
			for(unsigned int i = 0; i < amountOfClasses(); ++i){
				if(m_isClassUsed[i]){
					counter += packages[i]->amountOfTrainingStepsPerformed();
				}
			}
			InLinePercentageFiller::printLineWithRestTimeBasedOnMaxTime(counter, true);
			group.join_all();
			for(std::list<InformationPackage*>::const_iterator it = packagesForRetrain.begin(); it != packagesForRetrain.end(); ++it){
				(*it)->abortTraing(); // aborts training otherwise it will go on forever
			}
			for(unsigned int i = 0; i < amountOfClasses(); ++i){
				if(m_isClassUsed[i]){
					ThreadMaster::threadHasFinished(packages[i]);
					delete packages[i];
				}
			}
			groupForRetraining.join_all(); // to get a little bit of time until we wait on the finished training
			for(std::list<InformationPackage*>::const_iterator it = packagesForRetrain.begin(); it != packagesForRetrain.end(); ++it){
				ThreadMaster::threadHasFinished(*it);
				delete *it;
			}
//		}else{
//			const bool fitParams = CommandSettings::get_samplingAndTraining();
//			if(fitParams){
//				InLinePercentageFiller::setActMaxTime(durationOfWholeTraining);
//			}else{
//				InLinePercentageFiller::setActMax(amountOfClasses() + 1);
//			}
//			// initial start of nrOfParallel threads
//			int runningCounter = 0;
//			int counterForClass = 0;
//			for(; counterForClass < nrOfParallel; ++counterForClass){
//				group.add_thread(new boost::thread(boost::bind(&IVMMultiBinary::trainInParallel, this, counterForClass, durationOfTraining)));
//				++runningCounter;
//			}
//			unsigned int counter = 0;
//			while(true){
//				counter = 0;
//				unsigned int finished = 0;
//				bool stillOneRunning = false;
//				for(unsigned int i = 0; i < amountOfClasses(); ++i){
//					if(stillWorking[i]){
//						const int c = m_ivms[i]->getSampleCounter();
//						if(c > -1){
//							stillOneRunning = true;
//							counter += c;
//							counterRes[i] = c;
//						}else{
//							stillWorking[i] = false;
//							--runningCounter;
//							counter += counterRes[i];
//						}
//					}else{
//						++finished;
//						counter += counterRes[i];
//					}
//				}
//				if(!stillOneRunning){
//					break;
//				}
//				if(fitParams){
//					InLinePercentageFiller::printLineWithRestTimeBasedOnMaxTime(counter, false);
//				}else{
//					InLinePercentageFiller::setActValueAndPrintLine(finished);
//				}
//				if(runningCounter < nrOfParallel && counterForClass < amountOfClasses()){
//					group.add_thread(new boost::thread(boost::bind(&IVMMultiBinary::trainInParallel, this, counterForClass, durationOfTraining)));
//					++counterForClass;
//					++runningCounter;
//				}
//				usleep(0.1 * 1e6);
//			}
//			counter = 0;
//			for(unsigned int i = 0; i < amountOfClasses(); ++i){
//				counter += m_ivms[i]->getSampleCounter();
//			}
//			if(fitParams){
//				InLinePercentageFiller::printLineWithRestTimeBasedOnMaxTime(counter, true);
//			}else{
//				InLinePercentageFiller::setActValueAndPrintLine(amountOfClasses());
//			}
//			group.join_all();
//		}
			m_firstTraining = false;
	}
	int amountOfCorrect = 0;
	std::vector<int> counterClass(amountOfClasses(), 0);
	for(unsigned int i = 0; i < m_storage.size(); ++i){ // always test the whole data set
		const int correctLabel = m_storage[i]->getLabel();
		if(correctLabel == predict(*m_storage[i])){
			++m_correctAmountForTrainingDataForClasses[correctLabel];
			++amountOfCorrect;
		}
		++counterClass[correctLabel];
	}
	for(unsigned int i = 0; i < amountOfClasses(); ++i){
		m_correctAmountForTrainingDataForClasses[i] /= (double) counterClass[i];
	}
	m_correctAmountForTrainingData = amountOfCorrect / (double) m_storage.size();
}

void IVMMultiBinary::retrainAllIvmsIfNeeded(InformationPackage* wholePackage){
	if(wholePackage != nullptr && !m_firstTraining){ // should not be called on untrained ivms
		ThreadMaster::appendThreadToList(wholePackage);
		wholePackage->wait();
		for(unsigned int i = 0; i < amountOfClasses(); ++i){
			if(m_isClassUsed[i]){
				m_packages.push_back(new InformationPackage(InformationPackage::IVM_RETRAIN, m_correctAmountForTrainingDataForClasses[i], m_storage.size()));
				m_group.add_thread(new boost::thread(boost::bind(&IVMMultiBinary::retrainIvmIfNeeded, this, m_packages.back(), i)));
			}
		}
		wholePackage->finishedTask();
	}
}

void IVMMultiBinary::retrainIvmIfNeeded(InformationPackage* package, const int iClassNr){
	if(package != nullptr){
		if(m_isClassUsed[iClassNr]){
			m_ivms[iClassNr]->setInformationPackage(package);
			package->setStandartInformation("Ivm retraining for class: " + number2String(iClassNr));
			ThreadMaster::appendThreadToList(package); // wait is performed in the ivm->train()
			m_ivms[iClassNr]->train(true, 0, true); // endless training, which can be pausend and aborted inside of the ivm training
		}
		package->finishedTask();
	}
}

void IVMMultiBinary::initInParallel(const int startOfKernel, const int endOfKernel, Eigen::MatrixXd* differenceMatrix){
	GaussianKernel::calcDifferenceMatrix(startOfKernel, endOfKernel, *differenceMatrix, m_storage);
}

void IVMMultiBinary::trainInParallel(const int usedIvm, const double trainTime, InformationPackage* package){
	m_ivms[usedIvm]->setInformationPackage(package);
	package->setStandartInformation("Ivm training for class: " + number2String(usedIvm));
	ThreadMaster::appendThreadToList(package);
	m_ivms[usedIvm]->setKernelSeed((usedIvm + 1) * 459486);
	const bool ret = m_ivms[usedIvm]->train(true, 1); // package(task is finished) inside the binary ivm training!
//	m_ivms[usedIvm]->getKernel().setHyperParams(
//			Settings::getDirectDoubleValue("KernelParam.len"),
//			Settings::getDirectDoubleValue("KernelParam.fNoise"),
//			Settings::getDirectDoubleValue("KernelParam.sNoise"));
//	const bool ret = m_ivms[usedIvm]->train(false,1);
	if(!ret){
		printError("The ivm: " << usedIvm << ", could not be trained!");
	}
//	static boost::mutex mutex;
//	mutex.lock();
//	DataPoint p(2);
//	p << 0,0;
//	std::cout << "Used: " << usedIvm << std::endl;
//	m_ivms[usedIvm]->predict(p);
//	mutex.unlock();
}

void IVMMultiBinary::predict(const DataPoint& point, std::vector<double>& probabilities) const{
	probabilities.resize(amountOfClasses());
	for(unsigned int i = 0; i < amountOfClasses(); ++i){
		if(m_isClassUsed[i]){
			probabilities[i] = m_ivms[i]->predict(point);
		}
	}
}

int IVMMultiBinary::predict(const DataPoint& point) const{
	std::vector<double> probs(amountOfClasses(),0.);
	for(unsigned int i = 0; i < amountOfClasses(); ++i){
		if(m_isClassUsed[i]){
			probs[i] = m_ivms[i]->predict(point);
		}
	}
	return m_classOfIVMs[std::distance(probs.cbegin(), std::max_element(probs.cbegin(), probs.cend()))];
}

int IVMMultiBinary::predict(const ClassPoint& point) const{
	std::vector<double> probs(amountOfClasses(), 0.);
	for(unsigned int i = 0; i < amountOfClasses(); ++i){
		if(m_isClassUsed[i]){
			probs[i] = m_ivms[i]->predict(point);
		}
	}
	return m_classOfIVMs[std::distance(probs.cbegin(), std::max_element(probs.cbegin(), probs.cend()))];
}

void IVMMultiBinary::predictData(const Data& points, Labels& labels) const{
	labels.resize(points.size());
	for(unsigned int i = 0; i < points.size(); ++i){
		labels[i] = predict(*points[i]);
	}
}

void IVMMultiBinary::predictData(const Data& points, Labels& labels, std::vector< std::vector<double> >& probabilities) const{
	probabilities.resize(points.size());
	labels.resize(points.size());
	for(unsigned int i = 0; i < points.size(); ++i){
		probabilities[i].resize(amountOfClasses());
	}
	boost::thread_group group;
	std::vector<InformationPackage*> packages(amountOfClasses(), nullptr);
	for(unsigned int i = 0; i < amountOfClasses(); ++i){
		if(m_isClassUsed[i]){
			packages[i] = new InformationPackage(InformationPackage::IVM_PREDICT, 0, points.size());
			group.add_thread(new boost::thread(boost::bind(&IVMMultiBinary::predictDataInParallel, this, points, i, &probabilities, packages[i])));
		}
	}
	group.join_all();
	for(unsigned int i = 0; i < amountOfClasses(); ++i){
		ThreadMaster::threadHasFinished(packages[i]);
		delete packages[i];
	}
	for(unsigned int i = 0; i < points.size(); ++i){
//		double sum = 0;
//		for(unsigned int j = 0; j < amountOfClasses(); ++j){
//			sum += probabilities[i][j];
//		}
//		if(sum > 0.0){
		double highestValue;
		int highestArg;
		for(unsigned int j = 0; j < amountOfClasses(); ++j){
			//				probabilities[i][j] /= sum;
			if(probabilities[i][j] > highestValue){
				highestValue = probabilities[i][j];
				highestArg = j;
			}
		}
		labels[i] = highestArg;
		//			}
	}
}

void IVMMultiBinary::predictDataInParallel(const Data& points, const int usedIvm, std::vector< std::vector<double> >* probabilities, InformationPackage* package) const{
	package->setStandartInformation("Thread for ivm: " + number2String(usedIvm));
	ThreadMaster::appendThreadToList(package);
	package->wait();
	for(unsigned int i = 0; i < points.size(); ++i){
		(*probabilities)[i][usedIvm] = m_ivms[usedIvm]->predict(*points[i]);
		if(i % 5000 == 0 && i > 0){
			package->printLineToScreenForThisThread("5000 points predicted");
		}
		package->performedOneTrainingStep();
		if(package->shouldTrainingBePaused()){
			package->wait();
		}else if(package->shouldTrainingBeAborted()){
			printError("The prediciton can not be aborted!");
		}
	}
	package->finishedTask();
}

int IVMMultiBinary::amountOfClasses() const{
	return m_ivms.size();
}
