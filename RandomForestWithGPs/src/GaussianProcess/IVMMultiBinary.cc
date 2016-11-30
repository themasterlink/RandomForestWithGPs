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
		m_firstTraining(true){
	m_storage.attach(this);
}

void IVMMultiBinary::update(Subject* caller, unsigned int event){
	if(caller->classType() == m_storage.classType()){
		if(event == OnlineStorage<ClassPoint*>::APPENDBLOCK){
			// assumption that the correct OnlineStorage is callig and not a false one -> no check
			const unsigned int lastUpdateIndex = m_storage.getLastUpdateIndex();
			if(!m_init && lastUpdateIndex == 0){
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
				m_ivms.resize(classCounter.size());
				const bool calcDifferenceMatrixAlone = false;
				for(unsigned int i = 0; i < m_ivms.size(); ++i){
					m_ivms[i] = new IVM();
					Eigen::Vector2i usedClasses;
					usedClasses << m_classOfIVMs[i], -1;
					m_ivms[i]->init(m_storage.storage(),
							m_numberOfInducingPointsPerIVM, usedClasses,
							m_doEpUpdate, calcDifferenceMatrixAlone);
				}
				const int nrOfParallel = boost::thread::hardware_concurrency();
				const int size = (m_storage.size() * m_storage.size() + m_storage.size()) / 2;
				const int sizeOfPart =  size / nrOfParallel;
				Eigen::MatrixXd* differenceMatrix = new Eigen::MatrixXd(m_storage.size(), m_storage.size());
				boost::thread_group group;
				for(unsigned int i = 0; i < nrOfParallel; ++i){
					int start = sizeOfPart * i;
					int end = (i + 1 != nrOfParallel) ? sizeOfPart * (i + 1) : size;
					group.add_thread(new boost::thread(boost::bind(&IVMMultiBinary::initInParallel, this, i, start, end, differenceMatrix)));
				}
				group.join_all();
				m_init = true;
				train();
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
}

void IVMMultiBinary::train(){
	if(!m_firstTraining){
		printError("Not implemented yet!");
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
		const int nrOfParallel = boost::thread::hardware_concurrency();
		StopWatch sw;
		std::vector<int> counterRes(amountOfClasses(), 0);
		std::vector<bool> stillWorking(amountOfClasses(), true);
		std::vector<InformationPackage*> packages(amountOfClasses());

		for(unsigned int i = 0; i < amountOfClasses(); ++i){
			const double correctlyClassified = 0.; // TODO in future this value can be determined before hand!
			packages[i] = new InformationPackage(InformationPackage::IVM_TRAIN, correctlyClassified, m_storage.size());
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
			for(unsigned int i = 0; i < amountOfClasses(); ++i){
				group.add_thread(new boost::thread(boost::bind(&IVMMultiBinary::trainInParallel, this, i, durationOfTraining, packages[i])));
			}
			unsigned int counter = 0;
			while(durationOfWholeTraining > sw.elapsedSeconds()){
				counter = 0;
				bool stillOneRunning = false;
				for(unsigned int i = 0; i < amountOfClasses(); ++i){
					if(!packages[i]->isTaskFinished()){
						stillOneRunning = true;
					}
					const int c = packages[i]->amountOfTrainingStepsPerformed();
					counter += c;
				}
				if(!stillOneRunning){
					break;
				}
				InLinePercentageFiller::printLineWithRestTimeBasedOnMaxTime(counter, false);
				usleep(0.15 * 1e6);
			}
			counter = 0;
			for(unsigned int i = 0; i < amountOfClasses(); ++i){
				counter += packages[i]->amountOfTrainingStepsPerformed();
			}
			InLinePercentageFiller::printLineWithRestTimeBasedOnMaxTime(counter, true);
			group.join_all();
			for(unsigned int i = 0; i < amountOfClasses(); ++i){
				ThreadMaster::threadHasFinished(packages[i]);
				delete packages[i];
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
}

void IVMMultiBinary::initInParallel(const int usedIvm, const int startOfKernel, const int endOfKernel, Eigen::MatrixXd* differenceMatrix){
	m_ivms[usedIvm]->getKernel().calcDifferenceMatrix(startOfKernel, endOfKernel, differenceMatrix);
}

void IVMMultiBinary::trainInParallel(const int usedIvm, const double trainTime, InformationPackage* package){
	m_ivms[usedIvm]->setInformationPackage(package);
	package->setStandartInformation("Ivm training for class: " + number2String(usedIvm));
	ThreadMaster::appendThreadToList(package);
	m_ivms[usedIvm]->getKernel().setSeed((usedIvm + 1) * 459486);
	const bool ret = m_ivms[usedIvm]->train(trainTime,1);
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

int IVMMultiBinary::predict(const DataPoint& point) const{
	std::vector<double> probs(amountOfClasses());
	for(unsigned int i = 0; i < amountOfClasses(); ++i){
		probs[i] = m_ivms[i]->predict(point);
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
		packages[i] = new InformationPackage(InformationPackage::IVM_PREDICT, 0, points.size());
		group.add_thread(new boost::thread(boost::bind(&IVMMultiBinary::predictDataInParallel, this, points, i, &probabilities, packages[i])));
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
		if(i % 5000 == 0){
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
