/*
 * IVMMultiBinary.cc
 *
 *  Created on: 17.11.2016
 *      Author: Max
 */

#include "IVMMultiBinary.h"
#include "../Base/CommandSettings.h"
#include "../Data/DataBinaryWriter.h"
#include "../Utility/Util.h"

IVMMultiBinary::IVMMultiBinary(OnlineStorage<ClassPoint*>& storage,
		const unsigned int numberOfInducingPointsPerIVM,
		const bool doEPUpdate, const int orfClassLabel):
		m_storage(storage),
		m_numberOfInducingPointsPerIVM(numberOfInducingPointsPerIVM),
		m_doEpUpdate(doEPUpdate),
		m_init(false),
		m_firstTraining(true),
		m_correctAmountForTrainingData(0),
		m_orfForKernel(nullptr),
		m_orfClassLabel(orfClassLabel),
		m_amountOfAllClasses(0){
	m_storage.attach(this);
}

void IVMMultiBinary::update(Subject* caller, unsigned int event){
	if(caller->classType() == m_storage.classType()){
		if(event == OnlineStorage<ClassPoint*>::APPENDBLOCK){
			if(m_amountOfAllClasses == 0){
				m_amountOfAllClasses = ClassKnowledge::amountOfClasses();
			}else if(m_amountOfAllClasses != ClassKnowledge::amountOfClasses()){
				printError("The amount of classes should not change during execution!");
			}
			// assumption that the correct OnlineStorage is callig and not a false one -> no check
			const unsigned int lastUpdateIndex = m_storage.getLastUpdateIndex();
			if(!m_init && lastUpdateIndex == 0){
				// to find out the amount of used classes in this ivm look at the data
				std::map<unsigned int, unsigned int> classCounter;
				for(ClassData::const_iterator it = m_storage.begin(); it != m_storage.end(); ++it){
					const unsigned int label = (**it).getLabel();
					std::map<unsigned int, unsigned int>::iterator itClass = classCounter.find(label);
					if(itClass == classCounter.end()){
						classCounter.insert(std::pair<unsigned int, unsigned int>(label, 1u));
					}else{
						itClass->second += 1u;
					}
				}
				unsigned int amountOfUsedClasses = 0u;
				const unsigned int minimumNeededAmountOfElements = 0.75 * m_numberOfInducingPointsPerIVM;
				for (std::map<unsigned int, unsigned int>::const_iterator it = classCounter.begin(); it != classCounter.end(); ++it){
					m_isClassUsed.push_back(it->second > minimumNeededAmountOfElements);
					m_classOfIVMs.push_back(it->first); // guarentees that classOfIvms and isClassUsed have the same mapping
					if(m_isClassUsed.back()){
						++amountOfUsedClasses;
					}
				}
				m_generalClassesToIVMs.resize(m_amountOfAllClasses);
				std::fill(m_generalClassesToIVMs.begin(), m_generalClassesToIVMs.end(), UNDEF_CLASS_LABEL);
				for(unsigned int i = 0; i < m_classOfIVMs.size(); ++i){
					m_generalClassesToIVMs[m_classOfIVMs[i]] = i;
				}
				m_isClassUsed.resize(amountOfClasses());
				m_ivms.resize(amountOfClasses());
				m_correctAmountForTrainingDataForClasses.resize(m_amountOfAllClasses);
				std::fill(m_correctAmountForTrainingDataForClasses.begin(), m_correctAmountForTrainingDataForClasses.end(), 0.0);
				if(amountOfUsedClasses > 0){
					const bool calcDifferenceMatrixAlone = false;
					int kernelType = 0;
					Settings::getValue("IVM.kernelType", kernelType);
					if(kernelType == 1){ // RF
						int maxDepth, samplingAmount;
						Settings::getValue("RandomForestKernel.maxDepth", maxDepth);
						Settings::getValue("RandomForestKernel.samplingAmount", samplingAmount);
						m_orfForKernel = new OnlineRandomForest(m_storage, maxDepth, amountOfClasses());
						// train the trees before the ivms are used
						m_orfForKernel->setDesiredAmountOfTrees(samplingAmount);
						m_orfForKernel->update(&m_storage, OnlineStorage<ClassPoint*>::APPENDBLOCK);
					}
					for(unsigned int i = 0; i < amountOfClasses(); ++i){
						if(m_isClassUsed[i]){
							m_ivms[i] = new IVM(m_storage, true);
							if(m_ivms[i]->getKernelType() == IVM::RF){
								m_ivms[i]->setOnlineRandomForest(m_orfForKernel);
							}
							Eigen::Vector2i usedClasses;
							usedClasses << m_classOfIVMs[i], UNDEF_CLASS_LABEL;
							m_ivms[i]->init(m_numberOfInducingPointsPerIVM, usedClasses,
									m_doEpUpdate, calcDifferenceMatrixAlone);
						}
					}
					const unsigned int nrOfParallel = (unsigned int) boost::thread::hardware_concurrency();
					const unsigned int size = (m_storage.size() * m_storage.size() + m_storage.size()) / 2;
					const unsigned int sizeOfPart =  size / nrOfParallel;
					if(kernelType == 0 && amountOfUsedClasses > 0){ // GAUSS, calc the kernel matrix
						Eigen::MatrixXd* differenceMatrix = new Eigen::MatrixXd(m_storage.size(), m_storage.size());
						boost::thread_group* group = new boost::thread_group();
						std::vector<InformationPackage*> packages(nrOfParallel, nullptr);

						for(unsigned int i = 0; i < nrOfParallel; ++i){
							int start = sizeOfPart * i;
							int end = (i + 1 != nrOfParallel) ? sizeOfPart * (i + 1) : size;
							packages[i] = new InformationPackage(InformationPackage::IVM_INIT_DIFFERENCE_MATRIX, 0, sizeOfPart);
							group->add_thread(new boost::thread(boost::bind(&IVMMultiBinary::initInParallel, this, start, end, differenceMatrix, packages[i])));
						}
						group->join_all();
						for(unsigned int i = 0; i < nrOfParallel; ++i){
							ThreadMaster::threadHasFinished(packages[i]);
							SAVE_DELETE(packages[i]);
						}
						for(unsigned int i = 0; i < amountOfClasses(); ++i){
							if(m_isClassUsed[i]){
								m_ivms[i]->getGaussianKernel()->setDifferenceMatrix(differenceMatrix);
							}
						}
						SAVE_DELETE(group);
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
	SAVE_DELETE(m_orfForKernel);
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
				SAVE_DELETE(*it);
			}
			m_packages.clear();
		}
		InformationPackage* wholePackage = new InformationPackage(InformationPackage::IVM_MULTI_UPDATE, m_correctAmountForTrainingData, m_storage.size());
		m_group.add_thread(new boost::thread(boost::bind(&IVMMultiBinary::retrainAllIvmsIfNeeded, this, wholePackage)));
	}else{
//		printOnScreen("First Value: " << m_storage[0]->getLabel() << " with: " << m_storage[0]->transpose());
//		if(!Settings::getDirectBoolValue("IVM.hasLengthMoreThanParam")){
//			std::vector<double> means = {Settings::getDirectDoubleValue("KernelParam.lenMean"),
//					Settings::getDirectDoubleValue("KernelParam.fNoiseMean"),
//					Settings::getDirectDoubleValue("KernelParam.sNoiseMean")};
//			std::vector<double> sds = {Settings::getDirectDoubleValue("KernelParam.lenVar"),
//					Settings::getDirectDoubleValue("KernelParam.fNoiseVar"),
//					Settings::getDirectDoubleValue("KernelParam.sNoiseVar")};
//			std::stringstream stringStream;
//			stringStream << "Used means: ";
//			for(unsigned int i = 0; i < 3; ++i){
//				stringStream << means[i] << ", ";
//			}
//			stringStream << "used sds:";
//			for(unsigned int i = 0; i < 3; ++i){
//				stringStream << sds[i];
//				if(i != 2)
//					stringStream << ", ";
//			}
//			printOnScreen(stringStream.str());
//		}
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
				packages[i] = new InformationPackage(InformationPackage::IVM_TRAIN, m_correctAmountForTrainingDataForClasses[m_classOfIVMs[i]], m_storage.size());
			}
		}
		double durationOfWholeTraining = durationOfTraining;
		if(amountOfClasses() > nrOfParallel){
			durationOfWholeTraining *= ceil(amountOfClasses() / (double) nrOfParallel);
		}
//		if(amountOfClasses() <= nrOfParallel){
		if(m_orfClassLabel == UNDEF_CLASS_LABEL){

			const bool fitParams = CommandSettings::get_samplingAndTraining();
			if(fitParams){
				InLinePercentageFiller::setActMaxTime(durationOfWholeTraining);
			}else{
				InLinePercentageFiller::setActMaxTime(5); // max time for sampling
			}
		}

		// write to file
		DataBinaryWriter::toFile(m_storage.storage(), "dataFor_" + number2String(m_orfClassLabel) + ".binary");

		std::list<InformationPackage*> packagesForRetrain;
		std::vector<unsigned char> stateOfIvms(amountOfClasses(), 0);
		for(unsigned int i = 0; i < amountOfClasses(); ++i){
			if(m_isClassUsed[i]){
				group.add_thread(new boost::thread(boost::bind(&IVMMultiBinary::trainInParallel, this, m_ivms[i], i, durationOfTraining, packages[i])));
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
							//								printOnScreen("Add " <<  m_ivms[i]->getClassName() << " to retrain");
							//								// add to retrain
							//								packagesForRetrain.push_back(new InformationPackage(InformationPackage::IVM_RETRAIN, packages[i]->correctlyClassified(), m_storage.size()));
							//								groupForRetraining.add_thread(new boost::thread(boost::bind(&IVMMultiBinary::retrainIvmIfNeeded, this, packagesForRetrain.back(), i)));
							stateOfIvms[i] = 2;
						}
					}
					const int c = packages[i]->amountOfTrainingStepsPerformed();
					counter += c;
				}
			}
			if(m_orfClassLabel == UNDEF_CLASS_LABEL){
				InLinePercentageFiller::printLineWithRestTimeBasedOnMaxTime(counter, false);
			}
			usleep(0.15 * 1e6);
		}
		counter = 0;
		for(unsigned int i = 0; i < amountOfClasses(); ++i){
			if(m_isClassUsed[i]){
				counter += packages[i]->amountOfTrainingStepsPerformed();
			}
		}
		if(m_orfClassLabel == UNDEF_CLASS_LABEL){
			InLinePercentageFiller::printLineWithRestTimeBasedOnMaxTime(counter, true);
		}
		group.join_all();
		for(std::list<InformationPackage*>::const_iterator it = packagesForRetrain.begin(); it != packagesForRetrain.end(); ++it){
			(*it)->abortTraing(); // aborts training otherwise it will go on forever
		}
		for(unsigned int i = 0; i < amountOfClasses(); ++i){
			if(m_isClassUsed[i]){
				ThreadMaster::threadHasFinished(packages[i]);
				SAVE_DELETE(packages[i]);
			}
		}
		groupForRetraining.join_all(); // to get a little bit of time until we wait on the finished training
		for(std::list<InformationPackage*>::iterator it = packagesForRetrain.begin(); it != packagesForRetrain.end(); ++it){
			ThreadMaster::threadHasFinished(*it);
			SAVE_DELETE(*it);
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
	printOnScreen("Finish training of IVM Multi Binary");
	int amountOfCorrect = 0;
	std::vector<int> counterClass(m_amountOfAllClasses, 0);
	for(unsigned int i = 0; i < m_storage.size(); ++i){ // always test the whole data set
		const int correctLabel = m_storage[i]->getLabel();
		if(correctLabel == predict(*m_storage[i])){
			++m_correctAmountForTrainingDataForClasses[correctLabel];
			++amountOfCorrect;
		}
		++counterClass[correctLabel];
	}
	for(unsigned int i = 0; i < m_amountOfAllClasses; ++i){
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
				m_packages.push_back(new InformationPackage(InformationPackage::IVM_RETRAIN, m_correctAmountForTrainingDataForClasses[m_classOfIVMs[i]], m_storage.size()));
				m_group.add_thread(new boost::thread(boost::bind(&IVMMultiBinary::retrainIvmIfNeeded, this, m_ivms[i], m_packages.back(), i)));
			}
		}
		wholePackage->finishedTask();
	}
}

void IVMMultiBinary::retrainIvmIfNeeded(IVM* ivm, InformationPackage* package, const int iClassNr){
	if(package != nullptr && ivm != nullptr){
		if(m_isClassUsed[iClassNr]){
			ivm->setInformationPackage(package);
			package->setStandartInformation("Ivm retraining for class: " + ivm->getClassName());
			ThreadMaster::appendThreadToList(package); // wait is performed in the ivm->train()
			ivm->train(true, 0, true); // endless training, which can be pausend and aborted inside of the ivm training
		}
		package->finishedTask();
	}
}

void IVMMultiBinary::initInParallel(const int startOfKernel, const int endOfKernel, Eigen::MatrixXd* differenceMatrix, InformationPackage* package){
	ThreadMaster::appendThreadToList(package);
	package->setStandartInformation("Calc of difference matrix from: " + number2String(startOfKernel) + " to: " +  number2String(endOfKernel) + (m_orfClassLabel != UNDEF_CLASS_LABEL ? ", for orf class: " + number2String(m_orfClassLabel) : ""));
	package->wait();
	GaussianKernel::calcDifferenceMatrix(startOfKernel, endOfKernel, *differenceMatrix, m_storage, package);
	package->finishedTask();
}

void IVMMultiBinary::trainInParallel(IVM* ivm, const int usedIvm, const double trainTime, InformationPackage* package){
	if(ivm != nullptr){
		ivm->setInformationPackage(package);
		ivm->setClassName(m_orfClassLabel);
		package->setStandartInformation("Ivm training for class: " + ivm->getClassName());
		ThreadMaster::appendThreadToList(package);
		ivm->setKernelSeed((m_classOfIVMs[usedIvm] + 1) * 459486); // with: m_classOfIVMs it depends on the label of the ivm
		const bool ret = ivm->train(true, 1); // package(task is finished) inside the binary ivm training!
		//	ivm->getKernel().setHyperParams(
		//			Settings::getDirectDoubleValue("KernelParam.len"),
		//			Settings::getDirectDoubleValue("KernelParam.fNoise"),
		//			Settings::getDirectDoubleValue("KernelParam.sNoise"));
		//	const bool ret = ivm->train(false,1);
		m_isClassUsed[usedIvm] = ret;
		if(!ret){
			printError("The ivm: " << ivm->getClassName() << ", could not be trained!");
		}
	}
//	static boost::mutex mutex;
//	mutex.lock();
//	DataPoint p(2);
//	p << 0,0;
//	std::cout << "Used: " << usedIvm << std::endl;
//	ivm->predict(p);
//	mutex.unlock();
}

void IVMMultiBinary::predict(const DataPoint& point, std::vector<double>& probabilities) const{
	probabilities.resize(m_amountOfAllClasses);
	for(unsigned int i = 0; i < amountOfClasses(); ++i){
		if(m_isClassUsed[i]){
			probabilities[m_classOfIVMs[i]] = m_ivms[i]->predict(point);
		}
	}
}

int IVMMultiBinary::predict(const DataPoint& point) const{
	std::vector<double> probs(m_amountOfAllClasses, 0.);
	predict(point, probs);
	return std::distance(probs.cbegin(), std::max_element(probs.cbegin(), probs.cend()));
}

int IVMMultiBinary::predict(const ClassPoint& point) const{
	std::vector<double> probs(m_amountOfAllClasses, 0.);
	for(unsigned int i = 0; i < amountOfClasses(); ++i){
		if(m_isClassUsed[i]){
			probs[m_classOfIVMs[i]] = m_ivms[i]->predict(point);
		}
	}
	return std::distance(probs.cbegin(), std::max_element(probs.cbegin(), probs.cend()));
}

void IVMMultiBinary::predictData(const Data& points, Labels& labels) const{
	std::vector< std::vector<double> > probs;
	predictData(points, labels, probs);
}

void IVMMultiBinary::predictData(const ClassData& points, Labels& labels) const{
	std::vector< std::vector<double> > probs;
	predictData(points, labels, probs);
}

void IVMMultiBinary::predictData(const Data& points, Labels& labels, std::vector< std::vector<double> >& probabilities) const{
	probabilities.resize(points.size());
	labels.resize(points.size());
	for(unsigned int i = 0; i < points.size(); ++i){
		probabilities[i].resize(m_amountOfAllClasses);
	}
	boost::thread_group group;
	std::vector<InformationPackage*> packages(amountOfClasses(), nullptr);
	for(unsigned int i = 0; i < amountOfClasses(); ++i){
		if(m_isClassUsed[i]){
			packages[i] = new InformationPackage(InformationPackage::IVM_PREDICT, 0, points.size());
			group.add_thread(new boost::thread(boost::bind(&IVMMultiBinary::predictDataInParallel, this, m_ivms[i], points, i, &probabilities, packages[i])));
		}
	}
	group.join_all();
	for(unsigned int i = 0; i < amountOfClasses(); ++i){
		ThreadMaster::threadHasFinished(packages[i]);
		SAVE_DELETE(packages[i]);
	}
	for(unsigned int i = 0; i < points.size(); ++i){
		double highestValue = 0.;
		int highestArg;
		for(unsigned int j = 0; j < m_amountOfAllClasses; ++j){
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

void IVMMultiBinary::predictData(const ClassData& points, Labels& labels, std::vector< std::vector<double> >& probabilities) const{
	probabilities.resize(points.size());
	labels.resize(points.size());
	for(unsigned int i = 0; i < points.size(); ++i){
		probabilities[i].resize(m_amountOfAllClasses);
	}
	boost::thread_group group;
	std::vector<InformationPackage*> packages(amountOfClasses(), nullptr);
	for(unsigned int i = 0; i < amountOfClasses(); ++i){
		if(m_isClassUsed[i]){
			packages[i] = new InformationPackage(InformationPackage::IVM_PREDICT, 0, points.size());
			group.add_thread(new boost::thread(boost::bind(&IVMMultiBinary::predictClassDataInParallel, this, m_ivms[i], points, i, &probabilities, packages[i])));
		}
	}
	group.join_all();
	for(unsigned int i = 0; i < amountOfClasses(); ++i){
		ThreadMaster::threadHasFinished(packages[i]);
		SAVE_DELETE(packages[i]);
	}
	for(unsigned int i = 0; i < points.size(); ++i){
		double highestValue = 0.;
		int highestArg;
		for(unsigned int j = 0; j < m_amountOfAllClasses; ++j){
			if(probabilities[i][j] > highestValue){
				highestValue = probabilities[i][j];
				highestArg = j;
			}
		}
		labels[i] = highestArg;
	}
}

void IVMMultiBinary::predictDataInParallel(IVM* ivm, const Data& points, const int usedIvm, std::vector< std::vector<double> >* probabilities, InformationPackage* package) const{
	package->setStandartInformation("Thread for ivm: " + number2String(usedIvm));
	ThreadMaster::appendThreadToList(package);
	package->wait();
	const int percent10 = std::max((int) points.size() / 10, 10);
	const unsigned int usedClass = m_classOfIVMs[usedIvm]; // converts from the classes here to all classes
	for(unsigned int i = 0; i < points.size(); ++i){
		(*probabilities)[i][usedClass] = ivm->predict(*points[i]);
		if(i % percent10 == 0 && i > 0){
			printInPackageOnScreen(package, i / (double) percent10 * 10 << " %% points done");
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

void IVMMultiBinary::predictClassDataInParallel(IVM* ivm, const ClassData& points, const int usedIvm, std::vector< std::vector<double> >* probabilities, InformationPackage* package) const{
	package->setStandartInformation("Thread for ivm: " + number2String(usedIvm));
	ThreadMaster::appendThreadToList(package);
	package->wait();
	const int percent10 = std::max((int) points.size() / 10, 10);
	const unsigned int usedClass = m_classOfIVMs[usedIvm]; // converts from the classes here to all classes
	for(unsigned int i = 0; i < points.size(); ++i){
		(*probabilities)[i][usedClass] = ivm->predict(*points[i]);
		if(i % percent10 == 0 && i > 0){
			printInPackageOnScreen(package, i / (double) percent10 * 10 << " %% points done");
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


unsigned int IVMMultiBinary::amountOfClasses() const{
	return m_classOfIVMs.size();
}
