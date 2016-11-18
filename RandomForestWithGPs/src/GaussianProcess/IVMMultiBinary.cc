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
				for(unsigned int i = 0; i < m_ivms.size(); ++i){
					m_ivms[i] = new IVM();
				}
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
		double durationOfTraining = CommandSettings::get_samplingAndTraining();
		boost::thread_group group;
		const int nrOfParallel = boost::thread::hardware_concurrency();
		StopWatch sw;
		std::vector<int> counterRes(amountOfClasses(), 0);
		std::vector<bool> stillWorking(amountOfClasses(), true);
		double durationOfWholeTraining = durationOfTraining * amountOfClasses() / (double) nrOfParallel;
		if(amountOfClasses() <= nrOfParallel){
			const bool fitParams = CommandSettings::get_samplingAndTraining();
			if(fitParams){
				InLinePercentageFiller::setActMaxTime(durationOfTraining);
			}else{
				InLinePercentageFiller::setActMaxTime(5); // max time for sampling
			}
			for(unsigned int i = 0; i < amountOfClasses(); ++i){
				group.add_thread(new boost::thread(boost::bind(&IVMMultiBinary::trainInParallel, this, i, fitParams)));
			}
			unsigned int counter = 0;
			while(durationOfTraining > sw.elapsedSeconds()){
				counter = 0;
				bool stillOneRunning = false;
				for(unsigned int i = 0; i < amountOfClasses(); ++i){
					if(stillWorking[i]){
						const int c = m_ivms[i]->getSampleCounter();
						if(c > -1){
							stillOneRunning = true;
							counter += c;
							counterRes[i] = c;
						}else{
							stillWorking[i] = false;
						}
					}else{
						counter += counterRes[i];
					}
				}
				if(!stillOneRunning){
					break;
				}
				InLinePercentageFiller::printLineWithRestTimeBasedOnMaxTime(counter, false);
				usleep(0.15 * 1e6);
			}
			counter = 0;
			for(unsigned int i = 0; i < amountOfClasses(); ++i){
				counter += m_ivms[i]->getSampleCounter();
			}
			InLinePercentageFiller::printLineWithRestTimeBasedOnMaxTime(counter, true);
			group.join_all();
		}else{
			const bool fitParams = CommandSettings::get_samplingAndTraining();
			if(fitParams){
				InLinePercentageFiller::setActMaxTime(durationOfWholeTraining);
			}else{
				InLinePercentageFiller::setActMaxTime(5); // max time for sampling
			}
			// initial start of nrOfParallel threads
			int runningCounter = 0;
			int counterForClass = 0;
			for(; counterForClass < nrOfParallel; ++counterForClass){
				group.add_thread(new boost::thread(boost::bind(&IVMMultiBinary::trainInParallel, this, counterForClass, fitParams)));
				++runningCounter;
			}
			unsigned int counter = 0;
			while(true){
				counter = 0;
				bool stillOneRunning = false;
				for(unsigned int i = 0; i < amountOfClasses(); ++i){
					if(stillWorking[i]){
						const int c = m_ivms[i]->getSampleCounter();
						if(c > -1){
							stillOneRunning = true;
							counter += c;
							counterRes[i] = c;
						}else{
							stillWorking[i] = false;
							--runningCounter;
						}
					}else{
						counter += counterRes[i];
					}
				}
				if(!stillOneRunning){
					break;
				}
				InLinePercentageFiller::printLineWithRestTimeBasedOnMaxTime(counter, false);
				if(runningCounter < nrOfParallel - 1 && counterForClass < amountOfClasses()){
					group.add_thread(new boost::thread(boost::bind(&IVMMultiBinary::trainInParallel, this, counterForClass, fitParams)));
					++counterForClass;
					++runningCounter;
				}
				usleep(0.1 * 1e6);
			}
			counter = 0;
			for(unsigned int i = 0; i < amountOfClasses(); ++i){
				counter += m_ivms[i]->getSampleCounter();
			}
			InLinePercentageFiller::printLineWithRestTimeBasedOnMaxTime(counter, true);
			group.join_all();
		}
		m_firstTraining = false;
	}
}

void IVMMultiBinary::trainInParallel(const int usedIvm, const bool fitParams){
	Eigen::Vector2i usedClasses;
	usedClasses << m_classOfIVMs[usedIvm], -1;
	m_ivms[usedIvm]->init(m_storage.storage(),
			m_numberOfInducingPointsPerIVM, usedClasses, m_doEpUpdate);
	m_ivms[usedIvm]->getKernel().setSeed(usedIvm * 1389293);
	m_ivms[usedIvm]->train(fitParams);
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
	const int nrOfParallel = boost::thread::hardware_concurrency();
	probabilities.resize(points.size());
	labels.resize(points.size());
	for(unsigned int i = 0; i < points.size(); ++i){
		probabilities[i].resize(amountOfClasses());
	}
	if(amountOfClasses() <= nrOfParallel){
		boost::thread_group group;
		for(unsigned int i = 0; i < amountOfClasses(); ++i){
			group.add_thread(new boost::thread(boost::bind(&IVMMultiBinary::predictDataInParallel, this, points, i, &probabilities)));
		}
		group.join_all();
		for(unsigned int i = 0; i < points.size(); ++i){
//			double sum = 0;
//			for(unsigned int j = 0; j < amountOfClasses(); ++j){
//				sum += probabilities[i][j];
//			}
//			if(sum > 0.0){
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
	}else{
		for(unsigned int i = 0; i < points.size(); ++i){
			probabilities[i].resize(amountOfClasses());
			double sum = 0;
			for(unsigned int j = 0; j < amountOfClasses(); ++j){
				probabilities[i][j] = m_ivms[j]->predict(*points[i]);
				sum += probabilities[i][j];
			}
			if(sum > 1.0){
				double highestValue;
				int highestArg;
				for(unsigned int j = 0; j < amountOfClasses(); ++j){
					probabilities[i][j] /= sum;
					if(probabilities[i][j] > highestValue){
						highestValue = probabilities[i][j];
						highestArg = j;
					}
				}
				labels[i] = highestArg;
			}
		}
	}
}

void IVMMultiBinary::predictDataInParallel(const Data& points, const int usedIvm, std::vector< std::vector<double> >* probabilities) const{
	for(unsigned int i = 0; i < points.size(); ++i){
		(*probabilities)[i][usedIvm] = m_ivms[usedIvm]->predict(*points[i]);
	}
}

int IVMMultiBinary::amountOfClasses() const{
	return m_ivms.size();
}
