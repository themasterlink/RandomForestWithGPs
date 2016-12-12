/*
 * OnlineRandomForestIVMs.cc
 *
 *  Created on: 17.11.2016
 *      Author: Max
 */

#include "OnlineRandomForestIVMs.h"
#include "../Utility/ConfusionMatrixPrinter.h"
#include "../Data/ClassKnowledge.h"

OnlineRandomForestIVMs::OnlineRandomForestIVMs(OnlineStorage<ClassPoint*>& storage, const int maxDepth, const int amountOfUsedClasses):
	m_storage(storage),
	m_orf(storage, maxDepth, amountOfUsedClasses),
	m_ivms(amountOfUsedClasses, nullptr),
	m_onlineStoragesForIvms(amountOfUsedClasses, nullptr),
	m_amountOfUsedClasses(amountOfUsedClasses),
	m_firstTrainedDone(false){
	// removes orf to avoid that the update is called directly to the orf
	m_orf.getStorageRef().deattach(&m_orf);
	// instead call the update on the
	m_storage.attach(this);


}

OnlineRandomForestIVMs::~OnlineRandomForestIVMs() {
}

void OnlineRandomForestIVMs::update(Subject* caller, unsigned int event){
	switch(event){
	case OnlineStorage<ClassPoint*>::APPEND:{
		printError("This is not implemented yet!");
		break;
	}
	case OnlineStorage<ClassPoint*>::APPENDBLOCK:{
		update();
		break;
	}
	case OnlineStorage<ClassPoint*>::ERASE:{
		printError("This update type is not supported here!");
		break;
	}
	default: {
		printError("This update type is not supported here!");
		break;
	}
	}
}

void OnlineRandomForestIVMs::update(){
	if(!m_firstTrainedDone){
		m_orf.update(&m_storage, OnlineStorage<ClassPoint*>::APPENDBLOCK);
		std::list<int> predictedLabels;

		Eigen::MatrixXd conv = Eigen::MatrixXd::Zero(ClassKnowledge::amountOfClasses(), ClassKnowledge::amountOfClasses());
		for(OnlineStorage<ClassPoint*>::ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
			predictedLabels.push_back(m_orf.predict(**it));
			conv((*it)->getLabel(), predictedLabels.back()) += 1;
		}
		printOnScreen("Just ORFs:");
		ConfusionMatrixPrinter::print(conv);
		boost::thread_group group;
		for(unsigned int iClassNr = 0; iClassNr < amountOfClasses(); ++iClassNr){
			// for each class find all predicted values which should be considered in this class
			ClassData dataForPredictedClass;
			std::list<int>::const_iterator itPredictedLabel = predictedLabels.begin();
			std::vector<int> classCounter(amountOfClasses(), 0);
			for(OnlineStorage<ClassPoint*>::ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it, ++itPredictedLabel){
				if(*itPredictedLabel == iClassNr){
					dataForPredictedClass.push_back(*it);
					++classCounter[(*it)->getLabel()];
				}
			}
			const int sizeOfPointsForClass = dataForPredictedClass.size();
			printOnScreen("Size of class " << iClassNr << ": " << classCounter[iClassNr] << ", whole is: " << sizeOfPointsForClass);
			int nrOfInducingPoints = 40;
			Settings::getValue("IVM.nrOfInducingPoints", nrOfInducingPoints);
			if(sizeOfPointsForClass * 0.95 > classCounter[iClassNr] && nrOfInducingPoints + 100 < sizeOfPointsForClass){ // if less than 95 % of the points belong to the right class -> use ivms
				const bool doEpUpdate = false;
				group.add_thread(new boost::thread(boost::bind(&OnlineRandomForestIVMs::trainIvm, this, iClassNr, nrOfInducingPoints, doEpUpdate, dataForPredictedClass)));
				printOnScreen("Calc ivm" << iClassNr);
			}
		}
		group.join_all();
	}else{
		printError("Not implemented yet!");
	}
}

void OnlineRandomForestIVMs::trainIvm(const int usedIvm, const int nrOfInducingPoints, const bool doEpUpdate, ClassData& data){
	m_onlineStoragesForIvms[usedIvm] = new OnlineStorage<ClassPoint*>();
	m_ivms[usedIvm] = new IVMMultiBinary(*m_onlineStoragesForIvms[usedIvm], nrOfInducingPoints, doEpUpdate);
	m_onlineStoragesForIvms[usedIvm]->append(data); // this append will call the update of the ivm and will start the training
}

int OnlineRandomForestIVMs::predict(const DataPoint& point) const{
	const int label = m_orf.predict(point);
	if(label > -1){
		if(m_ivms[label] != nullptr){
			return m_ivms[label]->predict(point);
		}
		return label;
	}
	return -1;
}

int OnlineRandomForestIVMs::predict(const ClassPoint& point) const{
	const int label = m_orf.predict(point);
	if(label > -1){
		if(m_ivms[label] != nullptr){
			return m_ivms[label]->predict(point);
		}
		return label;
	}
	return -1;
}

void OnlineRandomForestIVMs::predictData(const Data& points, Labels& labels) const{
	labels.resize(points.size());
	int i = 0;
	for(DataConstIterator it = points.begin(); it != points.end(); ++it, ++i){
		labels[i] = predict(**it);
	}
}

void OnlineRandomForestIVMs::predictData(const ClassData& points, Labels& labels) const{
	labels.resize(points.size());
	int i = 0;
	for(ClassDataConstIterator it = points.begin(); it != points.end(); ++it, ++i){
		labels[i] = predict(**it);
	}
}

void OnlineRandomForestIVMs::predictData(const Data& points, Labels& labels, std::vector< std::vector<double> >& probabilities) const{
	m_orf.predictData(points, labels, probabilities);
	int i = 0;
	for(DataConstIterator it = points.begin(); it != points.end(); ++it, ++i){
		if(m_ivms[labels[i]] != nullptr){
			for(int j = 0; j < amountOfClasses(); ++j){
				probabilities[i][j] = 0;
			}
			m_ivms[labels[i]]->predict(**it, probabilities[i]);
		}
	}
}

int OnlineRandomForestIVMs::amountOfClasses() const{
	return m_amountOfUsedClasses;
}
