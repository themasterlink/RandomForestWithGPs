/*
 * OnlineRandomForestIVMs.cc
 *
 *  Created on: 17.11.2016
 *      Author: Max
 */

#include "OnlineRandomForestIVMs.h"
#include "../Utility/ConfusionMatrixPrinter.h"
#include "../Data/ClassKnowledge.h"
#include "../Base/Logger.h"

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
		OnlineStorage<ClassPoint*>* copyForORFs = new OnlineStorage<ClassPoint*>(m_storage);
		m_orf.update(copyForORFs, OnlineStorage<ClassPoint*>::APPENDBLOCK);
		m_orf.update();
		std::list<int> predictedLabels;
		unsigned int amountOfCorrect = 0;
		Eigen::MatrixXd conv = Eigen::MatrixXd::Zero(ClassKnowledge::amountOfClasses(), ClassKnowledge::amountOfClasses());
		for(OnlineStorage<ClassPoint*>::ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
			const unsigned int predictedLabel = m_orf.predict(**it);
			predictedLabels.push_back(predictedLabel);
			if(predictedLabel == (**it).getLabel()){
				++amountOfCorrect;
			}
			conv((*it)->getLabel(), predictedLabels.back()) += 1;
		}
		printOnScreen("Just ORFs:");
		ConfusionMatrixPrinter::print(conv);
		printOnScreen("Just ORFs correct: " << number2String(amountOfCorrect / (double) m_storage.size() * 100.0, 2));
		Logger::forcedWrite();
		boost::thread_group* group = new boost::thread_group();
		std::vector<ClassData*> datasForPredictedClasses(amountOfClasses(), nullptr);
		int nrOfInducingPoints = 40;
		Settings::getValue("IVM.nrOfInducingPoints", nrOfInducingPoints);
		for(unsigned int iClassNr = 0; iClassNr < amountOfClasses(); ++iClassNr){
			// for each class find all predicted values which should be considered in this class
			datasForPredictedClasses[iClassNr] = new ClassData();
			std::list<int>::const_iterator itPredictedLabel = predictedLabels.begin();
			std::vector<int> classCounter(amountOfClasses(), 0);
			for(OnlineStorage<ClassPoint*>::ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it, ++itPredictedLabel){
				if(*itPredictedLabel == iClassNr){
					datasForPredictedClasses[iClassNr]->push_back(*it);
					++classCounter[(*it)->getLabel()];
				}
			}
			const int sizeOfPointsForClass = datasForPredictedClasses[iClassNr]->size();
//			for(unsigned int iInnerClassNr = 0; iInnerClassNr < amountOfClasses(); ++iInnerClassNr){
//				printOnScreen("Size of class " << ClassKnowledge::getNameFor(iClassNr) << "_"<< ClassKnowledge::getNameFor(iInnerClassNr)<< ": " << classCounter[iInnerClassNr] << ", whole is: " << sizeOfPointsForClass);
				if(sizeOfPointsForClass * 0.95 > classCounter[iClassNr] && nrOfInducingPoints + 100 < sizeOfPointsForClass){ // if less than 95 % of the points belong to the right class -> use ivms
					const bool doEpUpdate = false;
					group->add_thread(new boost::thread(boost::bind(&OnlineRandomForestIVMs::trainIvm, this, iClassNr, nrOfInducingPoints, doEpUpdate, datasForPredictedClasses[iClassNr], iClassNr)));
					printOnScreen("Calc ivm for " << ClassKnowledge::getNameFor(iClassNr) << "_"<< ClassKnowledge::getNameFor(iClassNr));
				}
//			}
		}
		group->join_all();
		for(unsigned int iClassNr = 0; iClassNr < amountOfClasses(); ++iClassNr){
			SAVE_DELETE(datasForPredictedClasses[iClassNr]);
		}
		SAVE_DELETE(group);
		printOnScreen("Finished Training!");
	}else{
		printError("Not implemented yet!");
	}
}

void OnlineRandomForestIVMs::trainIvm(const int usedIvm, const int nrOfInducingPoints, const bool doEpUpdate, ClassData* data, const int orfClass){
	m_onlineStoragesForIvms[usedIvm] = new OnlineStorage<ClassPoint*>();
	m_ivms[usedIvm] = new IVMMultiBinary(*m_onlineStoragesForIvms[usedIvm], nrOfInducingPoints, doEpUpdate, orfClass);
	m_onlineStoragesForIvms[usedIvm]->append(*data); // this append will call the update of the ivm and will start the training
}

int OnlineRandomForestIVMs::predict(const DataPoint& point) const{
	const int label = m_orf.predict(point);
	if(label != UNDEF_CLASS_LABEL){
		if(m_ivms[label] != nullptr){
			return m_ivms[label]->predict(point);
		}
		return label;
	}
	return UNDEF_CLASS_LABEL;
}

int OnlineRandomForestIVMs::predict(const ClassPoint& point) const{
	const int label = m_orf.predict(point);
	if(label != UNDEF_CLASS_LABEL){
		if(m_ivms[label] != nullptr){
			return m_ivms[label]->predict(point);
		}
		return label;
	}
	return UNDEF_CLASS_LABEL;
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

unsigned int OnlineRandomForestIVMs::amountOfClasses() const{
	return m_amountOfUsedClasses;
}
