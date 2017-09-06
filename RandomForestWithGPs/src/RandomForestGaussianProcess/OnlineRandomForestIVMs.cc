/*
 * OnlineRandomForestIVMs.cc
 *
 *  Created on: 17.11.2016
 *      Author: Max
 */

#include "OnlineRandomForestIVMs.h"
#include "../Utility/ConfusionMatrixPrinter.h"

OnlineRandomForestIVMs::OnlineRandomForestIVMs(OnlineStorage<LabeledVectorX*>& storage, const int maxDepth, const int amountOfUsedClasses):
	m_storage(storage),
	m_orf(storage, maxDepth, amountOfUsedClasses),
	m_ivms(amountOfUsedClasses, nullptr),
	m_onlineStoragesForIvms(amountOfUsedClasses, nullptr),
	m_amountOfUsedClasses(amountOfUsedClasses),
	m_firstTrainedDone(false){
	// removes orf to avoid that the update is called directly to the orf
	m_orf.getStorageRef().detach(&m_orf);
	// instead call the update on the
	m_storage.attach(this);
}

OnlineRandomForestIVMs::~OnlineRandomForestIVMs() {
}

void OnlineRandomForestIVMs::update(Subject* caller, unsigned int event){
	UNUSED(caller);
	switch(event){
	case OnlineStorage<LabeledVectorX*>::Event::APPEND:{
		printErrorAndQuit("This is not implemented yet!");
		break;
	}
	case OnlineStorage<LabeledVectorX*>::Event::APPENDBLOCK:{
		update();
		break;
	}
	case OnlineStorage<LabeledVectorX*>::Event::ERASE:{
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
		OnlineStorage<LabeledVectorX*>* copyForORFs = new OnlineStorage<LabeledVectorX*>(m_storage);
		m_orf.update(copyForORFs, OnlineStorage<LabeledVectorX*>::Event::APPENDBLOCK);
//		m_orf.update();
		std::list<unsigned int> predictedLabels;
		unsigned int amountOfCorrect = 0;
		Matrix conv = Matrix::Zero(ClassKnowledge::instance().amountOfClasses(),
								   ClassKnowledge::instance().amountOfClasses());
		for(unsigned int i = 0; i < m_storage.size(); ++i){
			auto& point = *m_storage[i];
			const unsigned int predictedLabel = m_orf.predict(point);
			predictedLabels.push_back(predictedLabel);
			if(predictedLabel == point.getLabel()){
				++amountOfCorrect;
			}
			conv(point.getLabel(), predictedLabels.back()) += 1;
		}
		printOnScreen("Just ORFs:");
		ConfusionMatrixPrinter::print(conv);
		printOnScreen("Just ORFs correct: " << StringHelper::number2String(amountOfCorrect / (Real) m_storage.size() * 100.0, 2));
		UniquePtr<ThreadGroup> group = std::make_unique<ThreadGroup>();
		std::vector<UniquePtr<LabeledData> > datasForPredictedClasses(amountOfClasses());
		int nrOfInducingPoints = 40;
		Settings::instance().getValue("IVM.nrOfInducingPoints", nrOfInducingPoints);
		for(unsigned int iClassNr = 0; iClassNr < amountOfClasses(); ++iClassNr){
			// for each class find all predicted values which should be considered in this class
			datasForPredictedClasses[iClassNr] = std::make_unique<LabeledData>();
			auto itPredictedLabel = predictedLabels.cbegin();
			std::vector<unsigned int> classCounter(amountOfClasses(), 0);
			for(unsigned int i = 0; i < m_storage.size(); ++i){
//			for(OnlineStorage<LabeledVectorX*>::ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it, ++itPredictedLabel){
				if(*itPredictedLabel == iClassNr){
					datasForPredictedClasses[iClassNr]->emplace_back(m_storage[i]);
					++classCounter[m_storage[i]->getLabel()];
				}
			}
			const int sizeOfPointsForClass = datasForPredictedClasses[iClassNr]->size();
//			for(unsigned int iInnerClassNr = 0; iInnerClassNr < amountOfClasses(); ++iInnerClassNr){
//				printOnScreen("Size of class " << ClassKnowledge::instance().getNameFor(iClassNr) << "_"<< ClassKnowledge::instance().getNameFor(iInnerClassNr)<< ": " << classCounter[iInnerClassNr] << ", whole is: " << sizeOfPointsForClass);
			if(sizeOfPointsForClass * 0.95 > classCounter[iClassNr] && nrOfInducingPoints + 100 < sizeOfPointsForClass){ // if less than 95 % of the points belong to the right class -> use ivms
				const bool doEpUpdate = false;
				group->addThread(makeThread(&OnlineRandomForestIVMs::trainIvm, this, iClassNr, nrOfInducingPoints, doEpUpdate, datasForPredictedClasses[iClassNr].get(), iClassNr));
				printOnScreen("Calc ivm for " << ClassKnowledge::instance().getNameFor(iClassNr) << "_"
											  << ClassKnowledge::instance().getNameFor(iClassNr));
			}
//			}
		}
		group->joinAll();
		printOnScreen("Finished Training!");
	}else{
		printErrorAndQuit("Not implemented yet!");
	}
}

void OnlineRandomForestIVMs::trainIvm(const int usedIvm, const int nrOfInducingPoints, const bool doEpUpdate, LabeledData* data, const int orfClass){
	m_onlineStoragesForIvms[usedIvm] = new OnlineStorage<LabeledVectorX*>();
	m_ivms[usedIvm] = new IVMMultiBinary(*m_onlineStoragesForIvms[usedIvm], nrOfInducingPoints, doEpUpdate, orfClass);
	m_onlineStoragesForIvms[usedIvm]->append(*data); // this append will call the update of the ivm and will start the training
}

unsigned int OnlineRandomForestIVMs::predict(const VectorX& point) const{
	const unsigned int label = m_orf.predict(point);
	if(label != UNDEF_CLASS_LABEL){
		if(m_ivms[label] != nullptr){
			return m_ivms[label]->predict(point);
		}
		return label;
	}
	return UNDEF_CLASS_LABEL;
}

unsigned int OnlineRandomForestIVMs::predict(const LabeledVectorX& point) const{
	const unsigned int label = m_orf.predict(point);
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

void OnlineRandomForestIVMs::predictData(const LabeledData& points, Labels& labels) const{
	labels.resize(points.size());
	int i = 0;
	for(LabeledDataConstIterator it = points.begin(); it != points.end(); ++it, ++i){
		labels[i] = predict(**it);
	}
}

void OnlineRandomForestIVMs::predictData(const Data& points, Labels& labels, std::vector< std::vector<Real> >& probabilities) const{
	m_orf.predictData(points, labels, probabilities);
	int i = 0;
	for(DataConstIterator it = points.begin(); it != points.end(); ++it, ++i){
		if(m_ivms[labels[i]] != nullptr){
			for(unsigned int j = 0; j < amountOfClasses(); ++j){
				probabilities[i][j] = 0;
			}
			m_ivms[labels[i]]->predict(**it, probabilities[i]);
			labels[i] = argMax(probabilities[i]);
		}
	}
}

void OnlineRandomForestIVMs::predictData(const LabeledData& points, Labels& labels, std::vector< std::vector<Real> >& probabilities) const{
	m_orf.predictData(points, labels, probabilities);
	int i = 0;
	for(LabeledDataConstIterator it = points.begin(); it != points.end(); ++it, ++i){
		if(m_ivms[labels[i]] != nullptr){
			for(unsigned int j = 0; j < amountOfClasses(); ++j){
				probabilities[i][j] = 0;
			}
			m_ivms[labels[i]]->predict(**it, probabilities[i]);
			labels[i] = argMax(probabilities[i]);
		}
	}
}

unsigned int OnlineRandomForestIVMs::amountOfClasses() const{
	return m_amountOfUsedClasses;
}
