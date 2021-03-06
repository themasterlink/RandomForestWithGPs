/*
 * TotalStorage.cc
 *
 *  Created on: 14.10.2016
 *      Author: Max
 */

#include "TotalStorage.h"
#include "DataReader.h"
#include "../RandomNumberGenerator/RandomUniformNr.h"

TotalStorage::TotalStorage(): m_totalSize(0),
							  m_dataSetMode(DataSetMode::WHOLE){};


void TotalStorage::readData(const int amountOfData){
	std::string folderLocation;
	m_dataSetMode = DataSetMode::WHOLE;
	if(CommandSettings::instance().get_useFakeData()){
		Settings::instance().getValue("TotalStorage.folderLocFake", folderLocation);
	}else{
		Settings::instance().getValue("TotalStorage.folderLocReal", folderLocation);
	}
	const bool readTxt = false;
	bool didNormalizeStep = false;
	if(Settings::instance().getDirectBoolValue("TotalStorage.readFromFolder")){
		if(folderLocation == "../washington/"){
			m_dataSetMode = DataSetMode::SEPARATE; // seperate train und test set
			int testNr = 2;
			Settings::instance().getValue("TotalStorage.folderTestNr", testNr);
			boost::filesystem::path targetDir(folderLocation);
			boost::filesystem::directory_iterator end_itr;
			LabeledData wholeTrainingSet;
			for(boost::filesystem::directory_iterator itr(targetDir); itr != end_itr; ++itr){
				if(boost::filesystem::is_regular_file(itr->path()) && boost::filesystem::extension(itr->path()) == ".binary"){
					const std::string inputPath(itr->path().c_str());
//					bool StringHelper::endsWithBool = false;
//					for(unsigned int i = 0; i < 10; ++i){
//						if(i != testNr){
//							StringHelper::endsWithBool = StringHelper::endsWithBool || StringHelper::endsWith(inputPath, "rgbd_features_train_split" + StringHelper::number2String(i) + "_5th.binary");
//						}
//					}
					//finetuned_8192_
					if(StringHelper::endsWith(inputPath, "rgbd_features_train_split" + StringHelper::number2String(testNr) + "_5th.binary")){
						printOnScreen("As training:");
						DataReader::readFromBinaryFile(wholeTrainingSet, inputPath, INT_MAX);
//						DataReader::readFromFile(wholeTrainingSet, inputPath.substr(0, inputPath.length() - 4), INT_MAX, UNDEF_CLASS_LABEL, true);
					}else if(StringHelper::endsWith(inputPath, "rgbd_features_test_split" + StringHelper::number2String(testNr) + "_5th.binary")){
						printOnScreen("As test:");
						DataReader::readFromBinaryFile(m_testSet, inputPath, INT_MAX);
//						DataReader::readFromFile(m_testSet,  inputPath.substr(0, inputPath.length() - 4), INT_MAX, UNDEF_CLASS_LABEL, true);
					}
				}
			}
//			if(m_testSet.size() == 0){
//				printError("No test point found for testNr: " << testNr);
//				quitApplication();
//			}
			const unsigned int jumper = 1;
			if(jumper > 1){
				m_trainSet.reserve(wholeTrainingSet.size() + jumper);
				for(unsigned int i = 0; i < wholeTrainingSet.size(); ++i){
					if(i % jumper == 0){
						m_trainSet.emplace_back(wholeTrainingSet[i]);
					}else{
						delete wholeTrainingSet[i]; // remove point from the memory
					}
				}
				printOnScreen("Jumper for washington is " << jumper << " reduced from: " << wholeTrainingSet.size() << " to " << m_trainSet.size());
			}else{
				m_trainSet = wholeTrainingSet;
			}
//			std::set<unsigned int> classes;
//			for(LabeledDataConstIterator it = m_trainSet.begin(); it != m_trainSet.end(); ++it){
//				if(classes.find((**it).getLabel()) == classes.end()){
//					classes.insert((**it).getLabel());
//					ClassKnowledge::instance().instance().setNameFor(StringHelper::number2String((**it).getLabel()), (**it).getLabel());
//				}
//			}
		}else if(folderLocation == "../mnistOrg/" || folderLocation == "../uspsOrg/"){
			m_dataSetMode = DataSetMode::SEPARATE; // seperate train und test set
			DataSets train, test;
			DataReader::readFromFiles(train, folderLocation + "training/", (unsigned int) amountOfData, readTxt, didNormalizeStep);
			unsigned int totalSize = 0;
			for(DataSetsConstIterator it = train.begin(); it != train.end(); ++it){
				totalSize += it->second.size();
			}
			m_trainSet.reserve(totalSize);
			for(DataSetsConstIterator it = train.begin(); it != train.end(); ++it){
				for(unsigned int i = 0; i < it->second.size(); ++i){
					m_trainSet.emplace_back(it->second[i]);
				}
			}
			DataReader::readFromFiles(test, folderLocation + "test/", (unsigned int) amountOfData, readTxt, didNormalizeStep);
			unsigned int totalSizeTest = 0;
			for(DataSetsConstIterator it = test.begin(); it != test.end(); ++it){
				totalSizeTest += it->second.size();
			}
			m_testSet.reserve(totalSizeTest);
			for(DataSetsConstIterator it = test.begin(); it != test.end(); ++it){
				for(unsigned int i = 0; i < it->second.size(); ++i){
					m_testSet.emplace_back(it->second[i]);
				}
			}
		}else if(StringHelper::endsWith(folderLocation, "/washingtonData") || StringHelper::endsWith(folderLocation, "/washingtonMax")){
			m_dataSetMode = DataSetMode::SEPARATE; // seperate train und test set
			boost::filesystem::path targetDir(folderLocation);
			boost::filesystem::directory_iterator end_itr;
			LabeledData wholeTrainingSet;
			for(boost::filesystem::directory_iterator itr(targetDir); itr != end_itr; ++itr){
				if(boost::filesystem::is_regular_file(itr->path()) && boost::filesystem::extension(itr->path()) == ".binary"){
					const std::string inputPath(itr->path().c_str());
					if(StringHelper::endsWith(inputPath, "trn_flatten_complete.binary")){
						printOnScreen("As training:");
						DataReader::readFromBinaryFile(m_trainSet, inputPath, INT_MAX);
					}else if(StringHelper::endsWith(inputPath, "vld_flatten_complete.binary")){
						if(Settings::instance().getDirectBoolValue("TotalStorage.useValidationForTraining")){
							printOnScreen("As validation:");
						}else{
							printOnScreen("As additional training:");
						}
						DataReader::readFromBinaryFile(m_validationSet, inputPath, INT_MAX);
					}else if(StringHelper::endsWith(inputPath, "eval_flatten_complete.binary")){
						printOnScreen("As test:");
						DataReader::readFromBinaryFile(m_testSet, inputPath, INT_MAX);
					}
				}
			}
			if(Settings::instance().getDirectBoolValue("TotalStorage.useValidationForTraining")){
				m_trainSet.reserve(m_trainSet.size() + m_validationSet.size());
				m_trainSet.insert(m_trainSet.end(), m_validationSet.begin(), m_validationSet.end());
				m_validationSet.clear();
			}
		}else{
			DataReader::readFromFiles(m_storage, folderLocation, (unsigned int) amountOfData, readTxt, didNormalizeStep);
		}
	}else{
		LabeledData data;
		DataReader::readFromBinaryFile(data, "../binary/dataFor_0.binary", (unsigned int) amountOfData);
		for(unsigned int i = 0; i < data.size(); ++i){
			DataSetsIterator it = m_storage.find(ClassKnowledge::instance().instance().getNameFor(data[i]->getLabel()));
			if(it != m_storage.end()){
				it->second.emplace_back(data[i]);
			}else{
				LabeledData newData;
				m_storage.emplace(ClassKnowledge::instance().instance().getNameFor(data[i]->getLabel()), newData);
				DataSetsIterator newIt = m_storage.find(
						ClassKnowledge::instance().instance().getNameFor(data[i]->getLabel()));
				if(newIt != m_storage.end()){
					newIt->second.emplace_back(data[i]);
				}
			}
		}
	}
	if(Settings::instance().getDirectBoolValue("TotalStorage.removeUselessDimensions")){
		if(m_dataSetMode == DataSetMode::WHOLE){
			std::vector<bool> isUsed(ClassKnowledge::instance().instance().amountOfDims(), false);
			for(unsigned int dim = 0; dim < ClassKnowledge::instance().instance().amountOfDims(); ++dim){
				const Real value = m_storage.begin()->second[0]->coeff(dim);
				for(Iterator it = m_storage.begin(); it != m_storage.end() && !isUsed[dim]; ++it){
					const unsigned int start = m_storage.begin() == it ? 1 : 0;
					for(unsigned int i = start; i < it->second.size(); ++i){
						if(fabs(value - it->second[i]->coeff(dim)) >= 1e-7){
							isUsed[dim] = true;
							break;
						}
					}
				}
			}
			unsigned int newAmountOfDims = 0;
			for(unsigned int i = 0; i < isUsed.size(); ++i){
				if(isUsed[i]){
					++newAmountOfDims;
				}
			}
			printOnScreen("newAmountOfDims: " << newAmountOfDims);
			for(Iterator it = m_storage.begin(); it != m_storage.end(); ++it){
				for(unsigned int i = 0; i < it->second.size(); ++i){
					LabeledVectorX* newPoint = new LabeledVectorX(newAmountOfDims, it->second[i]->getLabel());
					unsigned int realIndex = 0;
					for(unsigned int j = 0; j < isUsed.size(); ++j){
						if(isUsed[j]){
							newPoint->coeffRef(realIndex) = it->second[i]->coeff(j);
							++realIndex;
						}
					}
					if(realIndex != newAmountOfDims){
						printError("Something went wrong!");
					}
					saveDelete(it->second[i]);
					it->second[i] = newPoint;
				}
			}
//			printOnScreen("First: " << m_storage.begin()->second[0]->transpose());
//			printOnScreen("Secon: " << m_storage.rbegin()->second[1]->transpose());
//			LabeledVectorX& f = *m_storage.begin()->second[0];
//			LabeledVectorX& s = *m_storage.begin()->second[1];
//			bool isSame = true;
//			for(unsigned int i = 0; i < newAmountOfDims; ++i){
//				if(fabs(f.coeff(i) - s.coeff(i)) >= 1e-7){
//					printOnScreen("Value at: " << i << " is different! " << f.coeff(i) << ", " << s.coeff(i));
//					isSame = false;
//					break;
//				}
//			}
//			if(isSame){
//				printOnScreen("Is the same!");
//			}
			ClassKnowledge::instance().instance().setAmountOfDims(newAmountOfDims);
		}else if(m_dataSetMode == DataSetMode::SEPARATE){
			std::vector<bool> isUsed(ClassKnowledge::instance().instance().amountOfDims(), false);
			printOnScreen("Amount of dims: " << ClassKnowledge::instance().instance().amountOfDims());
			for(unsigned int dim = 0; dim < ClassKnowledge::instance().instance().amountOfDims(); ++dim){
				const Real value = m_trainSet[0]->coeff(dim);
				for(unsigned int i = 1; i < m_trainSet.size(); ++i){
					if(fabs(value - m_trainSet[i]->coeff(dim)) >= 1e-7){
						isUsed[dim] = true;
						break;
					}
				}
			}
			unsigned int newAmountOfDims = 0; //ClassKnowledge::instance().instance().amountOfDims();
			for(unsigned int i = 0; i < isUsed.size(); ++i){
				if(isUsed[i]){
					++newAmountOfDims;
				}
			}

			printOnScreen("newAmountOfDims: " << newAmountOfDims);
			if(newAmountOfDims != ClassKnowledge::instance().instance().amountOfDims()){
				for(LabeledData* it : {&m_trainSet, &m_testSet}){
					for(unsigned int i = 0; i < it->size(); ++i){
						LabeledVectorX* newPoint = new LabeledVectorX(newAmountOfDims, (*it)[i]->getLabel());
						unsigned int realIndex = 0;
						for(unsigned int j = 0; j < isUsed.size(); ++j){
							if(isUsed[j]){
								newPoint->coeffRef(realIndex) = (*it)[i]->coeff(j);
								++realIndex;
							}
						}
						if(realIndex != newAmountOfDims){
							printError("Something went wrong!");
						}
						saveDelete((*it)[i]);
						(*it)[i] = newPoint;
					}
				}
			}
			ClassKnowledge::instance().instance().setAmountOfDims(newAmountOfDims);
		}
	}
	if(m_dataSetMode == DataSetMode::WHOLE){
		std::string type = "";
		Settings::instance().getValue("main.type", type);
		if(!type.compare(0, 6, "binary") &&
		   !CommandSettings::instance().get_onlyDataView()){ // type starts with binary -> remove all classes
			if(m_storage.size() > 2){
				Iterator it = m_storage.begin();
				++it; ++it; // go to the third element!
				for(;it != m_storage.end();){
					std::string name = it->first;
					for(unsigned int i = 0; i < it->second.size(); ++i){
						saveDelete(it->second[i]);
					}
					++it;
					m_storage.erase(name);
				}
			}
		}
		for(ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
			m_totalSize += it->second.size();
		}
		if(Settings::instance().getDirectBoolValue("TotalStorage.normalizeData") && !didNormalizeStep){
			DataConverter::centerAndNormalizeData(m_storage, m_center, m_var);
		}
	}else{
		std::string type = "";
		Settings::instance().getValue("main.type", type);
		unsigned int usedClass = 0;
		Settings::instance().getValue("TotalStorage.folderTestNr", usedClass);
		unsigned int removeClass;
		Settings::instance().getValue("TotalStorage.excludeClass", removeClass);
		if(!type.compare(0, 6, "binary") &&
		   !CommandSettings::instance().get_onlyDataView()){ // type starts with binary -> remove all classes
			for(LabeledData* it : {&m_trainSet, &m_testSet}){
				for(unsigned int i = 0; i < it->size(); ++i){
					if(usedClass == (*it)[i]->getLabel()){
						(*it)[i]->setLabel(0);
					}else{
						(*it)[i]->setLabel(1);
					}
				}
			}
			unsigned int amountOfDims = ClassKnowledge::instance().instance().amountOfDims();
			ClassKnowledge::instance().instance().init();
			ClassKnowledge::instance().instance().setAmountOfDims(amountOfDims);
			ClassKnowledge::instance().instance().setNameFor(StringHelper::number2String(usedClass), 0);
			ClassKnowledge::instance().instance().setNameFor("rest", 1);
		}else if(ClassKnowledge::instance().instance().hasClassName(removeClass)){
			std::list<LabeledVectorX*> trainList;
			m_removeFromTrainSet.reserve(
					m_trainSet.size() / ClassKnowledge::instance().instance().amountOfClasses() * 2);
			for(unsigned int i = 0; i < m_trainSet.size(); ++i){
				if(m_trainSet[i]->getLabel() != removeClass){
					trainList.emplace_back(m_trainSet[i]);
				}else{
					m_removeFromTrainSet.emplace_back(m_trainSet[i]);
				}
			}
			m_trainSet.clear();
			m_trainSet.reserve(trainList.size());
			for(auto& trainVecs : trainList){
				m_trainSet.emplace_back(trainVecs);
			}
			std::list<LabeledVectorX*> testList;
			m_removeFromTestSet.reserve(m_testSet.size() / ClassKnowledge::instance().instance().amountOfClasses() * 2);
			for(unsigned int i = 0; i < m_testSet.size(); ++i){
				if(m_testSet[i]->getLabel() != removeClass){
					testList.emplace_back(m_testSet[i]);
				}else{
					m_removeFromTestSet.emplace_back(m_testSet[i]);
				}
			}
			m_testSet.clear();
			m_testSet.reserve(testList.size());
			for(auto it = testList.begin(); it != testList.end(); ++it){
				m_testSet.emplace_back(*it);
			}
			printOnScreen("Removed class: " << removeClass << " from train: " << m_removeFromTrainSet.size() << ", from test: " << m_removeFromTestSet.size());
		}
		int amountOfSizeStep = 0;
		Settings::instance().getValue("TotalStorage.stepOverTrainingData", amountOfSizeStep);
		if(amountOfSizeStep > 1){
			std::string mainType;
			Settings::instance().getValue("main.type", mainType);
			std::list<LabeledVectorX*> trainList;
			const bool useWholeClass =
					!mainType.compare(0, 6, "binary") && !CommandSettings::instance().get_onlyDataView();
			printOnScreen("Usewholeclass: " << useWholeClass);
			if(useWholeClass){
				for(unsigned int i = 0; i < m_trainSet.size(); ++i){
					if(m_trainSet[i]->getLabel() == 0){
						trainList.emplace_back(m_trainSet[i]);
					}
				}
			}
			RandomUniformNr uniformNr(1,amountOfSizeStep,100);
			int nextStopPoint = uniformNr();
			for(unsigned int i = 0; i < m_trainSet.size(); ++i){
				if(nextStopPoint == i){
					if(useWholeClass){
						if(m_trainSet[i]->getLabel() == 1){
							trainList.emplace_back(m_trainSet[i]);
						}
					}else{
						trainList.emplace_back(m_trainSet[i]);
					}
					nextStopPoint += uniformNr();
				}else{
					// reduces the amount of memory used
					saveDelete(m_trainSet[i]);
				}
			}
			printOnScreen("Reduced training size from: " << m_trainSet.size() << " to: " << trainList.size());
			m_trainSet.clear();
			m_trainSet.reserve(trainList.size());
			for(auto it = trainList.begin(); it != trainList.end(); ++it){
				m_trainSet.emplace_back(*it);
			}
		}
		m_totalSize = (unsigned int) (m_trainSet.size() + m_testSet.size());
		if(Settings::instance().getDirectBoolValue("TotalStorage.normalizeData") && !didNormalizeStep){
			DataConverter::centerAndNormalizeData(m_trainSet, m_center, m_var); // first calc on training set
			DataConverter::centerAndNormalizeData(m_testSet, m_center, m_var);  // apply to test set
		}
	}
}

unsigned int TotalStorage::getTotalSize(){
	return m_totalSize;
}

unsigned int TotalStorage::getAmountOfClass(){
	if(m_dataSetMode == DataSetMode::WHOLE){
		return (unsigned int) m_storage.size();
	}else{
		return ClassKnowledge::instance().instance().amountOfClasses();
	}
}

void TotalStorage::getOnlineStorageCopyWithTest(OnlineStorage<LabeledVectorX*>& train,
		OnlineStorage<LabeledVectorX*>& test, const int amountOfPointsForTraining){
	if(m_dataSetMode == DataSetMode::WHOLE){
		LabeledData forTraining;
		LabeledData forTesting;
		getLabeledDataCopyWithTest(forTraining, forTesting, amountOfPointsForTraining);
		train.append(forTraining);
		test.append(forTesting);
	}else{
		train.append(m_trainSet);
		test.append(m_testSet);
	}
}

void TotalStorage::getLabeledDataCopyWithTest(LabeledData& train, LabeledData& test,
											  const int amountOfPointsForTraining){
	if(m_dataSetMode == DataSetMode::WHOLE){
		int minValue = amountOfPointsForTraining / getAmountOfClass();
		for(ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
			minValue = std::min(static_cast<int>(it->second.size()), minValue);
		}
		for(ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
			int counter = 0;
			for(LabeledDataConstIterator itData = it->second.begin(); itData != it->second.end(); ++itData){
				if(counter < minValue){
					train.emplace_back(*itData);
				}else{
					test.emplace_back(*itData);
				}
				++counter;
			}
		}
		printOnScreen("For training: " << train.size());
		printOnScreen("For testing: " << test.size());
	}else{
		train = m_trainSet;
		test = m_testSet;
	}
}

void TotalStorage::getRemovedOnlineStorageCopyWithTest(OnlineStorage<LabeledVectorX*>& train,
		OnlineStorage<LabeledVectorX*>& test){
	if(m_dataSetMode == DataSetMode::WHOLE){
		printErrorAndQuit("Not implemented yet!");
	}else{
		train.append(m_removeFromTrainSet);
		test.append(m_removeFromTestSet);
	}
}

void TotalStorage::getOnlineStorageCopySplitsWithTest(std::vector<OnlineStorage<LabeledVectorX*> >& trains, OnlineStorage<LabeledVectorX*>& test){
	if(m_dataSetMode != DataSetMode::WHOLE){
		if(!trains.empty()){
			auto amountOfSplits = (unsigned int)(trains.size());
			std::vector<LabeledData> forTrainings(amountOfSplits);
			for(unsigned int i = 0; i < m_trainSet.size(); ++i){
				forTrainings[i % amountOfSplits].emplace_back(m_trainSet[i]);
			}
			for(unsigned int i = 0; i < amountOfSplits; ++i){
				printOnScreen("Size of " << i << ": " << forTrainings[i].size());
				trains[i].append(forTrainings[i]);
			}
			test.append(m_testSet);
		}else{
			printErrorAndQuit("Amount of splits can not be zero!");
		}
	}else{
		printErrorAndQuit("Not implemented yet!");
	}
}

unsigned int TotalStorage::getSmallestClassSize(){
	if(m_dataSetMode == DataSetMode::WHOLE){
		unsigned int min = INT_MAX;
		for(ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
			min = std::min(min, (unsigned int) it->second.size());
		}
		return !m_storage.empty() ? min : 0;
	}else{
		const auto amountOfClasses = getAmountOfClass();
		if(amountOfClasses > 0){
			return (unsigned int) (m_trainSet.size() / getAmountOfClass());
		}else{
			printError("The class amount is zero!");
			return 0;
		}
	}
	return 0;
}

LabeledData* TotalStorage::getValidationSet(){
	if(!m_validationSet.empty()){
		return &m_validationSet;
	}else{
		return nullptr;
	}
}
