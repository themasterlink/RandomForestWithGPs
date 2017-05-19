/*
 * TotalStorage.cc
 *
 *  Created on: 14.10.2016
 *      Author: Max
 */

#include "TotalStorage.h"
#include "ClassKnowledge.h"
#include "DataReader.h"
#include "../Base/Settings.h"
#include "../Base/ScreenOutput.h"
#include "../RandomNumberGenerator/RandomUniformNr.h"
#include "../Base/CommandSettings.h"
#include "../Data/DataConverter.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

TotalStorage::InternalStorage TotalStorage::m_storage;
ClassData TotalStorage::m_trainSet;
ClassData TotalStorage::m_testSet;
ClassData TotalStorage::m_removeFromTrainSet;
ClassData TotalStorage::m_removeFromTestSet;
ClassPoint TotalStorage::m_defaultEle;
unsigned int TotalStorage::m_totalSize(0);
DataPoint TotalStorage::m_center;
DataPoint TotalStorage::m_var;
TotalStorage::Mode TotalStorage::m_mode = TotalStorage::Mode::WHOLE;

TotalStorage::TotalStorage(){}

TotalStorage::~TotalStorage(){}

ClassPoint* TotalStorage::getData(unsigned int classNr, unsigned int elementNr){
	if(m_storage.size() > 0){
		Iterator it = m_storage.find(ClassKnowledge::getNameFor(classNr));
		if(it != m_storage.end()){
			return it->second[elementNr];
		}
	}
	return &m_defaultEle;
}

void TotalStorage::readData(const int amountOfData){
	std::string folderLocation;
	m_mode = Mode::WHOLE;
	if(CommandSettings::get_useFakeData()){
		Settings::getValue("TotalStorage.folderLocFake", folderLocation);
	}else{
		Settings::getValue("TotalStorage.folderLocReal", folderLocation);
	}
	const bool readTxt = false;
	bool didNormalizeStep = false;
	if(Settings::getDirectBoolValue("TotalStorage.readFromFolder")){
		if(folderLocation == "../washington/"){
			m_mode = Mode::SEPERATE; // seperate train und test set
			int testNr = 2;
			Settings::getValue("TotalStorage.folderTestNr", testNr);
			boost::filesystem::path targetDir(folderLocation);
			boost::filesystem::directory_iterator end_itr;
			ClassData wholeTrainingSet;
			for(boost::filesystem::directory_iterator itr(targetDir); itr != end_itr; ++itr){
				if(boost::filesystem::is_regular_file(itr->path()) && boost::filesystem::extension(itr->path()) == ".binary"){
					const std::string inputPath(itr->path().c_str());
//					bool endsWithBool = false;
//					for(unsigned int i = 0; i < 10; ++i){
//						if(i != testNr){
//							endsWithBool = endsWithBool || endsWith(inputPath, "rgbd_features_train_split" + number2String(i) + "_5th.binary");
//						}
//					}
					//finetuned_8192_
					if(endsWith(inputPath, "rgbd_features_train_split" + number2String(testNr) + "_5th.binary")){
						printOnScreen("As training:");
						DataReader::readFromBinaryFile(wholeTrainingSet, inputPath, INT_MAX);
//						DataReader::readFromFile(wholeTrainingSet, inputPath.substr(0, inputPath.length() - 4), INT_MAX, UNDEF_CLASS_LABEL, true);
					}else if(endsWith(inputPath, "rgbd_features_test_split" + number2String(testNr) + "_5th.binary")){
						printOnScreen("As test:");
						DataReader::readFromBinaryFile(m_testSet, inputPath, INT_MAX);
//						DataReader::readFromFile(m_testSet,  inputPath.substr(0, inputPath.length() - 4), INT_MAX, UNDEF_CLASS_LABEL, true);
					}
				}
			}
//			if(m_testSet.size() == 0){
//				printError("No test point found for testNr: " << testNr);
//				Logger::forcedWrite();
//				exit(0);
//			}
			const unsigned int jumper = 1;
			if(jumper > 1){
				m_trainSet.reserve(wholeTrainingSet.size() + jumper);
				for(unsigned int i = 0; i < wholeTrainingSet.size(); ++i){
					if(i % jumper == 0){
						m_trainSet.push_back(wholeTrainingSet[i]);
					}else{
						delete wholeTrainingSet[i]; // remove point from the memory
					}
				}
				printOnScreen("Jumper for washington is " << jumper << " reduced from: " << wholeTrainingSet.size() << " to " << m_trainSet.size());
			}else{
				m_trainSet = wholeTrainingSet;
			}
//			std::set<unsigned int> classes;
//			for(ClassDataConstIterator it = m_trainSet.begin(); it != m_trainSet.end(); ++it){
//				if(classes.find((**it).getLabel()) == classes.end()){
//					classes.insert((**it).getLabel());
//					ClassKnowledge::setNameFor(number2String((**it).getLabel()), (**it).getLabel());
//				}
//			}
		}else if(folderLocation == "../mnistOrg/" || folderLocation == "../uspsOrg/"){
			m_mode = Mode::SEPERATE; // seperate train und test set
			DataSets train, test;
			DataReader::readFromFiles(train, folderLocation + "training/", amountOfData, readTxt, didNormalizeStep);
			unsigned int totalSize = 0;
			for(DataSetsConstIterator it = train.begin(); it != train.end(); ++it){
				totalSize += it->second.size();
			}
			m_trainSet.reserve(totalSize);
			for(DataSetsConstIterator it = train.begin(); it != train.end(); ++it){
				for(unsigned int i = 0; i < it->second.size(); ++i){
					m_trainSet.push_back(it->second[i]);
				}
			}
			DataReader::readFromFiles(test, folderLocation + "test/", amountOfData, readTxt, didNormalizeStep);
			unsigned int totalSizeTest = 0;
			for(DataSetsConstIterator it = test.begin(); it != test.end(); ++it){
				totalSizeTest += it->second.size();
			}
			m_testSet.reserve(totalSizeTest);
			for(DataSetsConstIterator it = test.begin(); it != test.end(); ++it){
				for(unsigned int i = 0; i < it->second.size(); ++i){
					m_testSet.push_back(it->second[i]);
				}
			}
		}else if(endsWith(folderLocation, "/washingtonData") || endsWith(folderLocation, "/washingtonData/")){
			m_mode = Mode::SEPERATE; // seperate train und test set
			boost::filesystem::path targetDir(folderLocation);
			boost::filesystem::directory_iterator end_itr;
			ClassData wholeTrainingSet;
			for(boost::filesystem::directory_iterator itr(targetDir); itr != end_itr; ++itr){
				if(boost::filesystem::is_regular_file(itr->path()) && boost::filesystem::extension(itr->path()) == ".binary"){
					const std::string inputPath(itr->path().c_str());
					if(!endsWith(inputPath, "eval_flatten_complete.binary")){
						printOnScreen("As training:");
						DataReader::readFromBinaryFile(wholeTrainingSet, inputPath, INT_MAX);
					}else{
						printOnScreen("As test:");
						DataReader::readFromBinaryFile(m_testSet, inputPath, INT_MAX);
					}
				}
			}
			m_trainSet = wholeTrainingSet;
		}else{
			DataReader::readFromFiles(m_storage, folderLocation, amountOfData, readTxt, didNormalizeStep);
		}
	}else{
		ClassData data;
		DataReader::readFromBinaryFile(data, "../binary/dataFor_0.binary", amountOfData);
		for(unsigned int i = 0; i < data.size(); ++i){
			DataSetsIterator it = m_storage.find(ClassKnowledge::getNameFor(data[i]->getLabel()));
			if(it != m_storage.end()){
				it->second.push_back(data[i]);
			}else{
				ClassData newData;
				m_storage.insert(DataSetPair(ClassKnowledge::getNameFor(data[i]->getLabel()), newData));
				DataSetsIterator newIt = m_storage.find(ClassKnowledge::getNameFor(data[i]->getLabel()));
				if(newIt != m_storage.end()){
					newIt->second.push_back(data[i]);
				}
			}
		}
	}
	if(Settings::getDirectBoolValue("TotalStorage.removeUselessDimensions")){
		if(m_mode == Mode::WHOLE){
			std::vector<bool> isUsed(ClassKnowledge::amountOfDims(), false);
			for(unsigned int dim = 0; dim < ClassKnowledge::amountOfDims(); ++dim){
				const double value = m_storage.begin()->second[0]->coeff(dim);
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
					ClassPoint* newPoint = new ClassPoint(newAmountOfDims, it->second[i]->getLabel());
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
					SAVE_DELETE(it->second[i]);
					it->second[i] = newPoint;
				}
			}
//			printOnScreen("First: " << m_storage.begin()->second[0]->transpose());
//			printOnScreen("Secon: " << m_storage.rbegin()->second[1]->transpose());
//			ClassPoint& f = *m_storage.begin()->second[0];
//			ClassPoint& s = *m_storage.begin()->second[1];
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
			ClassKnowledge::setAmountOfDims(newAmountOfDims);
		}else if(m_mode == Mode::SEPERATE){
			std::vector<bool> isUsed(ClassKnowledge::amountOfDims(), false);
			printOnScreen("Amount of dims: " << ClassKnowledge::amountOfDims());
			for(unsigned int dim = 0; dim < ClassKnowledge::amountOfDims(); ++dim){
				const double value = m_trainSet[0]->coeff(dim);
				for(unsigned int i = 1; i < m_trainSet.size(); ++i){
					if(fabs(value - m_trainSet[i]->coeff(dim)) >= 1e-7){
						isUsed[dim] = true;
						break;
					}
				}
			}
			unsigned int newAmountOfDims = 0; //ClassKnowledge::amountOfDims();
			for(unsigned int i = 0; i < isUsed.size(); ++i){
				if(isUsed[i]){
					++newAmountOfDims;
				}
			}
//			cv::Mat img(28, 28, CV_8UC3, cv::Scalar(0, 0, 0));
//			for(unsigned int r = 0; r < 28; ++r){
//				for(unsigned int c = 0; c < 28; ++c){
//					cv::Vec3b& color = img.at<cv::Vec3b>(r,c);
//					color[0] = (isUsed[r * 28 + c] ? 0.8 : 0) * 255;
//					color[1] = (isUsed[r * 28 + c] ? 0.8 : 0) * 255;
//					color[2] = (isUsed[r * 28 + c] ? 0.8 : 0) * 255;
//				}
//			}
//			cv::Mat outImg;
//			cv::resize(img, outImg, cv::Size(img.cols * 20,img.rows * 20), 0, 0, CV_INTER_NN);
//			cv::imwrite(Logger::getActDirectory() + "test.png", outImg);
//			openFileInViewer("test.png");

			printOnScreen("newAmountOfDims: " << newAmountOfDims);
			if(newAmountOfDims != ClassKnowledge::amountOfDims()){
				for(ClassData* it : {&m_trainSet, &m_testSet}){
					for(unsigned int i = 0; i < it->size(); ++i){
						ClassPoint* newPoint = new ClassPoint(newAmountOfDims, (*it)[i]->getLabel());
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
						SAVE_DELETE((*it)[i]);
						(*it)[i] = newPoint;
					}
				}
			}
			ClassKnowledge::setAmountOfDims(newAmountOfDims);
		}
	}
	if(m_mode == Mode::WHOLE){
		std::string type = "";
		Settings::getValue("main.type", type);
		if(!type.compare(0, 6, "binary") && !CommandSettings::get_onlyDataView()){ // type starts with binary -> remove all classes
			if(m_storage.size() > 2){
				Iterator it = m_storage.begin();
				++it; ++it; // go to the third element!
				for(;it != m_storage.end();){
					std::string name = it->first;
					for(unsigned int i = 0; i < it->second.size(); ++i){
						SAVE_DELETE(it->second[i]);
					}
					++it;
					m_storage.erase(name);
				}
			}
		}
		for(ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
			m_totalSize += it->second.size();
		}
		if(Settings::getDirectBoolValue("TotalStorage.normalizeData") && !didNormalizeStep){
			DataConverter::centerAndNormalizeData(m_storage, m_center, m_var);
		}
	}else{
		std::string type = "";
		Settings::getValue("main.type", type);
		unsigned int usedClass = 0;
		Settings::getValue("TotalStorage.folderTestNr", usedClass);
		unsigned int removeClass;
		Settings::getValue("TotalStorage.excludeClass", removeClass);
		if(!type.compare(0, 6, "binary") && !CommandSettings::get_onlyDataView()){ // type starts with binary -> remove all classes
			for(ClassData* it : {&m_trainSet, &m_testSet}){
				for(unsigned int i = 0; i < it->size(); ++i){
					if(usedClass == (*it)[i]->getLabel()){
						(*it)[i]->setLabel(0);
					}else{
						(*it)[i]->setLabel(1);
					}
				}
			}
			unsigned int amountOfDims = ClassKnowledge::amountOfDims();
			ClassKnowledge::init();
			ClassKnowledge::setAmountOfDims(amountOfDims);
			ClassKnowledge::setNameFor(number2String(usedClass), 0);
			ClassKnowledge::setNameFor("rest", 1);
		}else if(ClassKnowledge::hasClassName(removeClass)){
			std::list<ClassPoint*> trainList;
			m_removeFromTrainSet.reserve(m_trainSet.size() / ClassKnowledge::amountOfClasses() * 2);
			for(unsigned int i = 0; i < m_trainSet.size(); ++i){
				if(m_trainSet[i]->getLabel() != removeClass){
					trainList.push_back(m_trainSet[i]);
				}else{
					m_removeFromTrainSet.push_back(m_trainSet[i]);
				}
			}
			m_trainSet.clear();
			m_trainSet.reserve(trainList.size());
			for(std::list<ClassPoint*>::const_iterator it = trainList.begin(); it != trainList.end(); ++it){
				m_trainSet.push_back(*it);
			}
			std::list<ClassPoint*> testList;
			m_removeFromTestSet.reserve(m_testSet.size() / ClassKnowledge::amountOfClasses() * 2);
			for(unsigned int i = 0; i < m_testSet.size(); ++i){
				if(m_testSet[i]->getLabel() != removeClass){
					testList.push_back(m_testSet[i]);
				}else{
					m_removeFromTestSet.push_back(m_testSet[i]);
				}
			}
			m_testSet.clear();
			m_testSet.reserve(testList.size());
			for(std::list<ClassPoint*>::const_iterator it = testList.begin(); it != testList.end(); ++it){
				m_testSet.push_back(*it);
			}
			printOnScreen("Removed class: " << removeClass << " from train: " << m_removeFromTrainSet.size() << ", from test: " << m_removeFromTestSet.size());
		}
		int amountOfSizeStep = 0;
		Settings::getValue("TotalStorage.stepOverTrainingData", amountOfSizeStep);
		if(amountOfSizeStep > 1){
			std::string type = "";
			Settings::getValue("main.type", type);
			std::list<ClassPoint*> trainList;
			const bool useWholeClass = !type.compare(0, 6, "binary") && !CommandSettings::get_onlyDataView();
			printOnScreen("Usewholeclass: " << useWholeClass);
			if(useWholeClass){
				for(unsigned int i = 0; i < m_trainSet.size(); ++i){
					if(m_trainSet[i]->getLabel() == 0){
						trainList.push_back(m_trainSet[i]);
					}
				}
			}
			RandomUniformNr uniformNr(1,amountOfSizeStep,100);
			int nextStopPoint = uniformNr();
			for(unsigned int i = 0; i < m_trainSet.size(); ++i){
				if(nextStopPoint == i){
					if(useWholeClass){
						if(m_trainSet[i]->getLabel() == 1){
							trainList.push_back(m_trainSet[i]);
						}
					}else{
						trainList.push_back(m_trainSet[i]);
					}
					nextStopPoint += uniformNr();
				}else{
					// reduces the amount of memory used
					SAVE_DELETE(m_trainSet[i]);
				}
			}
			printOnScreen("Reduced training size from: " << m_trainSet.size() << " to: " << trainList.size());
			m_trainSet.clear();
			m_trainSet.reserve(trainList.size());
			for(std::list<ClassPoint*>::const_iterator it = trainList.begin(); it != trainList.end(); ++it){
				m_trainSet.push_back(*it);
			}
		}
		m_totalSize = m_trainSet.size() + m_testSet.size();
		if(Settings::getDirectBoolValue("TotalStorage.normalizeData") && !didNormalizeStep){
			DataConverter::centerAndNormalizeData(m_trainSet, m_center, m_var); // first calc on training set
			DataConverter::centerAndNormalizeData(m_testSet, m_center, m_var);  // apply to test set
		}
	}
}

ClassPoint* TotalStorage::getDefaultEle(){
	return &m_defaultEle;
}

unsigned int TotalStorage::getTotalSize(){
	return m_totalSize;
}

unsigned int TotalStorage::getAmountOfClass(){
	if(m_mode == Mode::WHOLE){
		return m_storage.size();
	}else{
		return ClassKnowledge::amountOfClasses();
	}
}

void TotalStorage::getOnlineStorageCopy(OnlineStorage<ClassPoint*>& storage){
	if(m_mode == Mode::WHOLE){
		storage.resize(m_totalSize);
		for(ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
			for(ClassDataConstIterator itData = it->second.begin(); itData != it->second.end(); ++itData){
				storage.append(*itData);
			}
		}
	}else{
		printError("Not implemented for this mode!");
	}
}

void TotalStorage::getOnlineStorageCopyWithTest(OnlineStorage<ClassPoint*>& train,
		OnlineStorage<ClassPoint*>& test, const int amountOfPointsForTraining){
	if(m_mode == Mode::WHOLE){
		int minValue = amountOfPointsForTraining / getAmountOfClass();
		for(ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
			minValue = std::min((int)it->second.size(), minValue);
		}
		std::vector<ClassPoint*> forTraining;
		std::vector<ClassPoint*> forTesting;
		for(ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
			int counter = 0;
			for(ClassDataConstIterator itData = it->second.begin(); itData != it->second.end(); ++itData){
				if(counter < minValue){
					forTraining.push_back(*itData);
				}else{
					forTesting.push_back(*itData);
				}
				++counter;
			}
		}
		printOnScreen("For training: " << forTraining.size());
		printOnScreen("For testing: " << forTesting.size());
		// to guarantee that the append block update is called which invokes the training
		train.append(forTraining);
		test.append(forTesting);
	}else{
		train.append(m_trainSet);
		test.append(m_testSet);
	}
}

void TotalStorage::getRemovedOnlineStorageCopyWithTest(OnlineStorage<ClassPoint*>& train,
		OnlineStorage<ClassPoint*>& test){
	if(m_mode == Mode::WHOLE){
		printError("Not implemented yet!");
		Logger::forcedWrite();
	}else{
		train.append(m_removeFromTrainSet);
		test.append(m_removeFromTestSet);
	}
}

void TotalStorage::getOnlineStorageCopySplitsWithTest(std::vector<OnlineStorage<ClassPoint*> >& trains, OnlineStorage<ClassPoint*>& test){
	if(m_mode != Mode::WHOLE){
		if(trains.size() != 0){
			const unsigned int amountOfSplits = trains.size();
			std::vector<ClassData> forTrainings(amountOfSplits);
			for(unsigned int i = 0; i < m_trainSet.size(); ++i){
				forTrainings[i % amountOfSplits].push_back(m_trainSet[i]);
			}
			for(unsigned int i = 0; i < amountOfSplits; ++i){
				printOnScreen("Size of " << i << ": " << forTrainings[i].size());
				trains[i].append(forTrainings[i]);
			}
			test.append(m_testSet);
		}else{
			printError("Amount of splits can not be zero!");
			Logger::forcedWrite();
		}
	}else{
		printError("Not implemented yet!");
		Logger::forcedWrite();
	}
}

unsigned int TotalStorage::getSize(unsigned int classNr){
	if(m_mode == Mode::WHOLE){
		Iterator it = m_storage.find(ClassKnowledge::getNameFor(classNr));
		if(it != m_storage.end()){
			return it->second.size();
		}
	}else{
		printError("Not implemented for this mode");
	}
	return 0;
}

unsigned int TotalStorage::getSmallestClassSize(){
	if(m_mode == Mode::WHOLE){
		unsigned int min = INT_MAX;
		for(ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
			min = std::min(min, (unsigned int) it->second.size());
		}
		return m_storage.size() != 0 ? min : 0;
	}else{
		return m_trainSet.size() / getAmountOfClass();
	}
	return 0;
}
