//
// Created by denn_ma on 5/30/17.
//

#include "TestManager.h"
#include "../Base/Settings.h"
#include "../Data/TotalStorage.h"
#include "../Utility/ConfusionMatrixPrinter.h"
#include "../Data/DataWriterForVisu.h"

// have to be defined before the test Information is defined
const std::string TestInformation::trainSettingName("TRAIN_SETTING");
const std::string TestInformation::testSettingName("TEST_SETTING");

TestManager::TestManager(): m_filePath(""){};

void TestManager::init(){;
	std::fstream file(m_filePath,std::ios::in);
	if(file.is_open()){
		std::string line;
		while(std::getline(file, line)){
			StringHelper::removeLeadingAndTrailingWhiteSpaces(line);
			StringHelper::removeCommentFromLine(line);
			if(line.length() > 2){
				TestMode mode = findMode(line);
				m_testInformation.addDefinitionOrInstruction(mode, line);
			}
		}
	}else{
		printErrorAndQuit("The test settings file is not there: " << m_filePath);
	}
}

TestMode TestManager::findMode(std::string &line){
	std::string firstWord = StringHelper::getFirstWord(line);
	if(firstWord.length() > 0){
		if(firstWord == "load"){
			return TestMode::LOAD;
		}else if(firstWord == "train"){
			return TestMode::TRAIN;
		}else if(firstWord == "test"){
			return TestMode::TEST;
		}else if(firstWord == "define"){
			std::vector<std::string> words;
			StringHelper::getWords(line, words);
			for(unsigned int i = 0; i < words.size(); ++i){
				StringHelper::removeLeadingAndTrailingWhiteSpaces(words[i]);
				if(words[i] == "split" || words[i] == "splits"){
					return TestMode::SPLIT;
				}
			}
			// no split found is just a usual define
			return TestMode::DEFINE;
		}else if(firstWord == "remove"){
			return TestMode::REMOVE;
		}else if(firstWord == "update"){
			return TestMode::UPDATE;
		}else if(firstWord == "combine"){
			return TestMode::COMBINE;
		}else if(firstWord == "endfor"){
			return TestMode::END_FOR;
		}else if(firstWord == "for"){
			return TestMode::FOR;
		}
	}
	return TestMode::UNDEFINED;
}

int TestManager::readAll(){
	int firstPoints; // all points
	Settings::instance().getValue("TotalStorage.amountOfPointsUsedForTraining", firstPoints);
	const Real share = Settings::instance().getDirectRealValue("TotalStorage.shareForTraining");
	firstPoints /= share;
	printOnScreen("Read " << firstPoints << " points per class");
	TotalStorage::instance().readData(firstPoints);
	printOnScreen("TotalStorage::instance().getSmallestClassSize(): " << TotalStorage::instance().getSmallestClassSize()
																	  << " with "
																	  << TotalStorage::instance().getAmountOfClass()
																	  << " classes");
	const auto trainAmount = (int) (share *
									(std::min((int) TotalStorage::instance().getSmallestClassSize(), firstPoints) *
									 (Real) TotalStorage::instance().getAmountOfClass()));
	return trainAmount;
}

void TestManager::run(){
	OnlineStorage<LabeledVectorX* > train;
	if(Settings::instance().getDirectBoolValue("OnlineRandomForest.activatePoolBased")){
		train.setStorageModeToPoolBase();
	}
	unsigned int height;
	Settings::instance().getValue("OnlineRandomForest.Tree.height", height);
	UniquePtr<OnlineRandomForest> orf;
	unsigned int i = 0;
	auto lastInfo = m_testInformation.getInstruction(i);
	while(true){
		auto testInfo = m_testInformation.getInstruction(i);
		if(testInfo.m_mode != TestMode::UNDEFINED){
			if(testInfo.m_scope != nullptr){
				testInfo.m_scope->replaceScopeVariables(testInfo.m_varName);
			}
			if(testInfo.m_mode != TestMode::FOR){
				if(testInfo.m_mode == TestMode::END_FOR){
					if(lastInfo.m_scope != nullptr){
						auto& forScope = *lastInfo.m_scope;
						++forScope.m_currentValue;
						if(forScope.m_currentValue < forScope.m_range.size()){
							i = m_testInformation.getInstructionNr(
									forScope); // find number of for, at the end + 1 to get to the next
						}
					}else{
						printErrorAndQuit("This can never happen!");
					}
				}else{
					auto usedDefinition = m_testInformation.getDefinition(testInfo.m_varName, testInfo.m_scope);
					if(usedDefinition.getVarName() != ""){
						if(testInfo.m_mode == TestMode::LOAD){
							readAll(); // TODO read TRAIN or TEST not both
							orf = std::make_unique<OnlineRandomForest>(train, height,
																	   (int) TotalStorage::instance().getAmountOfClass());
							orf->setValidationSet(TotalStorage::instance().getValidationSet());
//							if(TotalStorage::instance().getValidationSet()){
//								OnlineStorage<LabeledVectorX*> test;
//								test.appendUnique(*TotalStorage::instance().getValidationSet());
//								printOnScreen("Unique elements in validation set are: " << test.size());
//							}
						}else if(testInfo.m_mode == TestMode::TRAIN){
							auto data = getAllPointsFor(testInfo.m_varName, testInfo.m_scope);
							OnlineRandomForest::TrainingsConfig config;
							using Mode = TestInformation::Instruction::ExitMode;
							config.m_mode = testInfo.m_exitMode;
							if(config.isTimeMode()){
								printOnScreen("Time constraint: " << TimeFrame(testInfo.m_seconds));
								config.m_seconds = testInfo.m_seconds;
							}else if(config.isTreeAmountMode()){
								printOnScreen("Amount of tree constraint: " << testInfo.m_amountOfTrees);
								config.m_amountOfTrees = testInfo.m_amountOfTrees;
							}
							if(config.hasMemoryConstraint()){
								printOnScreen(
										"Memory constraint: " << StringHelper::convertMemorySpace(testInfo.m_memory));
								config.m_memory = testInfo.m_memory;
							}
							orf->setTrainingsMode(config);
							if(orf){
								train.appendUnique(data);
							}
							printOnScreen("Avg Time for Dynamic Decision Tree train: "
												  << GlobalStopWatch<DynamicDecisionTreeTrain>::instance().elapsedAvgAsTimeFrame()
												  << ", amount trained trees: "
												  << GlobalStopWatch<DynamicDecisionTreeTrain>::instance().getAvgCounter());
							printOnScreen("Avg Time for Big Decision Tree train: "
												  << GlobalStopWatch<BigDecisionTreeTrain>::instance().elapsedAvgAsTimeFrame()
												  << ", amount trained trees: "
												  << GlobalStopWatch<BigDecisionTreeTrain>::instance().getAvgCounter());
							printOnScreen("Avg amount of starting size for DDT: "
												  << GlobalCounter<DDDTAvgSizeOfUsedStorage>::instance().mean());
						}else if(testInfo.m_mode == TestMode::TEST){
							auto data = getAllPointsFor(testInfo.m_varName, testInfo.m_scope);
							printOnScreen("Perform test for: " << testInfo.m_varName);
							performTest(orf, data);
						}
					}
				}
			}
		}else{
			break;
		}
		lastInfo = std::move(testInfo);
		++i;
	}
	printOnScreen("Run through all instructions!");
	if(CommandSettings::instance().get_useFakeData() &&
	   (CommandSettings::instance().get_visuRes() > 0 || CommandSettings::instance().get_visuResSimple() > 0)){
		StopWatch sw;
		DataWriterForVisu::writeImg("orf.png", orf.get(), train.storage());
		printOnScreen("For drawing needed time: " << sw.elapsedAsTimeFrame());
		openFileInViewer("orf.png");
	}
}

LabeledData TestManager::getAllPointsFor(const std::string& defName, TestInformation::Instruction* scope){
	LabeledData res;
	printOnScreen("Get all points for: " << defName);
	auto usedDefinition = m_testInformation.getDefinition(defName, scope);
	if(usedDefinition.isTrainOrTestSetting()){ // stops recursion
 		// is either train or test so only the points from one are used
		int firstPoints; // all points
		Settings::instance().getValue("TotalStorage.amountOfPointsUsedForTraining", firstPoints);
		LabeledData dummy; // is necessary for interface consistency
		if(usedDefinition.getVarName() == TestInformation::trainSettingName){
			TotalStorage::instance().getLabeledDataCopyWithTest(res, dummy, firstPoints);
		}else if(usedDefinition.getVarName() == TestInformation::testSettingName){
			TotalStorage::instance().getLabeledDataCopyWithTest(dummy, res, firstPoints);
		}
	}else{
		if(usedDefinition.m_splitAmount < 1){
			if(usedDefinition.m_firstFromVariable.length() > 0){
				res = getAllPointsFor(usedDefinition.m_firstFromVariable, scope);
				if(usedDefinition.m_secondFromVariable.length() > 0){ // combine action
					auto res2 = getAllPointsFor(usedDefinition.m_secondFromVariable, scope);
					res.insert(res.end(), res2.begin(), res2.end()); // combine both
				}
			}else{
				printErrorAndQuit("This can not happen the definition is wrong: " << usedDefinition);
			}
		}else{
			if(usedDefinition.m_firstFromVariable.length() > 0){
				if(usedDefinition.m_secondFromVariable.length() > 0){
					printErrorAndQuit("A combine and a split can not be performed in the same define use two defines for that");
				}
				int splitNr = m_testInformation.getSplitNumber(defName);
				const auto jumper = (unsigned int) usedDefinition.m_splitAmount;
				if(splitNr < 0){
					printErrorAndQuit("This split number is not supported here: " << usedDefinition.m_firstFromVariable << ", " << usedDefinition.getVarName());
				}else if(splitNr >= jumper){
					printErrorAndQuit("Accessing an split element, which is not available!");
				}
				LabeledData temp = getAllPointsFor(usedDefinition.m_firstFromVariable, scope);
				res.reserve(temp.size() / usedDefinition.m_splitAmount + usedDefinition.m_splitAmount);
				for(unsigned int i = splitNr; i < temp.size(); i += jumper){
					res.emplace_back(temp[i]);
				}
			}else{
				printErrorAndQuit("This can not happen the definition is wrong: " << usedDefinition);
			}
		}
	}
	removeClassesFrom(res, usedDefinition);
	return res;
}

void TestManager::removeClassesFrom(LabeledData& data, const TestInformation::TestDefineName& info){
	if(info.m_classes.size() == 1 && info.m_classes[0] == UNDEF_CLASS_LABEL){ // == "all"
		if(info.m_includeClasses){
			return; // no change necessary
		}else{
			data.resize(0); // remove all points (not really useful action)
			return;
		}
	}
	unsigned int lastUsed = 0;
	for(unsigned int i = 0; i < data.size(); ++i){
		bool keepIt = !info.m_includeClasses;
		const unsigned int currectLabel = data[i]->getLabel();
		for(const auto& classNr : info.m_classes){
			if(currectLabel == classNr){
				keepIt = info.m_includeClasses;
				break;
			}
		}
		if(keepIt){
			data[lastUsed] = data[i];
			++lastUsed;
		}
	}
	data.resize(lastUsed); // remove all the empty pointers at the end
}

void TestManager::performTest(const UniquePtr<OnlineRandomForest>& orf, const LabeledData& test){
	if(test.size() == 0){
		printOnScreen("Test size is zero!");
		return;
	}
	int amountOfCorrect = 0;
	Labels labels;
	StopWatch sw;
	std::vector<std::vector<Real> > probs;
	orf->predictData(test, labels, probs);
	printOnScreen("Needed " << sw.elapsedAsTimeFrame());
	Matrix conv = Matrix::Zero(orf->amountOfClasses(), orf->amountOfClasses());
//	std::vector<std::list<Real> > lists(orf->amountOfClasses(), std::list<Real>());
	AvgNumber oc, uc;
	AvgNumber ocBVS, ucBVS;
	const unsigned int amountOfClasses = ClassKnowledge::instance().amountOfClasses();
	const Real logBase = (Real) logReal(amountOfClasses);
	for(unsigned int i = 0; i < labels.size(); ++i){
		if(labels[i] != UNDEF_CLASS_LABEL){
			Real entropy = 0;
			for(unsigned int j = 0; j < amountOfClasses; ++j){
				if(probs[i][j] > 0){
					entropy -= probs[i][j] * logReal(probs[i][j]) / logBase;
				}
			}
			Real max1 = 0, max2 = 0;
			for(unsigned int j = 0; j < amountOfClasses; ++j){
				if(probs[i][j] > max1){
					max2 = max1;
					max1 = probs[i][j];
				}
			}
			Real entropyBVS = max2 / max1;
			if(test[i]->getLabel() == labels[i]){
				++amountOfCorrect;
				uc.addNew((Real) entropy);
				ucBVS.addNew((Real) entropyBVS);
			}else{
				oc.addNew((Real) (1. - entropy));
				ocBVS.addNew((Real) (1. - entropyBVS));
//				printOnScreen("Class: " << ClassKnowledge::instance().getNameFor(test[i]->getLabel()) << ", for 0: " << probs[i][0] << ", for 1: " << probs[i][1]);
			}
//			lists[labels[i]].emplace_back(probs[i][labels[i]]); // adds only the winning label to the list
			conv(test[i]->getLabel(), labels[i]) += 1;

		}
	}
	printOnScreen("Test size: " << test.size());
	printOnScreen("Result:    " << amountOfCorrect / (Real) test.size() * 100. << " %");
	printOnScreen("Overconf:  " << oc.mean() * 100.0 << "%%");
	printOnScreen("Underconf: " << uc.mean() * 100.0 << "%%");
	printOnScreen("Overconf BVS:  " << ocBVS.mean() * 100.0 << "%%");
	printOnScreen("Underconf BVS: " << ucBVS.mean() * 100.0 << "%%");
	if(conv.rows() < 40){ // otherwise not useful
		ConfusionMatrixPrinter::print(conv);
	}
}

