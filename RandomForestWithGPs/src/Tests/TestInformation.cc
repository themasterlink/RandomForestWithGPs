//
// Created by denn_ma on 5/31/17.
//

#include <regex>
#include "TestInformation.h"


TestInformation::TestInformation(): m_currentScope(nullptr){
	TestDefineName trainSetting;
	trainSetting.setVarName(trainSettingName);
	trainSetting.m_firstFromVariable = trainSettingName;
	trainSetting.useAllClasses();
	TestDefineName testSetting;
	testSetting.setVarName(testSettingName);
	testSetting.m_firstFromVariable = testSettingName;
	testSetting.useAllClasses();
	TestDefineName allSetting;
	allSetting.setVarName("all");
	allSetting.m_firstFromVariable = trainSettingName;
	allSetting.m_secondFromVariable = testSettingName;
	allSetting.useAllClasses();
	m_definitions.emplace(trainSetting.getVarName(), trainSetting);
	m_definitions.emplace(testSetting.getVarName(), testSetting);
	m_definitions.emplace(allSetting.getVarName(), allSetting);
}


void TestInformation::addDefinitionOrInstruction(const TestMode mode, const std::string& definition){
	std::vector<std::string> words;
	std::string def = definition;
	StringHelper::getWords(def, words);
	switch(mode){
		case TestMode::FOR:{
			if(words.size() > 2 && words[1] == "in"){
				Instruction instruction(mode, words[0], m_currentScope);
				instruction.m_currentValue = 0;
				StringHelper::decipherRange(words, 2, (unsigned int) (words.size() - 1), instruction.m_range);
				if(instruction.m_range.size() == 0 || instruction.m_range[0] == UNDEF_CLASS_LABEL){
					printErrorAndQuit("This range is not valid: " << def);
				}
				m_instructions.emplace_back(std::move(instruction));
				m_currentScope = &m_instructions.back();
			}else{
				printErrorAndQuit("For is badly formatted: " << def);
			}
			break;
		}
		case TestMode::END_FOR:{
			if(m_currentScope != nullptr){
				m_instructions.emplace_back(Instruction(mode, "", m_currentScope));
				m_currentScope = m_currentScope->m_scope; // get one scope up
			}else{
				printError("End for without opening for!");
			}
			break;
		}
		case TestMode::LOAD:{
			if(words.size() == 1 && (m_definitions.find(words[0]) != m_definitions.end() || words[0] == "all")){
				m_instructions.emplace_back(mode, words[0], m_currentScope);
			}else{
				printErrorAndQuit("Loading is only possible for defined or predefined sets: " << def);
			}
			break;
		}
		case TestMode::DEFINE:{
			TestDefineName defineName;
			defineName.setVarName(words[0]);
			bool foundFrom = false, foundClasses = false;
			for(unsigned int i = 0; i < words.size() - 1; ++i){
				if(words[i] == "from"){
					if(m_definitions.find(getDefinition(words[i + 1], m_currentScope).getVarName()) !=
					   m_definitions.end()){
						defineName.m_firstFromVariable = words[i + 1];
					}else{
						printErrorAndQuit("A type definition can only refer to the basic types " << trainSettingName << " or "
										    << testSettingName << " or a different type def: " << words[i + 1]);
					}
					foundFrom = true;
				}else if(words[i + 1] == "classes"){
					// classes -> check with or without
					if(words[i] == "as" || words[i] == "with"){
						defineName.m_includeClasses = true;
					}else if(words[i] == "without"){
						defineName.m_includeClasses = false;
					}else{
						printErrorAndQuit("The definition before classes must be \"as\" or \"without\", "
										   << "this is not valid" << words[i]);
					}
					//find from
					for(unsigned int j = i + 2; j < words.size(); ++j){
						if(words[j] == "from"){
							foundClasses = StringHelper::decipherRange(words, i + 2, j - 1, defineName.m_classes);
							break;
						}
					}
					if(!foundClasses){
						printErrorAndQuit("The classes were incorrect: " << def);
					}
				}
			}
			if(!foundFrom){
				printErrorAndQuit("From must be defined in the def: " << def);
			}
			if(!foundClasses){
				// default is use all classes
				defineName.m_includeClasses = true;
				defineName.m_classes.emplace_back(UNDEF_CLASS_LABEL);
			}
			m_definitions.emplace(defineName.getVarName(), defineName);
			break;
		}
		case TestMode::REMOVE: {
			printError("Not implemented yet");
			break;
		}
		case TestMode::COMBINE: {
			if(words.size() == 5 && words[1] == "with" && words[3] == "in"){
				const std::string name = getDefinition(words[0], m_currentScope).getVarName();
				if(m_definitions.find(name) != m_definitions.end()){
					const std::string name2 = getDefinition(words[2], m_currentScope).getVarName();
					if(m_definitions.find(name2) != m_definitions.end()){
						TestDefineName defineName;
						defineName.setVarName(words[4]);
						defineName.m_firstFromVariable = words[0];
						defineName.m_secondFromVariable= words[2];
						defineName.m_includeClasses = false;
						m_definitions.emplace(defineName.getVarName(), defineName);
					}else{
						printErrorAndQuit("This type was not defined before: " << words[2] << ", used in: " << def);
					}
				}else{
					printErrorAndQuit("This type was not defined before: " << words[0] << ", used in: " << def);
				}
			}else{
				printErrorAndQuit("The combine statement must consist out of: \"first with second in new\", first and second must be defined before!");
			}
			break;
		}
		case TestMode::TRAIN:
		case TestMode::UPDATE:{
			if(words.size() > 0){
				const std::string name = getDefinition(words[0], m_currentScope).getVarName();
				if(m_definitions.find(name) != m_definitions.end()){
					m_instructions.emplace_back(mode, words[0], m_currentScope);
					m_instructions.back().processLine(words);
				}else{
					printErrorAndQuit("This type was not defined before: " << words[0] << ", used in: " << def);
				}
			}else{
				printErrorAndQuit("Loading is only possible for defined or predefined sets: " << def);
			}
			break;
		}
		case TestMode::TEST:
		{
			if(words.size() == 1){
				const std::string name = getDefinition(words[0], m_currentScope).getVarName();
				if(m_definitions.find(name) != m_definitions.end()){
					m_instructions.emplace_back(mode, words[0], m_currentScope);
				}else{
					printErrorAndQuit("This type was not defined before: " << words[0] << ", used in: " << def);
				}
			}else{
				printErrorAndQuit("Loading is only possible for defined or predefined sets: " << def);
			}
			break;
		}
		case TestMode::SPLIT:
		{
			if(words.size() > 0){
				TestDefineName defineName;
				defineName.setVarName(words[0]);
				if(m_definitions.find(words.back()) != m_definitions.end()){
					defineName.m_firstFromVariable = words.back();
					defineName.m_includeClasses = false;
					string2Int(words[2], defineName.m_splitAmount,
							   "The split number: " << words[2] << " is no number!");
					m_definitions.emplace(defineName.getVarName(), defineName);
					m_instructions.emplace_back(mode, words[0], m_currentScope);
				}else{
					printErrorAndQuit("The used dataset in this split has to be defined before: " << def);
				}
			}else{
				printErrorAndQuit("Splitting needs a new name for the definition: " << def);
			}
			break;
		}
		case TestMode::UNDEFINED:
			printError("This line is undefined: " << def);
			break;
	}

}

TestInformation::TestDefineName TestInformation::getDefinition(const std::string& def, Instruction* currentScope){
	std::string name = def;
	if(currentScope != nullptr){
		currentScope->replaceScopeVariables(name); // just to get the default field
	}
	auto it = m_definitions.find(name);
	if(it != m_definitions.end()){
		return it->second;
	}
	// search for split value
	auto pos = name.find('[');
	if(pos != name.npos){
		it = m_definitions.find(name.substr(0, pos));
		if(it != m_definitions.end()){
			auto splitNr = getSplitNumber(name);
			if(splitNr >= it->second.m_splitAmount){
				printErrorAndQuit("This split number is not allowed for this name: " << name);
			}
			return it->second;
		}
	}
	printErrorAndQuit("This data set does not exist: " << name << ", currentScope: " << (currentScope != nullptr));
	return TestDefineName();
}

TestInformation::Instruction TestInformation::getInstruction(const unsigned int i){
	if(i < m_instructions.size()){
		return m_instructions[i];
	}
	return Instruction(TestMode::UNDEFINED, "", nullptr);
}

int TestInformation::getSplitNumber(const std::string& name){
	auto start = name.find('[');
	auto end = name.find(']');
	if(start != name.npos && end != name.npos){
		++start;
		int ret = 0;
		string2Int(name.substr(start, end - start), ret, "The name: " << name << " does not contain a split number!");
		return ret;
	}
	return -1;
}

unsigned int TestInformation::getInstructionNr(const TestInformation::Instruction& instruction){
	for(unsigned int i = 0; i < m_instructions.size(); ++i){
		auto& cur = m_instructions[i];
		if(instruction == cur){
			return i;
		}
	}
	return 0;
}


bool TestInformation::TestDefineName::isTrainOrTestSetting() const{
	return m_varName == TestInformation::trainSettingName || m_varName == TestInformation::testSettingName;
}

void TestInformation::TestDefineName::useAllClasses(){
	m_classes.clear();
	m_classes.emplace_back((unsigned int) UNDEF_CLASS_LABEL);
	m_includeClasses = true;
}

std::string TestInformation::TestDefineName::getVarName() const{
	auto start = m_varName.find('[');
	if(start != m_varName.npos){
		return m_varName.substr(0, start);
	}
	return m_varName;
}

void TestInformation::TestDefineName::setVarName(std::string name){
	m_varName = std::move(name); // only one copy necessary
}

void TestInformation::Instruction::addType(const std::string& preposition, const std::string& nrType){
	if(preposition == "for"){
		// time
		std::string cpyNrType(nrType);
		StringHelper::removeLeadingAndTrailingWhiteSpaces(cpyNrType);
		double fac = 1.0;
		if(StringHelper::endsWith(cpyNrType, 'm')){
			fac = 60;
		}else if(StringHelper::endsWith(cpyNrType, 'h')){
			fac = 60 * 60;
		}else if(StringHelper::endsWith(cpyNrType, 'd')){
			fac = 60 * 60 * 24;
		}
		if(StringHelper::endsWith(cpyNrType, 's')
		   || StringHelper::endsWith(cpyNrType, 'm')
			  || StringHelper::endsWith(cpyNrType, 'h')
				 || StringHelper::endsWith(cpyNrType, 'd')){
			cpyNrType = cpyNrType.substr(0, cpyNrType.size() - 1);
		}// else assume seconds
		StringHelper::removeLeadingAndTrailingWhiteSpaces(cpyNrType);
		string2Real(cpyNrType, m_seconds, "The time: " << cpyNrType << " is no real value!");
		m_seconds *= fac; // add factor for minute, hour and day
		if(m_exitMode == Instruction::ExitMode::UNDEFINED){
			m_exitMode = Instruction::ExitMode::TIME;
		}else if(m_exitMode == Instruction::ExitMode::MEMORY){
			m_exitMode = Instruction::ExitMode::TIME_WITH_MEMORY;
		}else{
			printErrorAndQuit("This should not happen!");
		}
	}else if(preposition == "with" || preposition == "withonly"){
		// memory space
		std::string cpyNrType(nrType);
		StringHelper::removeLeadingAndTrailingWhiteSpaces(cpyNrType);
		MemoryType fac = 1;
		if(StringHelper::endsWith(cpyNrType, "gb")){
			fac = 1000000000;
			cpyNrType = cpyNrType.substr(0, cpyNrType.length() - 2);
		}else if(StringHelper::endsWith(cpyNrType, "mb")){
			fac = 1000000;
			cpyNrType = cpyNrType.substr(0, cpyNrType.length() - 2);
		}else if(StringHelper::endsWith(cpyNrType, "kb")){
			fac = 1000;
			cpyNrType = cpyNrType.substr(0, cpyNrType.length() - 2);
		}// else assume gb
		StringHelper::removeLeadingAndTrailingWhiteSpaces(cpyNrType);
		Real realMemory;
		string2Real(cpyNrType, realMemory, "The memory value: " << cpyNrType << " is not a number!");
		m_memory = (MemoryType) realMemory;
		m_memory *= fac;
		if(m_exitMode == Instruction::ExitMode::UNDEFINED){
			m_exitMode = Instruction::ExitMode::MEMORY;
		}else if(m_exitMode == Instruction::ExitMode::TIME){
			m_exitMode = Instruction::ExitMode::TIME_WITH_MEMORY;
		}else if(m_exitMode == Instruction::ExitMode::TREEAMOUNT){
			m_exitMode = Instruction::ExitMode::TREEAMOUNT_WITH_MEMORY;
		}else{
			printErrorAndQuit("This should not happen!");
		}
	}else if(preposition == "until"){ // tree amount
		// memory space
		std::string cpyNrType(nrType);
		StringHelper::removeLeadingAndTrailingWhiteSpaces(cpyNrType);
		if(StringHelper::endsWith(cpyNrType, "trees")){
			cpyNrType = cpyNrType.substr(0, cpyNrType.length() - 5);
		}
		if(StringHelper::endsWith(cpyNrType, "tree")){
			cpyNrType = cpyNrType.substr(0, cpyNrType.length() - 4);
		}
		// else assume trees
		StringHelper::removeLeadingAndTrailingWhiteSpaces(cpyNrType);
		int iTrees = 0;
		string2Int(cpyNrType, iTrees, "The memory value: " << cpyNrType << " is not a number!");
		m_amountOfTrees = iTrees;
		if(m_exitMode == Instruction::ExitMode::UNDEFINED){
			m_exitMode = Instruction::ExitMode::TREEAMOUNT;
		}else if(m_exitMode == Instruction::ExitMode::MEMORY){
			m_exitMode = Instruction::ExitMode::TREEAMOUNT_WITH_MEMORY;
		}else{
			printErrorAndQuit("This should not happen!");
		}
	}else{
		printErrorAndQuit("This preposition is unknown: " << preposition);
	}
}

void TestInformation::Instruction::processLine(const std::vector<std::string>& words){
	m_exitMode = Instruction::ExitMode::UNDEFINED;
	for(unsigned int i = 1; i < words.size() - 1; ++i){
		std::string cpyWord(words[i]);
		StringHelper::removeLeadingAndTrailingWhiteSpaces(cpyWord);
		if(cpyWord == "for"){
			for(unsigned int j = 1; j < words.size() - 1; ++j){
				std::string cpyWord2(words[j]);
				StringHelper::removeLeadingAndTrailingWhiteSpaces(cpyWord2);
				if(cpyWord2 == "until"){
					printErrorAndQuit("The amount of tree condition can not be used with a time condition in the same line!");
				}
			}
		}
	}
	// first word is name, start with 1
	for(unsigned int i = 1; i < words.size() - 1; ++i){
		std::string cpyWord(words[i]);
		StringHelper::removeLeadingAndTrailingWhiteSpaces(cpyWord);
		if(cpyWord == "for" || cpyWord == "until"){
			// find ending of for:
			int breakValue = -1;
			for(unsigned int j = 1; j < words.size() - 1; ++j){
				std::string cpyWord2(words[j]);
				StringHelper::removeLeadingAndTrailingWhiteSpaces(cpyWord2);
				if(cpyWord2 == "with" || cpyWord2 == "withonly"){
					breakValue = j;
				}
			}
			if(breakValue == -1){
				breakValue = (int) words.size();
			}
			std::string nrType;
			for(unsigned int k = i + 1; k < breakValue ; ++k){
				nrType += words[k] + " ";
			}
			addType(cpyWord, nrType);
		}
	}
	// search for memory constraint
	for(unsigned int i = 1; i < words.size() - 1; ++i){
		std::string cpyWord(words[i]);
		StringHelper::removeLeadingAndTrailingWhiteSpaces(cpyWord);
		if(cpyWord == "with" || cpyWord == "withonly"){
			unsigned int start = i;
			if(words[i + 1] == "only"){
				++start;
			}
			// find ending of with:
			int breakValue = -1;
			for(unsigned int j = start; j < words.size() - 1; ++j){
				std::string cpyWord2(words[j]);
				StringHelper::removeLeadingAndTrailingWhiteSpaces(cpyWord2);
				if(cpyWord2 == "for" || cpyWord2 == "until"){
					breakValue = j;
				}
			}
			if(breakValue == -1){
				breakValue = (int) words.size();
			}
			std::string nrType;
			for(unsigned int k = start + 1; k < breakValue; ++k){
				nrType += words[k] + " ";
			}
			addType(cpyWord, nrType);
		}
	}
}

void TestInformation::Instruction::replaceScopeVariables(std::string& definition){
	if(m_mode == TestMode::FOR){
		auto pos = definition.find("$");
		if(pos != definition.npos){
			if(m_range.size() > 0){
				std::string sub = definition.substr(pos + 1, m_varName.length());
				if(sub == m_varName){
					// replace with current number
					definition.replace(pos, m_varName.length() + 1,
									   StringHelper::number2String(m_range[m_currentValue]));
					replaceScopeVariables(definition);
				}else if(m_scope != nullptr){
					m_scope->replaceScopeVariables(definition);
				}
			}else{
				printError("The range is used before it was initialized!");
			}
		}
	}else{
		printErrorAndQuit("This function can only be called on FOR objects");
	}
}

bool TestInformation::Instruction::operator==(const TestInformation::Instruction& instruction) const{
	return instruction.m_mode == m_mode && instruction.m_varName == m_varName &&
		   instruction.m_currentValue == m_currentValue && instruction.m_scope == m_scope &&
		   instruction.m_amountOfTrees == m_amountOfTrees && instruction.m_exitMode == m_exitMode &&
		   instruction.m_memory == m_memory && instruction.m_seconds == m_seconds &&
		   instruction.m_range == m_range;
}
