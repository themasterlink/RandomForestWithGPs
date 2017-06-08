//
// Created by denn_ma on 5/31/17.
//

#include "TestInformation.h"


TestInformation::TestInformation(){
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


void TestInformation::addDefinitionOrInstruction(const TestMode mode, const std::string& def){
	std::vector<std::string> words;
	StringHelper::getWords(def, words);
	switch(mode){
		case TestMode::LOAD:{
			if(words.size() == 1 && (m_definitions.find(words[0]) != m_definitions.end() || words[0] == "all")){
				m_instructions.emplace_back(mode, words[0]);
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
					if(m_definitions.find(words[i + 1]) != m_definitions.end()){
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
							foundClasses = decipherClasses(words, i + 2, j - 1, defineName.m_classes);
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
			m_definitions.emplace(defineName.getVarName(), defineName);
			break;
		}
		case TestMode::REMOVE: {
			printError("Not implemented yet");
			break;
		}
		case TestMode::COMBINE: {
			if(words.size() == 5 && words[1] == "with" && words[3] == "in"){
				const std::string name = getDefinition(words[0]).getVarName();
				if(m_definitions.find(name) != m_definitions.end()){
					const std::string name2 = getDefinition(words[2]).getVarName();
					if(m_definitions.find(name2) != m_definitions.end()){
						TestDefineName defineName;
						defineName.setVarName(words[4]);
						defineName.m_firstFromVariable = words[0];
						defineName.m_secondFromVariable= words[2];
						defineName.m_includeClasses = false;
						const std::string name = getDefinition(words[0]).getVarName();
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
				const std::string name = getDefinition(words[0]).getVarName();
				if(m_definitions.find(name) != m_definitions.end()){
					m_instructions.emplace_back(mode, words[0]);
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
				const std::string name = getDefinition(words[0]).getVarName();
				if(m_definitions.find(name) != m_definitions.end()){
					m_instructions.emplace_back(mode, words[0]);
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
					try{
						defineName.m_splitAmount = std::stoi(words[2]);
					}catch(std::exception &e){
						printErrorAndQuit("The split number: " << words[2] << " is no number!");
					}
					m_definitions.emplace(defineName.getVarName(), defineName);
					m_instructions.emplace_back(mode, words[0]);
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

bool TestInformation::decipherClasses(const std::vector<std::string> &words, const unsigned int start,
									  const unsigned int end, std::vector<unsigned int> &usedClasses){
	usedClasses.clear();
	if(words[start] == "all"){
		usedClasses.push_back((unsigned int) UNDEF_CLASS_LABEL);
		return true;
	}
	if(StringHelper::startsWith(words[start],'{') && StringHelper::endsWith(words[end], '}')){
		// more than one class
		// split words even more
		std::vector<std::string> numbers;
		for(unsigned int i = start; i <= end; ++i){
			if(words[i] != "{" && words[i] != "}"){
				std::string word(words[i]);
				if(StringHelper::startsWith(word, ",")){
					word = word.substr(1, word.length() - 1);
				}
				while(word.length() > 0){
					StringHelper::removeLeadingWhiteSpaces(word);
					StringHelper::sizeType pos = word.find(',');
					if(pos != word.npos && pos != 0){
						numbers.emplace_back(word.substr(0,pos));
						word = word.substr(pos + 1, word.length() - pos);
					}else{
						numbers.emplace_back(word);
						word = "";
					}
				}
			}
		}
		if(StringHelper::startsWith(numbers[0], '{')){
			numbers[0] = numbers[0].substr(1, numbers[0].length() - 1);
			if(numbers.front().length() == 0){
				for(unsigned int i = 0; i < numbers.size() - 1; ++i){
					numbers[i] = numbers[i + 1];
				}
				numbers.pop_back();
			}
		}
		if(StringHelper::startsWith(numbers.back(), '}')){
			numbers[numbers.size() - 1] = numbers.back().substr(0, numbers.back().length() - 1);
			if(numbers.back().length() == 0){
				numbers.pop_back();
			}
		}
		unsigned int lastUsedClass = 0;
		for(unsigned int i = 0; i < numbers.size(); ++i){
			if(numbers[i] == "..."){
				// get next and use all in between
				unsigned int tillClass = lastUsedClass;
				if(i + 1 < numbers.size()){
					try{
						tillClass = (unsigned int) std::stoi(numbers[i + 1]);
					}catch(std::exception &e){
						printError("The word could not be transferred to a class: " << numbers[i + 1]);
						return false;
					}
				}else{
					printError("The ... can not be the end, there must be an ending class");
					return false;
				}
				for(unsigned int k = lastUsedClass + 1; k < tillClass; ++k){
					usedClasses.emplace_back(k);
				}
			}else{
				try{
					usedClasses.emplace_back(std::stoi(numbers[i]));
				}catch(std::exception &e){
					printError("The word could not be transferred to a class: " << numbers[i]);
					return false;
				}
			}
			lastUsedClass = usedClasses.back();
		}
	}else if(start == end){
		try{
			usedClasses.emplace_back(std::stoi(words[start]));
		}catch(std::exception& e){
			printError("The word could not be transferred to a class: " << words[start]);
			return false;
		}
	}
	return usedClasses.size() != 0;

}

TestInformation::TestDefineName TestInformation::getDefinition(const std::string &name){
	auto it = m_definitions.find(name);
	if(it != m_definitions.end()){
		return it->second;
	}
	// search for split value
	auto pos = name.find('[');
	if(pos != name.npos){
		it = m_definitions.find(name.substr(0, pos));
		if(it != m_definitions.end()){
			return it->second;
		}
	}
	printErrorAndQuit("This data set does not exist: " << name);
	return TestDefineName();
}

TestInformation::Instruction TestInformation::getInstruction(const unsigned int i){
	if(i < m_instructions.size()){
		return m_instructions[i];
	}
	return Instruction(TestMode::UNDEFINED, "");
}

int TestInformation::getSplitNumber(const std::string& name){
	auto start = name.find('[');
	auto end = name.find(']');
	if(start != name.npos && end != name.npos){
		++start;
		try{
			return std::stoi(name.substr(start, end - start));
		}catch(std::exception& e){
			printErrorAndQuit("The name: " << name << " does not contain a split number!");
		}
	}
	return -1;
}


bool TestInformation::TestDefineName::isTrainOrTestSetting() const{
	return m_varName == TestInformation::trainSettingName || m_varName == TestInformation::testSettingName;
}

void TestInformation::TestDefineName::useAllClasses(){
	m_classes.clear();
	m_classes.push_back((unsigned int) UNDEF_CLASS_LABEL);
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
		try{
			m_seconds = std::stof(cpyNrType);
		}catch(std::exception &e){
			printErrorAndQuit("The time: " << cpyNrType << " is no real value!");
		}
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
		try{
			m_memory = (MemoryType) std::stof(cpyNrType);
		}catch(std::exception &e){
			printErrorAndQuit("The memory value: " << cpyNrType << " is not a number!");
		}
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
		try{
			m_amountOfTrees = (unsigned int) std::stoi(cpyNrType);
		}catch(std::exception &e){
			printErrorAndQuit("The memory value: " << cpyNrType << " is not a number!");
		}
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
