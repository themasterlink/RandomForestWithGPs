//
// Created by denn_ma on 5/30/17.
//

#include "TestManager.h"

void TestManager::init(const std::string &filePath){
	std::fstream file(filePath,std::ios::in);
	if(file.is_open()){
		std::string line;
		while(std::getline(file, line)){
			StringHelper::removeStartAndEndingWhiteSpaces(line);
			if(line.length() > 2){
				TestMode mode = findMode(line);
				std::cout << "line: \"" << line << "\"" << std::endl;
				std::vector<std::string> restWords;
				StringHelper::getWords(line, restWords);
				for(auto& word : restWords){
					std::cout << "word: \"" << word << "\"" << std::endl;
				}
				switch(mode){
					case TestMode::DEFINE:
						break;
					case TestMode::LOAD:
						break;
					case TestMode::TRAIN:
						break;
					case TestMode::TEST:
						break;
					case TestMode::UNDEFINED:
						break;
				}
			}
		}
	}

}

TestManager::TestMode TestManager::findMode(std::string &line){
	std::string firstWord = StringHelper::getFirstWord(line);
	if(firstWord.length() > 0){
		if(firstWord == "load"){
			return TestMode::LOAD;
		}else if(firstWord == "train"){
			return TestMode::TRAIN;
		}else if(firstWord == "test"){
			return TestMode::TEST;
		}else if(firstWord == "define"){
			return TestMode::DEFINE;
		}
	}
	return TestMode::UNDEFINED;
}

void TestManager::run(){

}
