//
// Created by denn_ma on 5/30/17.
//

#ifndef RANDOMFORESTWITHGPS_STRINGHELPER_H
#define RANDOMFORESTWITHGPS_STRINGHELPER_H

#include <sstream>
#include <fstream>
#include <istream>
#include <iostream>
#include <vector>
#include "../Base/BaseType.h"

namespace StringHelper {

	using sizeType = std::string::size_type;

	template<typename T>
	inline std::string number2String(const T& in){
		std::stringstream ss;
		ss << in;
		return ss.str();
	}

	std::string number2String(const Real& in, const int precision = -1);

	std::string convertMemorySpace(MemoryType mem);

	bool startsWith(const std::string& word, const std::string& cmp);

	bool endsWith(const std::string& word, const std::string& cmp);

	bool startsWith(const std::string& word, const char cmp);

	bool endsWith(const std::string& word, const char cmp);

	void removeLeadingWhiteSpaces(std::string& line);

	void removeTrailingWhiteSpaces(std::string& line);

	void removeLeadingAndTrailingWhiteSpaces(std::string& line);

	std::string getFirstWord(std::string& line);

	void getWords(const std::string& line, std::vector<std::string>& words);

	void removeCommentFromLine(std::string& line, const char commentSymbol = '#');

};

#endif //RANDOMFORESTWITHGPS_STRINGHELPER_H
