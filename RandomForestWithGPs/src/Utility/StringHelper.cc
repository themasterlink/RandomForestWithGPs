//
// Created by denn_ma on 5/31/17.
//

#include "StringHelper.h"
#include <boost/date_time.hpp>

namespace StringHelper{

	std::string getFirstWord(std::string &line){
		sizeType pos = line.find(' ');
		if(pos != line.npos){
			const std::string firstWord(std::move(line.substr(0, pos)));
			line = line.substr(pos + 1, line.length() - pos);
			return firstWord;
		}else if(line.length() > 0){
// move line to it -> no copy necessary
			const std::string firstWord(std::move(line));
			line = "";
			return firstWord;
		}
		return "";
	}

	void getWords(const std::string &line, std::vector<std::string> &words){
		std::string copyLine(line);
		while(copyLine.length() > 0){
			std::string firstWord = getFirstWord(copyLine);
			removeLeadingAndTrailingWhiteSpaces(firstWord);
			if(firstWord.length() > 0){
				words.emplace_back(firstWord);
			}else{
				break;
			}
		}
	}

	void removeCommentFromLine(std::string &line, const char commentSymbol){
		sizeType pos = line.find(commentSymbol);
		if(pos != line.npos){
			line = line.substr(0, pos);
			// remove whitespaces between comment symbol and message
			removeTrailingWhiteSpaces(line);
		}
	}

	void removeLeadingAndTrailingWhiteSpaces(std::string& line){
		removeLeadingWhiteSpaces(line);
		removeTrailingWhiteSpaces(line);
	}

	void removeTrailingWhiteSpaces(std::string& line){
		long i = (long) (line.length() - 1);
		for(; i >= 0; --i){
			if(line[i] != ' ' && line[i] != '\t'){
				line = line.substr(0, (unsigned long) (i + 1));
				return;
			}
		}
	}

	void removeLeadingWhiteSpaces(std::string& line){
		sizeType i = 0;
		for(; i < line.length(); ++i){
			if(line[i] != ' ' && line[i] != '\t'){
				line = line.substr(i, line.length() - i);
				return;
			}
		}
	}

	bool startsWith(const std::string &word, const std::string &cmp){
		if(word.size() > cmp.size()){
			if(cmp.size() == 0){
				return false;
			}
			int t = 0;
			for(int i = 0; i < cmp.size(); ++i, ++t){
				if(cmp[i] != word[t]){
					return false;
				}
			}
			return true;
		}else if(word.size() < cmp.size()){
			return false;
		}else{
			return word == cmp;
		}
	}

	bool endsWith(const std::string &word, const std::string &cmp){
		if(word.size() > cmp.size()){
			int t = (int) (word.size() - 1);
			if(cmp.size() == 0){
				return false;
			}
			for(int i = (int) (cmp.size() - 1); i > -1; --i, --t){
				if(cmp[i] != word[t]){
					return false;
				}
			}
			return true;
		}else if(word.size() < cmp.size()){
			return false;
		}else{
			return word == cmp;
		}
	}

	bool startsWith(const std::string &word, const char cmp){
		return word.length() > 0 && word.front() == cmp;
	}

	bool endsWith(const std::string &word, const char cmp){
		return word.length() > 0 && word.back() == cmp;
	}

	std::string number2String(const Real &in, const int precision){
		if(precision > 0){
			std::stringstream str2;
			str2.setf(std::ios::fixed);
			str2 << std::setprecision(precision) << in;
			return str2.str();
		}else{
			std::stringstream str2;
			str2 << in;
			return str2.str();
		}
	}

	std::string convertMemorySpace(MemoryType mem){
		std::stringstream ss;
		bool useSpace = false;
		std::vector<std::string> names = {" GB", " MB", " kB", " B"};
		unsigned int i = 0;
		for(MemoryType val = 1000000000; val > 0; val /= 1000){
			if(mem >= val){
				if(useSpace){
					ss << " ";
				}
				ss << mem / val << names[i];
				mem %= val;
				useSpace = true;
			}
			++i;
		}
		return ss.str();
	}

	std::string getActualTimeOfDayAsString(){
		boost::gregorian::date dayte(boost::gregorian::day_clock::local_day());
		boost::posix_time::ptime midnight(dayte);
		boost::posix_time::ptime now(boost::posix_time::microsec_clock::local_time());
		boost::posix_time::time_duration td = now - midnight;
		std::stringstream clockTime;
		if(td.fractional_seconds() > 0){
			const char cFracSec = StringHelper::number2String(td.fractional_seconds())[0];
			clockTime << td.hours() << ":" << td.minutes() << ":" << td.seconds() << "." << cFracSec;
		}else{
			clockTime << td.hours() << ":" << td.minutes() << ":" << td.seconds() << "." << 0;
		}
		return clockTime.str();
	}

	bool isEqualTrue(const std::string& string){
		return string == "true" || string == "1" || string == "True";
	}

	bool isEqualFalse(const std::string& string){
		return string == "false" || string == "0" || string == "false";
	}

	bool decipherRange(const std::vector<std::string>& words, const unsigned int start, const unsigned int end,
					   std::vector<unsigned int>& resultingRange){
		resultingRange.clear();
		if(words[start] == "all"){
			resultingRange.emplace_back(UNDEF_CLASS_LABEL);
			return true;
		}
		if(StringHelper::startsWith(words[start], '{') && StringHelper::endsWith(words[end], '}')){
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
							numbers.emplace_back(word.substr(0, pos));
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
					int tillClass = lastUsedClass;
					if(i + 1 < numbers.size()){
						string2Int(numbers[i + 1], tillClass,
								   "The word could not be transferred to a class: " << numbers[i + 1]);
					}else{
						printError("The ... can not be the end, there must be an ending class");
						return false;
					}
					for(unsigned int k = lastUsedClass + 1; k < tillClass; ++k){
						resultingRange.emplace_back(k);
					}
				}else{
					int value;
					string2Int(numbers[i], value, "The word could not be transferred to a class: " << numbers[i]);
					resultingRange.emplace_back(value);
				}
				lastUsedClass = resultingRange.back();
			}
		}else if(start == end){
			int value;
			string2Int(words[start], value, "The word could not be transferred to a class: " << words[start]);
			resultingRange.emplace_back(value);
		}
		return resultingRange.size() != 0;
	}

	bool decipherRange(const std::string& rangeLine, const unsigned int start, const unsigned int end,
					   std::vector<unsigned int>& resultingRange){
		std::vector<std::string> words;
		getWords(rangeLine, words);
		return decipherRange(words, start, end, resultingRange);
	}

	namespace Intern {

		void string2IntIntern(const std::string& intStr, int& intValue, const std::string& errorMsg){
			try{
				intValue = std::stoi(intStr);
			}catch(std::exception& e){
				printErrorAndQuit(errorMsg);
			}
		}

		void string2RealIntern(const std::string& realStr, Real& realValue, const std::string& errorMsg){
			try{
				realValue = (Real) std::stod(realStr);
			}catch(std::exception& e){
				printErrorAndQuit(errorMsg);
			}
		}
	};
} // close namespace