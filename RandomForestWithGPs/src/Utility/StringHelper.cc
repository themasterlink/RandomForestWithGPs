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
			if(in < 10000.){
				char buffer[10 + precision];
				std::stringstream str;
				str << "%." << precision << "f";
				sprintf(buffer, str.str().c_str(), in);
				return std::string(buffer);
			}else{
				char buffer[350 + precision]; // higher should be impossible
				std::stringstream str;
				str << "%." << precision << "f";
				sprintf(buffer, str.str().c_str(), in);
				return std::string(buffer);
			}
		}else{
			std::stringstream ss;
			ss << in;
			return ss.str();
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

} // close namespace