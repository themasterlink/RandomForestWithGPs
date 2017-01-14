/*
 * UsefulStuff.h
 *
 *  Created on: 31.05.2016
 *      Author: Max
 */

#ifndef UTILITY_UTIL_H_
#define UTILITY_UTIL_H_

#include <iostream>
#include <sstream>
#include <fstream>
#include <istream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include "StopWatch.h"
#include "InLinePercentageFiller.h"
#include "../Base/ScreenOutput.h"
#include "../Base/Logger.h"
#include "boost/filesystem.hpp"

#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

#define EPSILON 1e-15

#define NEG_DBL_MAX -DBL_MAX

#define SAVE_DELETE(pointer) \
	delete(pointer); \
	pointer = nullptr \

// for not implemented functions and params which have no use in a function, because for example they are inherited
#define UNUSED(expr) \
 	 (void)(expr) \

inline std::string number2String(const double& in, const int precision = -1){
	if(precision != -1){
		if(in < 10000.){
			char buffer[10 + precision];
			std::string format = "%."+number2String(precision)+ "f";
			sprintf(buffer,format.c_str(), in);
			std::stringstream ss;
			ss << buffer;
			return ss.str();
		}else{
			char buffer[350 + precision]; // higher should be impossible
			std::string format = "%."+number2String(precision)+ "f";
			sprintf(buffer,format.c_str(), in);
			std::stringstream ss;
			ss << buffer;
			return ss.str();
		}
	}else{
		std::stringstream ss;
		ss << in;
		return ss.str();
	}
}

template<typename T>
inline std::string number2String(const T& in){
	std::stringstream ss;
	ss << in;
	return ss.str();
}

inline void openFileInViewer(const std::string& filename){
	if(boost::filesystem::exists(Logger::getActDirectory() + filename)){
		Logger::addSpecialLineToFile("open " + filename, "System");
		system(("open " + Logger::getActDirectory() + filename).c_str());
	}
}
// much fast than pow(2, exp)
inline unsigned int pow2(const unsigned int exponent){
	return ((unsigned int) 1u) << exponent;
}

inline bool endsWith(const std::string& first, const std::string& second){
	if(first.size() > second.size()){
		int t = first.size() - 1;
		if(second.size() == 0){
			return false;
		}
		for(int i = second.size() - 1; i > -1; --i, --t){
			if(second[i] != first[t]){
				return false;
			}
		}
		return true;
	}else if(first.size() < second.size()){
		int t = second.size() - 1;
		if(first.size() == 0){
			return false;
		}
		for(int i = first.size() - 1; i > -1; --i, --t){
			if(second[t] != first[i]){
				return false;
			}
		}
		return true;
	}else{
		return first == second;
	}
}

#ifdef USE_SCREEN_OUPUT

#define printError(message) \
	do{	std::stringstream str; str << "Error in " << __PRETTY_FUNCTION__ << ":" << number2String(__LINE__) << ": " << message; ScreenOutput::printErrorLine(str.str()); }while(false) \

#define printWarning(message) \
	printOnScreen("Warning in " << __PRETTY_FUNCTION__ << ":" << number2String(__LINE__) << ": " << message) \

#define printLine() \
	printOnScreen("Debug in " << __PRETTY_FUNCTION__ << ":" << number2String(__LINE__)) \

#define printDebug(message) \
	printOnScreen("Debug in " << __PRETTY_FUNCTION__ << ":" << number2String(__LINE__) << ": " << message) \

#else

#define printError(message) \
	std::cout << RED << "Error in " << __PRETTY_FUNCTION__ << ":" << number2String(__LINE__) << ": " << message << RESET << std::endl; \

#define printWarning(message) \
	 std::cout << "Warning in " << __PRETTY_FUNCTION__ << ":" << number2String(__LINE__) << ": " << message << std::endl;\

#define printLine() \
	 std::cout << "Debug in " << __PRETTY_FUNCTION__ << ":" << number2String(__LINE__) << std::endl; \

#define printDebug(message) \
	 std::cout << "Debug in " << __PRETTY_FUNCTION__ << ":" << number2String(__LINE__) << ": " << message << std::endl; \

#endif

inline int_fast32_t highEndian2LowEndian(int_fast32_t i) {
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int_fast32_t)c1 << 24) + ((int_fast32_t)c2 << 16) + ((int_fast32_t)c3 << 8) + c4;
}

#endif /* UTILITY_UTIL_H_ */
