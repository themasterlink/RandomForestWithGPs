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
#include <algorithm>
#include "../Base/Types.h"
#include "StopWatch.h"
#include "InLinePercentageFiller.h"
#include "../Base/ScreenOutput.h"
#include "../Base/Logger.h"
#include <boost/filesystem.hpp>

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

static const auto UNDEF_CLASS_LABEL = 99999999u;

static const auto EPSILON = Real(1e-15);

#define SAVE_DELETE(pointer) \
	delete(pointer); \
	pointer = nullptr \

// for not implemented functions and params which have no use in a function, because for example they are inherited
#define UNUSED(expr) \
 	 (void)(expr) \

inline std::string number2String(const Real& in, const int precision = -1){
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

inline std::string convertMemorySpace(MemoryType mem){
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

template<typename T>
inline std::string number2String(const T& in){
	std::stringstream ss;
	ss << in;
	return ss.str();
}

inline void openFileInViewer(const std::string& filename){
	if(boost::filesystem::exists(Logger::getActDirectory() + filename)){
		Logger::addSpecialLineToFile("eog " + filename, "System");
		system(("eog " + Logger::getActDirectory() + filename).c_str());
	}
}
// much fast than pow(2, exp)
// template ensures that it can be used with any kind of int, unsigned or not
template<typename T>
constexpr inline T pow2(const T exponent) noexcept {
	return ((unsigned int) T(1)) << exponent;
}

// for this case the pow(2.0, exponent) should be used and not this constexpr function
template<>
constexpr inline Real pow2(const Real exponent) noexcept = delete;

template<>
constexpr inline char pow2(const char exponent) noexcept = delete; // prohibits the use with charbegin,

template<>
constexpr inline bool pow2(const bool exponent) noexcept = delete; // prohibits the use with char

template<class T>
inline auto argMax(const T& begin, const T& end){
	return std::distance(begin, std::max_element(begin, end));
}

template<class T>
inline auto argMin(const T& begin, const T& end){
	return std::distance(begin, std::min_element(begin, end));
}

template<class T>
void overwriteConst(const T& ref, const T& newValue){
	T* iPointer = const_cast<T*>(&ref);
	*iPointer = newValue;
}

inline bool startsWith(const std::string& word, const std::string& cmp){
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

inline bool endsWith(const std::string& word, const std::string& cmp){
	if(word.size() > cmp.size()){
		int t = word.size() - 1;
		if(cmp.size() == 0){
			return false;
		}
		for(int i = cmp.size() - 1; i > -1; --i, --t){
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

template<class T>
inline void sleepFor(const T seconds){
	usleep((__useconds_t) seconds * (__useconds_t) 1e6);
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

//template<typename _Tp>
//struct __MakeUniq
//{ typedef unique_ptr<_Tp> __single_object; };
//
//
//template<typename _Tp, typename... _Args>
//inline typename
//__MakeUniq<_Tp>::__single_object makeUnique(_Args&&... __args)
//{ return std::unique_ptr<_Tp>(new _Tp(std::forward<_Args>(__args)...)); }


#endif /* UTILITY_UTIL_H_ */
