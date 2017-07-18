/*
 * Util.h
 *
 *  Created on: 31.05.2016
 *      Author: Max
 */

#ifndef UTILITY_UTIL_H_
#define UTILITY_UTIL_H_

#include "StringHelper.h"
#include <algorithm>
#include "../Base/Types.h"
#include "StopWatch.h"
#include "InLinePercentageFiller.h"
#include "../Base/ScreenOutput.h"
#include "../Base/Logger.h"
#include "../Base/CommandSettings.h"
#include <boost/filesystem.hpp>

static const std::string RESET("\033[0m");
static const std::string BLACK("\033[30m");      /* Black */
static const std::string RED("\033[31m");        /* Red */
static const std::string GREEN("\033[32m");      /* Green */
static const std::string YELLOW("\033[33m");     /* Yellow */
static const std::string BLUE("\033[34m");       /* Blue */
static const std::string MAGENTA("\033[35m");    /* Magenta */
static const std::string CYAN("\033[36m");       /* Cyan */
static const std::string WHITE("\033[37m");      /* White */

template<typename T>
void saveDelete(T*& pointer){
	delete(pointer);
	pointer = nullptr;
}

// for not implemented functions and params which have no use in a function, because for example they are inherited
#define UNUSED(expr) \
     (void)(expr) \

inline void openFileInViewer(const std::string& filename){
	if(boost::filesystem::exists(Logger::instance().getActDirectory() + filename)){
		Logger::instance().addSpecialLineToFile("eog " + filename + " &", "System");
		system(("eog " + Logger::instance().getActDirectory() + filename + " &> /dev/null &").c_str());
	}
}

// much fast than pow(2, exp)
// template ensures that it can be used with any kind of int, unsigned or not
template<typename T>
constexpr inline T pow2(const T exponent) noexcept{
	return (T(1)) << exponent;
}

// for this case the pow(2.0, exponent) should be used and not this constexpr function
template<>
constexpr inline Real pow2(const Real exponent) noexcept = delete;

template<>
constexpr inline char pow2(const char exponent) noexcept = delete; // prohibits the use with char

template<>
constexpr inline bool pow2(const bool exponent) noexcept = delete; // prohibits the use with bool

template<class T, class internSizeType = typename T::size_type>
inline internSizeType argMax(const T& container){
	using internT = typename T::value_type;
	auto max = std::numeric_limits<internT>::lowest();
	internSizeType value = 0;
	internSizeType it = 0;
	for(const auto& ele: container){
		if(ele > max){
			max = ele;
			value = it;
		}
		++it;
	}
	return value;
}

template<class T, class internSizeType = typename T::size_type>
inline internSizeType argMin(const T& container){
	using internT = typename T::value_type;
	auto min = std::numeric_limits<internT>::max();
	internSizeType value = 0;
	internSizeType it = 0;
	for(const auto& ele: container){
		if(ele < min){
			min = ele;
			value = it;
		}
		++it;
	}
	return value;
}

template<class T>
constexpr void overwriteConst(const T& ref, const T& newValue){
	T* iPointer = const_cast<T*>(&ref);
	*iPointer = newValue;
}

template<class T>
inline void sleepFor(const T seconds){
#if(BUILD_SYSTEM_CMAKE == 1)
	usleep((__useconds_t) (seconds * (Real) 1e6));
#else
	usleep((useconds_t) (seconds * (Real) 1e6));
#endif
}

inline void systemCall(const std::string& call){
	system(call.c_str());
}

inline void quitApplication(const bool wait = true){
	Logger::instance().forcedWrite();
//	if(wait){
//		if(CommandSettings::instance().get_settingsFile() == CommandSettings::instance().defaultvalue_settingsFile()){
//			printOnScreen("Press any key to quit application");
//			getchar();
//		}
//	}
	sleepFor(0.5);
	ThreadMaster::instance().stopExecution();
	ThreadMaster::instance().blockUntilFinished();
	sleepFor(0.5);
	ScreenOutput::instance().quitForScreenMode();
	exit(0);
}

class VerboseMode : public Singleton<VerboseMode> {

	friend class Singleton<VerboseMode>;
public:

	void setVerboseLevel(unsigned int verboseLevel){ m_verboseLevel = verboseLevel; }

	unsigned int getVerboseLevel() const { return m_verboseLevel; }

	const bool isVerboseLevelHigher() const { return getVerboseLevel() != 0; }

private:

	VerboseMode() : m_verboseLevel(0) {};
	virtual ~VerboseMode() = default;

	unsigned int m_verboseLevel;
};

#ifdef USE_SCREEN_OUPUT

#define printError(message) \
    do{    std::stringstream str3; str3 << "Error in " << __PRETTY_FUNCTION__ << ":" << StringHelper::number2String(__LINE__) << ": " << message; ScreenOutput::instance().printErrorLine(str3.str()); }while(false) \

#define printWarning(message) \
    printOnScreen("Warning in " << __PRETTY_FUNCTION__ << ":" << StringHelper::number2String(__LINE__) << ": " << message) \

#define printLine() \
    printOnScreen("Debug in " << __PRETTY_FUNCTION__ << ":" << StringHelper::number2String(__LINE__)) \

#define printDebug(message) \
    printOnScreen("Debug in " << __PRETTY_FUNCTION__ << ":" << StringHelper::number2String(__LINE__) << ": " << message) \

#else

#define printError(message) \
	printOnScreen(RED << "Error in " << __PRETTY_FUNCTION__ << ":" << StringHelper::number2String(__LINE__) << ": " << message << RESET) \

#define printWarning(message) \
	 printOnScreen("Warning in " << __PRETTY_FUNCTION__ << ":" << StringHelper::number2String(__LINE__) << ": " << message) \

#define printLine() \
	 printOnScreen("Debug in " << __PRETTY_FUNCTION__ << ":" << StringHelper::number2String(__LINE__)) \

#define printDebug(message) \
	 printOnScreen("Debug in " << __PRETTY_FUNCTION__ << ":" << StringHelper::number2String(__LINE__) << ": " << message) \

#endif

#define printErrorAndQuit(message) \
        printError(message); quitApplication()

inline int_fast32_t highEndian2LowEndian(int_fast32_t i){
	unsigned char c1, c2, c3, c4;

	c1 = (unsigned char) (i & 255);
	c2 = (unsigned char) ((i >> 8) & 255);
	c3 = (unsigned char) ((i >> 16) & 255);
	c4 = (unsigned char) ((i >> 24) & 255);

	return ((int_fast32_t) c1 << 24) + ((int_fast32_t) c2 << 16) + ((int_fast32_t) c3 << 8) + c4;
}

//template<typename _Tp>
//struct __MakeUniq
//{ typedef unique_ptr<_Tp> __single_object; };
//
//
//template<typename _Tp, typename... _Args>
//inline typename
//__MakeUniq<_Tp>::__single_object makeUnique(_Args&&... __args)
//{ return UniquePtr<_Tp>(new _Tp(std::forward<_Args>(__args)...)); }

inline Real sqrtReal(const Real val){
#ifdef USE_DOUBLE
	return sqrt(val);
#else
	return sqrtf(val);
#endif
}

inline Real logReal(const Real val){
#ifdef USE_DOUBLE
	return log(val);
#else
	return logf(val);
#endif
}

constexpr Real operator""_r(long double val){
#ifdef USE_DOUBLE
	return val;
#else
	return (Real) val;
#endif
}

inline Real expReal(const Real val){
#ifdef USE_DOUBLE
	return exp(val);
#else
	return expf(val);
#endif
}

inline Real absReal(const Real val){
#ifdef USE_DOUBLE
	return fabs(val);
#else
	return fabsf(val);
#endif
}

inline Real modReal(const Real val, const Real mod){
#ifdef USE_DOUBLE
	return fmod(val, mod);
#else
	return fmodf(val, mod);
#endif
}

#define lockStatementWith(statement, mutex) \
    mutex.lock(); statement; mutex.unlock() \

#define lockStatementWithSave(statement, variable, mutex) \
    mutex.lock(); variable = statement; mutex.unlock() \

#define CALL_MEMBER_FCT(object, ptrToMember) ((object).*(ptrToMember)) // change to std::invoke in c++ 17

static const auto UNDEF_CLASS_LABEL = (unsigned int) pow2(16) - 3;

static const auto EPSILON = Real(1e-15);


#endif /* UTILITY_UTIL_H_ */
