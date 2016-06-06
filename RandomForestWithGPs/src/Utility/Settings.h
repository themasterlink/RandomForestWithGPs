/*
 * Settings.h
 *
 *  Created on: 04.06.2016
 *      Author: Max
 */

#ifndef UTILITY_SETTINGS_H_
#define UTILITY_SETTINGS_H_

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/thread.hpp> // Boost mutex
#include "Util.h"

class Settings{

public:
	static void init(const std::string& settingsfile);

	static void writeDefaultFile(const std::string& settingsfile);

	template<typename T>
	static void getValue(const std::string& nameOfValue, T& value,
			const T& defaultValue);

	template<typename T>
	static void getValue(const std::string& nameOfValue, T& value);

	static void getValue(const std::string& nameOfValue, bool& value,
			const bool& defaultValue);

	static void getValue(const std::string& nameOfValue, bool& value);


private:
	Settings();
	virtual ~Settings();

	static boost::property_tree::ptree m_root;

	static boost::mutex m_mutex;

	static bool m_init;
};

template<typename T>
void Settings::getValue(const std::string& nameOfValue, T& value,
		const T& defaultValue){
	if(m_init){
		m_mutex.lock();
		if(boost::optional<T> ret = m_root.get_optional<T>(nameOfValue)){
			value = *ret;
		}else{
			printWarning("This name was not in the init file: " << nameOfValue);
			value = defaultValue;
		}
		m_mutex.unlock();
	}else{
		value = defaultValue;
	}
}

template<typename T>
void Settings::getValue(const std::string& nameOfValue, T& value){
	if(m_init){
		m_mutex.lock();
		if(boost::optional<T> ret = m_root.get_optional<T>(nameOfValue)){
			value = *ret;
		}else{
			printWarning("This name was not in the init file: " << nameOfValue);
		}
		m_mutex.unlock();
	}
}

#endif /* UTILITY_SETTINGS_H_ */