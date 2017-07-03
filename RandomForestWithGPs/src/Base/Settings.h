/*
 * Settings.h
 *
 *  Created on: 04.06.2016
 *      Author: Max
 */

#ifndef BASE_SETTINGS_H_
#define BASE_SETTINGS_H_

#include "../Utility/Util.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/thread.hpp> // Boost mutex

class Settings{

SINGELTON_MACRO(Settings);

public:
	void init(const std::string& settingsfile);

	void writeDefaultFile(const std::string& settingsfile);

	template<typename T>
	void getValue(const std::string& nameOfValue, T& value,
				  const T& defaultValue);

	template<typename T>
	void getValue(const std::string& nameOfValue, T& value);

	template<typename T>
	T getDirectValue(const std::string& nameOfValue);

	void getValue(const std::string& nameOfValue, bool& value,
				  const bool& defaultValue);

	void getValue(const std::string& nameOfValue, bool& value);

	bool getDirectBoolValue(const std::string& nameOfValue);

	Real getDirectRealValue(const std::string& nameOfValue);

	std::string getFilePath(){ return m_filePath; };

private:

	boost::property_tree::ptree m_root;

	Mutex m_mutex;

	bool m_init;

	std::string m_filePath;
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
			printError("This name was not in the init file: " << nameOfValue);
		}
		m_mutex.unlock();
	}
}

template<typename T>
T Settings::getDirectValue(const std::string& nameOfValue){
	T value;
	getValue(nameOfValue, value);
	return value;
}

#endif /* BASE_SETTINGS_H_ */
