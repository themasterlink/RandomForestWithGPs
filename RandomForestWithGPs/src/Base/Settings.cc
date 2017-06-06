/*
 * Settings.cc
 *
 *  Created on: 04.06.2016
 *      Author: Max
 */

#include "../Base/Settings.h"

boost::property_tree::ptree Settings::m_root;
boost::mutex Settings::m_mutex;
bool Settings::m_init = false;
std::string Settings::m_filePath = "";

Settings::Settings(){
	// TODO Auto-generated constructor stub

}

Settings::~Settings(){
	// TODO Auto-generated destructor stub
}

void Settings::init(const std::string& settingsfile){
	m_filePath = settingsfile;
	m_mutex.lock();
	try{
		boost::property_tree::read_json(settingsfile, m_root);
	} catch(std::exception const& e){
		printError(e.what());
		m_mutex.unlock();
		return;
	}
	m_init = true;
	m_mutex.unlock();
}

void Settings::getValue(const std::string& nameOfValue, bool& value,
		const bool& defaultValue){
	if(m_init){
		m_mutex.lock();
		if(boost::optional<std::string> ret = m_root.get_optional<std::string>(nameOfValue)){
			if(*ret == "true" || *ret == "1"){
				value = true;
			}else if(*ret == "false" || *ret == "0"){
				value = false;
			}else{
				printWarning("This bool was wrong formatted: " << nameOfValue << ", this formatting is not supported: " << *ret);
				value = defaultValue;
			}
		}else{
			printWarning("This name was not in the init file: " << nameOfValue);
			value = defaultValue;
		}
		m_mutex.unlock();
	}else{
		value = defaultValue;
	}
}

void Settings::getValue(const std::string& nameOfValue, bool& value){
	if(m_init){
		m_mutex.lock();
		if(boost::optional<std::string> ret = m_root.get_optional<std::string>(nameOfValue)){
			if(*ret == "true" || *ret == "1"){
				value = true;
			}else if(*ret == "false" || *ret == "0"){
				value = false;
			}else{
				printWarning("This bool was wrong formatted: " << nameOfValue << ", this formatting is not supported: " << *ret << ", value is now false!");
				value = false;
			}
		}else{
			printWarning("This name was not in the init file: " << nameOfValue);
		}
		m_mutex.unlock();
	}
}

bool Settings::getDirectBoolValue(const std::string& nameOfValue){
	bool value = false;
	if(m_init){
		m_mutex.lock();
		if(boost::optional<std::string> ret = m_root.get_optional<std::string>(nameOfValue)){
			if(*ret == "true" || *ret == "1"){
				value = true;
			}else if(*ret == "false" || *ret == "0"){
				value = false;
			}else{
				printWarning("This bool was wrong formatted: " << nameOfValue << ", this formatting is not supported: " << *ret << ", value is now false!");
				value = false;
			}
		}else{
			printWarning("This name was not in the init file: " << nameOfValue);
		}
		m_mutex.unlock();
	}
	return value;
}


Real Settings::getDirectRealValue(const std::string &nameOfValue){
	Real value;
	getValue(nameOfValue, value);
	return value;
}


void Settings::writeDefaultFile(const std::string& settingsfile){
	boost::property_tree::ptree root;
	root.put("OnlineRandomForest.amountOfTrainedTrees", 200);
	root.put("OnlineRandomForest.Tree.height", 6);

	root.put("Training.path", "../testData/testInput2.txt");
	root.put("Test.path", "../testData/testInput3.txt");
	root.put("Write2D.path", "../testData/trainedResult2.txt");
	root.put("Write2D.doWriting", true);
	boost::property_tree::write_json(settingsfile, root);
}
