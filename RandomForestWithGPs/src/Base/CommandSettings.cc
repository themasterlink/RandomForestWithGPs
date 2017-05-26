/*
 * CommandSettings.cc
 *
 *  Created on: 27.10.2016
 *      Author: Max
 */


#include "CommandSettings.h"
#include "../Utility/Util.h"

std::list<Param> CommandSettings::m_params;
DEFINE_PARAM(bool, useFakeData);
DEFINE_PARAM(int, visuRes);
DEFINE_PARAM(int, visuResSimple);
DEFINE_PARAM(bool, onlyDataView);
DEFINE_PARAM(Real, samplingAndTraining);
DEFINE_PARAM(bool, plotHistos);
DEFINE_PARAM(std::string, settingsFile);
DEFINE_PARAM(std::string, convertFile);

Param::Param(std::string name, const std::string (*type)(), void* ref){
	this->name = name;
	this->type = type;
	this->ref = ref;
}

void CommandSettings::init(){
	INIT_PARAM(bool, useFakeData);
	INIT_PARAM(int, visuRes);
	INIT_PARAM(int, visuResSimple);
	INIT_PARAM(bool, onlyDataView);
	INIT_PARAM(Real, samplingAndTraining);
	INIT_PARAM(bool, plotHistos);
	INIT_PARAM(std::string, settingsFile);
	INIT_PARAM(std::string, convertFile);
}

void CommandSettings::setValues(boost::program_options::variables_map& vm){
	for(auto& vmEle : vm){
		if(vmEle.first == "help"){ continue; }
		bool found = false;
		for(auto& param : m_params){
			if(vmEle.first == param.name && !vmEle.second.empty()){
				found = true;
				const std::string type = param.type();
				if(type == "bool"){
					*(bool*)param.ref = !*(bool*)(param.ref); // it is there -> flip default
				}else if(type == "int"){
					*(int*)param.ref = (int) vm[param.name].as<int>();
				}else if(type == "Real"){
					*((Real*)param.ref) = vm[param.name].as<Real>();
				}else if(type == "string" || type == "std::string"){
					*(std::string*)param.ref = (std::string) vm[param.name].as<std::string>();
				}
				break;
			}
		}
		if(!found){
			printError("The given param: " << vmEle.first << ", was not defined in the CommandSettings class!");
		}
	}
}

void CommandSettings::printAllSettingsToLog(){
	std::stringstream line;
	line << "Program was started with:";
	for(auto& param : m_params){
		line << " ";
		const std::string type = param.type();
		if(type == "bool"){
			if(*(bool*)param.ref){
				line << param.name;
			}
		}else if(type == "int"){
			line << param.name << " " << *(int*)param.ref;
		}else if(type == "Real"){
			line << param.name << " " << *(Real*)param.ref;
		}else if(type == "string" || type == "std::string"){
			line << param.name << " " << *(std::string*)param.ref;
		}else{
			printError("Unknown type: " << type);
		}
	}
	Logger::addSpecialLineToFile(line.str(), "CommandSettings");
}
