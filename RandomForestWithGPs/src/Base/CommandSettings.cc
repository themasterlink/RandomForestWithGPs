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
	for(auto it = vm.begin(); it != vm.end(); ++it){
		if(it->first == "help"){ continue; }
		bool found = false;
		for(auto itParam = m_params.begin(); itParam != m_params.end(); ++itParam){
			if(it->first == itParam->name && !it->second.empty()){
				found = true;
				const std::string type = itParam->type();
				if(type == "bool"){
					*(bool*)itParam->ref = !*(bool*)(itParam->ref); // it is there -> flip default
				}else if(type == "int"){
					*(int*)itParam->ref = (int) vm[itParam->name].as<int>();
				}else if(type == "Real"){
					*((Real*)itParam->ref) = vm[itParam->name].as<Real>();
				}else if(type == "string" || type == "std::string"){
					*(std::string*)itParam->ref = (std::string) vm[itParam->name].as<std::string>();
				}
				break;
			}
		}
		if(!found){
			printError("The given param: " << it->first << ", was not defined in the CommandSettings class!");
		}
	}
}

void CommandSettings::printAllSettingsToLog(){
	std::stringstream line;
	line << "Program was started with:";
	for(auto itParam = m_params.begin(); itParam != m_params.end(); ++itParam){
		line << " ";
		const std::string type = itParam->type();
		if(type == "bool"){
			if(*(bool*)itParam->ref){
				line << itParam->name;
			}
		}else if(type == "int"){
			line << itParam->name << " " << *(int*)itParam->ref;
		}else if(type == "Real"){
			line << itParam->name << " " << *(Real*)itParam->ref;
		}else if(type == "string" || type == "std::string"){
			line << itParam->name << " " << *(std::string*)itParam->ref;
		}else{
			printError("Unknown type: " << type);
		}
	}
	Logger::addSpecialLineToFile(line.str(), "CommandSettings");
}
