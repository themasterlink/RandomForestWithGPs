//
// Created by denn_ma on 8/31/17.
//

#include "GlobalLifeTimeMeasurement.h"

void GlobalLifeTimeMeasurement::dieTreeId(unsigned int treeId){
	m_mutex.lock();
	for(auto it = m_livingList.begin(); it != m_livingList.end(); ++it){
		if(it->first == treeId){
			m_deathList.emplace_back(it->second.second, LifeTimePerformance(it->second.first, m_roundCounter));
			m_livingList.erase(it);
			break;
		}
	}
	m_mutex.unlock();
}

void GlobalLifeTimeMeasurement::setPerformance(unsigned int treeId, const Real performance){
	m_mutex.lock();
	for(auto it = m_livingList.begin(); it != m_livingList.end(); ++it){
		if(it->first == treeId){
			it->second.first = performance;
			break;
		}
	}
	m_mutex.unlock();
}

unsigned int GlobalLifeTimeMeasurement::addNewTreeId(){
	m_mutex.lock();
	unsigned int ret = m_idCounter;
	++m_idCounter;
	m_livingList.emplace_back(ret, LifeTimePerformance(0, m_roundCounter));
	m_mutex.unlock();
	return ret;
}

void GlobalLifeTimeMeasurement::endAllTrees(){
	m_mutex.lock();
	for(auto& pair : m_livingList){
		m_deathList.emplace_back(pair.second.second, LifeTimePerformance(pair.second.first, m_roundCounter));
	}
	m_mutex.unlock();
}

void GlobalLifeTimeMeasurement::writeToFile(const std::string& fileName){
	m_mutex.lock();
	const std::string& filePath = Logger::instance().getActDirectory() + fileName;
	std::string output = "start,performance,end\n";
	for(auto& pair : m_deathList){
		std::stringstream str2;
		str2 << pair.first << "," << pair.second.first << "," << pair.second.second << "\n";
		output += str2.str();
	}
	std::ofstream file;
	file.open(filePath, std::ofstream::out | std::ofstream::trunc);
	file.write(output.c_str(), output.length());
	file.close();
	m_mutex.unlock();
}