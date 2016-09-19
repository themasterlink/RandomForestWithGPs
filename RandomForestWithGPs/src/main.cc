//============================================================================
// Name        : RandomForestWithGPs.cpp
// Author      : 
// Version     :
// Copyright   : 
// Description :
//============================================================================

#include "Tests/tests.h"
#include "Utility/Settings.h"
#include <boost/program_options.hpp>

#include "Data/DataBinaryWriter.h"

void compress(const std::string& path){
	StopWatch sw;
	// read in Settings
	DataSets dataSets;
	DataReader::readFromFiles(dataSets, path, 500);
	for(DataSets::const_iterator it = dataSets.begin(); it != dataSets.end(); ++it){
		std::string outPath = "../realTest/" + it->first + "/vectors.binary";
		DataBinaryWriter::toFile(it->second, outPath);
	}
	std::cout << "Time needed for compressing: " << sw.elapsedAsPrettyTime() << std::endl;
}

int main(int ac, char* av[]){

	boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("compress", "set compression level")
        ("useFakeData", "use fake data")
    ;
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(ac, av, desc), vm);
    boost::program_options::notify(vm);
    if (vm.count("help")) {
    	std::cout << desc << "\n";
    	return 0;
    }
	std::cout << "Start" << std::endl;
    Settings::init("../Settings/init.json");
    std::string path;
    Settings::getValue("RealData.folderPath", path);
    if(vm.count("compress")){
    	compress(path);
    	return 0;
    }

	bool useGP;
	Settings::getValue("OnlyGp.useGP", useGP);
	if(useGP){
		executeForBinaryClass(path, !vm.count("useFakeData"));
		return 0;
	}

	return 0;
}

