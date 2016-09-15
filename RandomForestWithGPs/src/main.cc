//============================================================================
// Name        : RandomForestWithGPs.cpp
// Author      : 
// Version     :
// Copyright   : 
// Description :
//============================================================================

#include "Tests/tests.h"
#include "Utility/Settings.h"

int main(){

	std::cout << "Start" << std::endl;
	// read in Settings
	Settings::init("../Settings/init.json");
	std::string path;
	Settings::getValue("RealData.folderPath", path);


	bool useGP;
	Settings::getValue("OnlyGp.useGP", useGP);
	if(useGP){
		executeForBinaryClass(path);
		return 0;
	}

	return 0;
}

