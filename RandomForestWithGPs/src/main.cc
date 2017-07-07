//============================================================================
// Name        : RandomForestWithGPs.cpp
// Author      : 
// Version     :
// Copyright   : 
// Description :
//============================================================================

#include "Tests/tests.h"
#include "Tests/TestManager.h"
#include "Data/DataBinaryWriter.h"
#include "Data/DataReader.h"

void handleProgrammOptions(int ac, char* av[]){
	CommandSettings::instance().init();
	boost::program_options::options_description desc("Allowed options");
	desc.add_options()
	        		("help", "produce help message")
					("useFakeData", "use fake data")
					("visuRes", boost::program_options::value<int>()->default_value(0), "visualize something, if possible")
					("visuResSimple", boost::program_options::value<int>()->default_value(0), "visualize something, if possible")
					("onlyDataView", "only visualize the data, no training performed")
					("samplingAndTraining", boost::program_options::value<Real>()->default_value(0), "sample and train the hyper params, else just use be configured params")
					("plotHistos", "should some histogramms be plotted")
					("settingsFile", boost::program_options::value<std::string>()->default_value("../Settings/init.json"), "Give the filepath of the settingsfile")
					("convertFile", boost::program_options::value<std::string>()->default_value(""), "Give the filepath of the desired file which should be converted into binary")
					;
	boost::program_options::variables_map vm;
	try{
		boost::program_options::store(boost::program_options::parse_command_line(ac, av, desc), vm);
	} catch (std::exception& e) {
		std::cout << "The given program options are wrong: " << e.what() << std::endl;
		quitApplication();
	};
	CommandSettings::instance().setValues(vm);
	boost::program_options::notify(vm);
	if (vm.count("help")) {
		std::cout << desc << "\n";
		quitApplication();
	}
}

void doOnlyDataView(){
	if(CommandSettings::instance().get_onlyDataView()){
		const int firstPoints = 10000000; // all points
		TotalStorage::instance().readData(firstPoints);
		OnlineStorage<LabeledVectorX*> train;
		OnlineStorage<LabeledVectorX*> test;
		// starts the training by its own
		TotalStorage::instance().getOnlineStorageCopyWithTest(train, test, TotalStorage::instance().getTotalSize());
		printOnScreen("TotalStorage::instance().getTotalSize(): " << TotalStorage::instance().getTotalSize());
		DataWriterForVisu::writeSvg("justData.svg", train.storage());
		openFileInViewer("justData.svg");
		const bool wait = false;
		quitApplication(wait);
	}else if(CommandSettings::instance().get_convertFile().length() > 0){
		printOnScreen("Convert file mode:");
		LabeledData data;
		const std::string inputPath = CommandSettings::instance().get_convertFile();
		const std::string typeLessPath = inputPath.substr(0, inputPath.length() - 4); // for txt and csv
		if(!boost::filesystem::exists(boost::filesystem::path(typeLessPath + ".binary"))){
			const auto containDegrees = true;
			DataReader::readFromFile(data, typeLessPath, INT_MAX, UNDEF_CLASS_LABEL, true, containDegrees);
			if(data.size() > 0){
				printOnScreen("Data amount: " << data.size() << ", dim: " << data[0]->rows());
				DataBinaryWriter::toFile(data, typeLessPath + ".binary");
			}
		}else{
			printOnScreen("This file was already converted!");
		}
		quitApplication();
	}
}

int main(int ac, char** av){
	printOnScreen("Start");
#ifdef USE_OPEN_CV
	printOnScreen("OpenCv was used");
#else
	printOnScreen("OpenCv was not used");
#endif
	handleProgrammOptions(ac,av);
	TestManager::instance().setFilePath("../Settings/testSettingsPy.init"); // must be called before the logger
	const std::string settingsFile = CommandSettings::instance().get_settingsFile();
	Settings::instance().init(settingsFile);
	ThreadMaster::instance().start(); // must be performed after Settings init!
	ScreenOutput::instance().start(); // should be started after ThreadMaster and Settings
	ClassKnowledge::instance().init();

	doOnlyDataView();

	Logger::instance().start();
	printOnScreen("Settingsfile: " << settingsFile);
	CommandSettings::instance().printAllSettingsToLog();
	if(CommandSettings::instance().get_samplingAndTraining() > 0){
		printOnScreen("Training time: " << TimeFrame(CommandSettings::instance().get_samplingAndTraining()));
	}

	TestManager::instance().init();
	TestManager::instance().run();
	quitApplication();
	return 0;
}

