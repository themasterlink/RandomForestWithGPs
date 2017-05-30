//============================================================================
// Name        : RandomForestWithGPs.cpp
// Author      : 
// Version     :
// Copyright   : 
// Description :
//============================================================================

#include "Tests/tests.h"
#include <boost/program_options.hpp>
#include "Base/CommandSettings.h"
#include "Base/Settings.h"
#include "Base/Logger.h"
#include "Base/ThreadMaster.h"
#include <panel.h>

/*#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
*/
#include "Data/DataBinaryWriter.h"
#include "Base/ScreenOutput.h"
//#include <src/cmaes_interface.h>

#include <ctime>
#include <iostream>
#include <string>
#include <boost/asio.hpp>
#include <boost/array.hpp>
#include "Tests/TestManager.h"

/*
void compress(const std::string& path){
	StopWatch sw;
	// read in Settings
	DataSets dataSets;
	int trainAmount = 400;
	Settings::getValue("MultiBinaryGP.trainingAmount", trainAmount);
	int testAmount = 100;
	Settings::getValue("MultiBinaryGP.testingAmount", testAmount);
	const bool readTxt = true;
	DataReader::readFromFiles(dataSets, path, trainAmount + testAmount, readTxt);
	for(DataSets::const_iterator it = dataSets.begin(); it != dataSets.end(); ++it){
		std::string outPath = "../realTest/" + it->first + "/vectors.binary";
		DataBinaryWriter::toFile(it->second, outPath);
	}
	std::cout << "Time needed for compressing: " << sw.elapsedAsPrettyTime() << std::endl;
}*/

void handleProgrammOptions(int ac, char* av[]){
	CommandSettings::init();
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
	};
	CommandSettings::setValues(vm);
	boost::program_options::notify(vm);
	if (vm.count("help")) {
		std::cout << desc << "\n";
		quitApplication();
	}
}

	/*const cv::Mat input = cv::imread("../dog.jpg", CV_LOAD_IMAGE_COLOR);
	std::vector<cv::KeyPoint> keypoints;
	cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> detector = cv::xfeatures2d::SiftFeatureDetector::create();
	detector->detect(input, keypoints);

	// Add results to image and save.
	cv::Mat output;
	cv::drawKeypoints(input, keypoints, output);
	cv::imwrite("../sift_result.jpg", output);
*/

std::string make_daytime_string(){
  std::time_t now = std::time(0);
  return std::ctime(&now);
}

//void socketsTest(){
//	try
//	{
//		// Any program that uses asio need to have at least one io_service object
//		boost::asio::io_service io_service;
//
//		// acceptor object needs to be created to listen for new connections
//		boost::asio::ip::tcp::acceptor acceptor(io_service, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 13));
//
//		for (;;){
//			// creates a socket
//			boost::asio::ip::tcp::socket socket(io_service);
//
//			// wait and listen
//			acceptor.accept(socket);
//			std::cout << "huh" << std::endl;
//
//			// prepare message to send back to client
//			std::string message = make_daytime_string();
//
//			boost::system::error_code ignored_error;
//
//			// writing the message for current time
//			boost::asio::write(socket, boost::asio::buffer(message), ignored_error);
//		}
//	}
//	catch (std::exception& e)
//	{
//		std::cerr << e.what() << std::endl;
//	}
//}
//
//void clientTest(char* av[]){
//	try
//	{
//		// the user should specify the server - the 2nd argument
//
//		// Any program that uses asio need to have at least one io_service object
//		boost::asio::io_service io_service;
//
//		// Convert the server name that was specified as a parameter to the application, to a TCP endpoint.
//		// To do this, we use an ip::tcp::resolver object.
//		boost::asio::ip::tcp::resolver resolver(io_service);
//
//		// A resolver takes a query object and turns it into a list of endpoints.
//		// We construct a query using the name of the server, specified in argv[1],
//		// and the name of the service, in this case "daytime".
//		boost::asio::ip::tcp::resolver::query query(av[1], "daytime");
//
//		// The list of endpoints is returned using an iterator of type ip::tcp::resolver::iterator.
//		// A default constructed ip::tcp::resolver::iterator object can be used as an end iterator.
//		boost::asio::ip::tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);
//
//		// Now we create and connect the socket.
//		// The list of endpoints obtained above may contain both IPv4 and IPv6 endpoints,
//		// so we need to try each of them until we find one that works.
//		// This keeps the client program independent of a specific IP version.
//		// The boost::asio::connect() function does this for us automatically.
//		// The connection is open. All we need to do now is read the response from the daytime service.
//		while(true){
//			boost::asio::ip::tcp::socket socket(io_service);
//			boost::asio::connect(socket, endpoint_iterator);
//
//			while(true){
//				// We use a boost::array to hold the received data.
//				boost::array<char, 128> buf;
//				boost::system::error_code error;
//				std::cout << "ask for something" << std::endl;
//				// The boost::asio::buffer() function automatically determines
//				// the size of the array to help prevent buffer overruns.
//				size_t len = socket.read_some(boost::asio::buffer(buf), error);
//
//				// When the server closes the connection,
//				// the ip::tcp::socket::read_some() function will exit with the boost::asio::error::eof error,
//				// which is how we know to exit the loop.
//				if (error == boost::asio::error::eof)
//					break; // Connection closed cleanly by peer.
//				//				else if (error)
//				//					throw boost::system::system_error(error); // Some other error.
//
//				std::cout.write(buf.data(), len);
//			}
//			sleepFor(1);
//		}
//	}
//	// handle any exceptions that may have been thrown.
//	catch (std::exception& e)
//	{
//		std::cerr << e.what() << std::endl;
//	}
//}


int main(int ac, char** av){
//	system("cd \"Debug OpenCV2\"");
//	system("pwd");
#ifdef DEBUG
//	printOnScreen("Debug does not use inputParams!");
	std::vector<std::string> input = {"RandomForest", "--samplingAndTraining", "180", "--useFakeData"}; //
	ac = input.size();
	av = new char*[ac];
	for(int i = 0; i < ac; ++i){
		av[i] = const_cast<char*>(input[i].c_str());
	}
#endif

	TestManager::init("../Settings/testSettings.init");
 	exit(0);
	printOnScreen("Start");
//	getchar();
	handleProgrammOptions(ac,av);
//	cmaes::cmaes_t evo; /* an CMA-ES type struct or "object" */
//	cmaes::cmaes_boundary_transformation_t cmaesBoundaries;
//	double *m_arFunvals, *m_hyperParamsValues;
//	double lowerBounds[] = {0.2,0.2};
//	double upperBounds[] = {25., 25.0};
//	int nb_bounds = 2; /* numbers used from lower and upperBounds */
//	unsigned long dimension;
//
//	double *const*pop;
//	/* initialize boundaries, be sure that initialSigma is smaller than upper minus lower bound */
//	cmaes::cmaes_boundary_transformation_init(&cmaesBoundaries, lowerBounds, upperBounds, nb_bounds);
//	/* Initialize everything into the struct evo, 0 means default */
//	const int seed = 12389;
//	m_arFunvals = cmaes::cmaes_init(&evo, 0, nullptr, nullptr, seed, 0, "../Settings/cmaes_initials.par");
//	dimension = (unsigned long) cmaes::cmaes_Get(&evo, "dimension");
//	if(dimension != nb_bounds){
//		printError("The dimension in the settings does not fit!");
//	}
//	m_hyperParamsValues = cmaes::cmaes_NewDouble(dimension); /* calloc another vector */
//	const int sampleLambda = cmaes::cmaes_Get(&evo, "lambda");
//	std::list<Vector2> points;
//	std::list<Real> values;
//	for(unsigned int k = 0; k < 50; ++k){
//		/* generate lambda new search points, sample population */
//
//		pop = cmaes::cmaes_SamplePopulation(&evo); /* do not change content of pop */
//
//		/* transform into bounds and evaluate the new search points */
//		for(int i = 0; i < sampleLambda; ++i) {
//			//							const double corr = m_package->correctlyClassified();
//			//							const double probDiff = corr < 60. ? 0. : corr < 80 ? 0.1 : corr < 90 ? 0.2 : 0.3;
//			cmaes::cmaes_boundary_transformation(&cmaesBoundaries, pop[i], m_hyperParamsValues, dimension);
//			m_arFunvals[i] =  (((m_hyperParamsValues[0] - 10.) * (m_hyperParamsValues[0] - 10.) + (m_hyperParamsValues[1] - 10.) + (m_hyperParamsValues[1] - 10.)));
//			values.push_back(m_arFunvals[i]);
//			points.push_back(Vector2(m_hyperParamsValues[0], m_hyperParamsValues[1]));
//		}
//		cmaes::cmaes_UpdateDistribution(&evo, m_arFunvals);  /* assumes that pop[i] has not been modified */
//	}
//	DataWriterForVisu::writePointsIn2D("test.svg", points, values);
//	openFileInViewer("test.svg");
//	exit(0);
//	if(ac > 1){
//		clientTest(av);
//	}else{
//		socketsTest();
//	}
//	return 0;
	const std::string settingsFile = CommandSettings::get_settingsFile();
	Settings::init(settingsFile);
	ThreadMaster::start(); // must be performed after Settings init!
	ScreenOutput::start(); // should be started after ThreadMaster and Settings
	ClassKnowledge::init();
//	std::cout << RESET << "Start" << std::endl;

	if(CommandSettings::get_onlyDataView()){
		const int firstPoints = 10000000; // all points
		TotalStorage::readData(firstPoints);
		OnlineStorage<LabeledVectorX*> train;
		OnlineStorage<LabeledVectorX*> test;
		// starts the training by its own
		TotalStorage::getOnlineStorageCopyWithTest(train, test, TotalStorage::getTotalSize());
		printOnScreen("TotalStorage::getTotalSize(): " << TotalStorage::getTotalSize());
		DataWriterForVisu::writeSvg("justData.svg", train.storage());
		openFileInViewer("justData.svg");
		const bool wait = false;
		quitApplication(wait);
	}else if(CommandSettings::get_convertFile().length() > 0){
		printOnScreen("Convert file mode:");
		LabeledData data;
		const std::string inputPath = CommandSettings::get_convertFile();
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
	Logger::start();
	printOnScreen("Settingsfile: " << settingsFile);
	CommandSettings::printAllSettingsToLog();
	if(CommandSettings::get_samplingAndTraining() > 0){
		printOnScreen("Training time: " << TimeFrame(CommandSettings::get_samplingAndTraining()));
	}
//	VectorX testVec = VectorX::Random(1000);
//	VectorX testVec2 = VectorX::Random(1000);
//	Real res = 0;
//	for(unsigned int k = 0; k < 100000; ++k){
//		for(unsigned int i = 0; i < 1000; ++i){
//
//		}
//	}


	/*
    const int nr = 300;
    Matrix Sigma, controlSigma;
    controlSigma = Matrix::Random(nr, nr);
    for(int i = 0; i < nr; ++i){
    	for(int j = 0; j < nr; ++j){
    		controlSigma(i,j) = i * nr + j;
    	}
    }
    Sigma = controlSigma;
    const Real deltaTau = 1.0;
    StopWatch sigmaUp, sigmaUpNew;
    for(int t = 0; t < 1000; ++t){
    	int i = t % nr;
    	sigmaUp.startTime();
    	VectorX si = Sigma.col(i);
    	Real denom = 1.0 + deltaTau * si[i];
    	//if(fabs(denom) > EPSILON)
    	Sigma -= (deltaTau / denom) * (si * si.transpose());
    	sigmaUp.recordActTime();
    	sigmaUpNew.startTime();
    	denom = 1.0 + deltaTau * controlSigma(i,i); // <=> 1.0 + deltaTau[i] * si[i] for si = Sigma.col(i)
    	const Real fac = deltaTau / denom;
    	const VectorX oldSigmaRow = controlSigma.col(i);
    	for(int p = 0; p < nr; ++p){
    		controlSigma(p,p) -= fac * oldSigmaRow[p] * oldSigmaRow[p];
    		for(int q = p + 1; q < nr; ++q){
    			const Real sub = fac * oldSigmaRow[p] * oldSigmaRow[q];
    			controlSigma(p,q) -= sub;
    			controlSigma(q,p) -= sub;
    		}
    	}
    	sigmaUpNew.recordActTime();
    	for(int p = 0; p < nr; ++p){
    		for(int q = 0; q < nr; ++q){
    			if(fabs(controlSigma(p,q) - Sigma(p,q)) > fabs(Sigma(p,q)) * EPSILON){
    				printError("Calc is wrong!" << p << ", " << q << ": " << fabs(controlSigma(p,q) - Sigma(p,q)));
    				p = q = nr;
    			}
    		}
    	}
    }
    std::cout << "sigma up time: " << sigmaUp.elapsedAvgAsPrettyTime() << std::endl;
    std::cout << "total sigma up time: " << sigmaUp.elapsedAvgAsTimeFrame() * ((Real) 10)<< std::endl;
    std::cout << "new sigma up time: " << sigmaUpNew.elapsedAvgAsPrettyTime() << std::endl;
    std::cout << "total new sigma up time: " << sigmaUpNew.elapsedAvgAsTimeFrame() * ((Real) 10)<< std::endl;
*/
	std::string type;
	Settings::getValue("main.type", type);
	StopWatch sw;
	if(type == "binaryIvm"){
		executeForBinaryClassIVM();
		printOnScreen("For IVM: " << sw.elapsedAsTimeFrame());
	}else if(type == "multiIvm"){
		executeForMutliClassIVM();
		printOnScreen("For IVMs: " << sw.elapsedAsTimeFrame());
	}else if(type == "binaryORF"){
		executeForBinaryClassORF();
		printOnScreen("For binary ORFs: " << sw.elapsedAsTimeFrame());
	}else if(type == "ORF"){
		executeForBinaryClassORF();
		printOnScreen("For ORFs: " << sw.elapsedAsTimeFrame());
	}else if(type == "ORFIVMs"){
		executeForBinaryClassORFIVM();
		printOnScreen("For ORFIVMs: " << sw.elapsedAsTimeFrame());
	}else{
		printError("Type \"main.type\" can only be binaryIvm, multiIvm or ORF not: " << type);
	}
	quitApplication();

	return 0;
}

