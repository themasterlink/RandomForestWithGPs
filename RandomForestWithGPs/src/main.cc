//============================================================================
// Name        : RandomForestWithGPs.cpp
// Author      : 
// Version     :
// Copyright   : 
// Description :
//============================================================================

#include "Tests/tests.h"
#include <boost/program_options.hpp>

#include "Base/Settings.h"
/*#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
*/
#include "Data/DataBinaryWriter.h"

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
}

int main(int ac, char* av[]){
	/*const cv::Mat input = cv::imread("../dog.jpg", CV_LOAD_IMAGE_COLOR);
	std::vector<cv::KeyPoint> keypoints;
	cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> detector = cv::xfeatures2d::SiftFeatureDetector::create();
	detector->detect(input, keypoints);

	// Add results to image and save.
	cv::Mat output;
	cv::drawKeypoints(input, keypoints, output);
	cv::imwrite("../sift_result.jpg", output);
*/
	boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("compress", "set compression level")
        ("useFakeData", "use fake data")
        ("visu", "visualize something, if possible")
    ;
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(ac, av, desc), vm);
    boost::program_options::notify(vm);
    if (vm.count("help")) {
    	std::cout << desc << "\n";
    	return 0;
    }
	std::cout << RESET << "Start" << std::endl;
	Settings::init("../Settings/init.json");
    std::string path;
    Settings::getValue("RealData.folderPath", path);
    if(vm.count("compress")){
    	compress(path);
    	return 0;
    }
/*
    const int nr = 300;
    Eigen::MatrixXd Sigma, controlSigma;
    controlSigma = Eigen::MatrixXd::Random(nr, nr);
    for(int i = 0; i < nr; ++i){
    	for(int j = 0; j < nr; ++j){
    		controlSigma(i,j) = i * nr + j;
    	}
    }
    Sigma = controlSigma;
    const double deltaTau = 1.0;
    StopWatch sigmaUp, sigmaUpNew;
    for(int t = 0; t < 1000; ++t){
    	int i = t % nr;
    	sigmaUp.startTime();
    	Eigen::VectorXd si = Sigma.col(i);
    	double denom = 1.0 + deltaTau * si[i];
    	//if(fabs(denom) > EPSILON)
    	Sigma -= (deltaTau / denom) * (si * si.transpose());
    	sigmaUp.recordActTime();
    	sigmaUpNew.startTime();
    	denom = 1.0 + deltaTau * controlSigma(i,i); // <=> 1.0 + deltaTau[i] * si[i] for si = Sigma.col(i)
    	const double fac = deltaTau / denom;
    	const Eigen::VectorXd oldSigmaRow = controlSigma.col(i);
    	for(int p = 0; p < nr; ++p){
    		controlSigma(p,p) -= fac * oldSigmaRow[p] * oldSigmaRow[p];
    		for(int q = p + 1; q < nr; ++q){
    			const double sub = fac * oldSigmaRow[p] * oldSigmaRow[q];
    			controlSigma(p,q) -= sub;
    			controlSigma(q,p) -= sub;
    		}
    	}
    	sigmaUpNew.recordActTime();
    	for(int p = 0; p < nr; ++p){
    		for(int q = 0; q < nr; ++q){
    			if(fabs(controlSigma(p,q) - Sigma(p,q)) > fabs(Sigma(p,q)) * 1e-5){
    				printError("Calc is wrong!" << p << ", " << q << ": " << fabs(controlSigma(p,q) - Sigma(p,q)));
    				p = q = nr;
    			}
    		}
    	}
    }
    std::cout << "sigma up time: " << sigmaUp.elapsedAvgAsPrettyTime() << std::endl;
    std::cout << "total sigma up time: " << sigmaUp.elapsedAvgAsTimeFrame() * ((double) 10)<< std::endl;
    std::cout << "new sigma up time: " << sigmaUpNew.elapsedAvgAsPrettyTime() << std::endl;
    std::cout << "total new sigma up time: " << sigmaUpNew.elapsedAvgAsTimeFrame() * ((double) 10)<< std::endl;
*/
	bool useGP;
	Settings::getValue("OnlyGp.useGP", useGP);
	if(useGP){
		StopWatch sw;
		executeForBinaryClassIVM(path, !vm.count("useFakeData"), vm.count("visu"));
		std::cout << "For IVM: " << sw.elapsedAsTimeFrame() << std::endl;
		/*sw.startTime();
		executeForBinaryClass(path, !vm.count("useFakeData"));
		std::cout << "For GP: " << sw.elapsedAsTimeFrame() << std::endl;*/
		return 0;
	}

	return 0;
}

