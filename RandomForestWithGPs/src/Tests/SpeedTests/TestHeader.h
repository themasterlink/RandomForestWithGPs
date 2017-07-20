//
// Created by denn_ma on 7/20/17.
//

#ifndef RANDOMFORESTWITHGPS_TESTHEADER_H
#define RANDOMFORESTWITHGPS_TESTHEADER_H

#include <regex>
#include "ArgMaxAndMin.h"
#include "EigenVectorX.h"
#include "RandomNumbers.h"

namespace SpeedTests {

	void runTest(const std::string& nameOfTest){
		std::regex eigenVsVectorAndArray("(\\s)*(array|vector|eigen|Array|Vector|Eigen)(\\s)*((vs|Vs)(.)?(\\s)*(array|vector|eigen|Array|Vector|Eigen))?(\\s)*");
		std::regex randomNr("(\\s)*(Random|random)(\\s)*(nr|Nr)?(\\s)*");
		std::regex argMaxAndMin("(\\s)*(arg|Arg)(\\s)*(max|Max)((\\s)*(and|And)?(\\s)*(min|Min))?(\\s)*");
		std::regex all("(\\s)*(all|All)(\\s)*((test|Test)(s)?)?(\\s)*");
		if(std::regex_match(nameOfTest, all)){
			EigenVsVector::runAll();
			ArgMax::runAll();
			RandomNumber::runAll();
		}else if(std::regex_match(nameOfTest, eigenVsVectorAndArray)){
			EigenVsVector::runAll();
		}else if(std::regex_match(nameOfTest, argMaxAndMin)){
			ArgMax::runAll();
		}else if(std::regex_match(nameOfTest, randomNr)){
			RandomNumber::runAll();
		}else{
			std::cout << "This test case is not known!" << std::endl;
		}
	}

}

#endif //RANDOMFORESTWITHGPS_TESTHEADER_H
