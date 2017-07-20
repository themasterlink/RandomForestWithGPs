//
// Created by denn_ma on 7/20/17.
//

#ifndef RANDOMFORESTWITHGPS_RANDOMNUMBERS_H
#define RANDOMFORESTWITHGPS_RANDOMNUMBERS_H

#include "../../RandomNumberGenerator/RandomUniformNr.h"

namespace SpeedTests {

	namespace RandomNumber {

		Real generateRandomRealNr(unsigned int& seed){
			seed *= 16807;
			return (Real)seed * 4.6566129e-010f;
		}

		void testRandomUniformNr(){
			int iTest = 0;
			RandomUniformNr nr(0,100,1231);
			const unsigned int tries = 500000000;
			StopWatch sw;
			for(unsigned int i = 0; i < tries; ++i){
				iTest += nr();
			}
			std::cout << "Time for random uniform nr: " << sw.elapsedSeconds() << ", " << iTest << std::endl;
			unsigned int iUTest = 0;
			RandomUniformUnsignedNr unr(100,1231);
			StopWatch sw2;
			for(unsigned int i = 0; i < tries; ++i){
				iUTest += unr();
			}
			std::cout << "Time for random unsigned uniform nr: " << sw2.elapsedSeconds() << ", " << iUTest << std::endl;

			unsigned int iUTest2 = 0;
			GeneratorType gen;
			std::uniform_int_distribution<unsigned int> dis2;
			StopWatch sw2_1;
			for(unsigned int i = 0; i < tries; ++i){
				iUTest2 += dis2(gen);
			}
			std::cout << "Time for random unsigned uniform nr: " << sw2_1.elapsedSeconds() << ", " << iUTest2 << std::endl;


			std::uniform_real_distribution<Real> dis;
			Real val = 0.0_r;
			std::vector<Real> minMax(100000);
			StopWatch sw3;
			for(unsigned int i = 0; i < 100000; ++i){
				minMax[i] = dis(gen);
			}
			Real min, max;
			DataConverter::getMinMax(minMax, min, max);
			std::cout << "Min: " << min << ", max: " << max << std::endl;
			std::cout << "Time for real random uniform nr (std): " << sw3.elapsedSeconds() << ", " << val << std::endl;
			val = 0.0_r;
			unsigned int seed = 1239;
			StopWatch sw4;
			for(unsigned int i = 0; i < 100000; ++i){
				seed *= 16807;
				minMax[i] = (Real) seed * 4.6566129e-010f;
			}
			DataConverter::getMinMax(minMax, min, max);
			std::cout << "Min: " << min << ", max: " << max << std::endl;
			std::cout << "Time for real random uniform nr (std): " << sw4.elapsedSeconds() << ", " << val << std::endl;



		}

		void runAll(){
			testRandomUniformNr();
		}

	}

}


#endif //RANDOMFORESTWITHGPS_RANDOMNUMBERS_H
