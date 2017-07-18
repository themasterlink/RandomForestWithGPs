//
// Created by denn_ma on 7/18/17.
//

#ifndef RANDOMFORESTWITHGPS_ARGMAXANDMIN_H
#define RANDOMFORESTWITHGPS_ARGMAXANDMIN_H

#include "../../Utility/Util.h"
#include "../../RandomNumberGenerator/RandomUniformNr.h"

namespace SpeedTests {

	namespace ArgMax {

		template<class T>
		inline auto argMaxVersion1(const T& begin, const T& end){
			return std::distance(begin, std::max_element(begin, end));
		}

		template<class T>
		inline auto argMaxVersion2(const T& container){
			using internT = typename T::value_type;
			auto max = std::numeric_limits<internT>::lowest();
			unsigned int value = 0;
			unsigned int it = 0;
			for(const auto& ele: container){
				if(ele > max){
					max = ele;
					value = it;
				}
				++it;
			}
			return value;
		}

		void testSpeedForReal(){
			GeneratorType gen;
			RandomDistributionReal dist(0, 1);
			std::vector<Real> realVec;
			const unsigned int len = 100000;
			realVec.reserve(len);
			for(unsigned int i = 0; i < len; ++i){
				realVec.push_back(dist(gen));
			}
			const unsigned int trys = 100000;
			unsigned long first = 0;
			StopWatch sw;
			for(unsigned int i = 0; i < trys; ++i){
				first += argMaxVersion1(realVec.begin(), realVec.end());
			}
			std::cout << "First: " << sw.elapsedAsTimeFrame() << std::endl;
			std::cout << "Value: " << first / trys << std::endl;
			first = 0;
			StopWatch sw2;
			for(unsigned int i = 0; i < trys; ++i){
				first += argMaxVersion2(realVec);
			}
			std::cout << "Second: " << sw2.elapsedAsTimeFrame() << std::endl;
			std::cout << "Value: " << first / trys << std::endl;
			std::cout << "Real Value: " << realVec[first / trys] << std::endl;

		}

		void testSpeedForInt(){
			GeneratorType gen;
			RandomUniformNr dist(1, 2322, 1232);
			std::vector<int> realVec;
			const unsigned int len = 100000;
			realVec.reserve(len);
			for(unsigned int i = 0; i < len; ++i){
				realVec.push_back(dist());
			}
			const unsigned int trys = 100000;
			unsigned long first = 0;
			StopWatch sw;
			for(unsigned int i = 0; i < trys; ++i){
				first += argMaxVersion1(realVec.begin(), realVec.end());
			}
			std::cout << "First: " << sw.elapsedAsTimeFrame() << std::endl;
			std::cout << "Value: " << first / trys << std::endl;
			first = 0;
			StopWatch sw2;
			for(unsigned int i = 0; i < trys; ++i){
				first += argMaxVersion2(realVec);
			}
			std::cout << "Second: " << sw2.elapsedAsTimeFrame() << std::endl;
			std::cout << "Value: " << first / trys << std::endl;
			std::cout << "Int Value: " << realVec[first / trys] << std::endl;
		}

		void runAll(){
			testSpeedForInt();
			testSpeedForReal();
		}
	}
}


#endif //RANDOMFORESTWITHGPS_ARGMAXANDMIN_H
