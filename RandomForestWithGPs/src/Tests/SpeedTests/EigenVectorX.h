//
// Created by denn_ma on 7/18/17.
//

#ifndef RANDOMFORESTWITHGPS_EIGENVECTORX_H
#define RANDOMFORESTWITHGPS_EIGENVECTORX_H

#include "../../Utility/Util.h"
#include "../../RandomNumberGenerator/RandomUniformNr.h"

namespace SpeedTests {

	namespace EigenVsVector {

		Real testAccess(const Real* vec, const unsigned int len, const unsigned int tries){
			Real res = 0.0;
			for(unsigned int i = 0; i < tries; ++i){
				for(unsigned int j = 0; j < len; ++j){
					res += vec[j];
				}
			}
			return res;
		}

		Real testAccess(const VectorX& vec, const unsigned int tries){
			Real res = 0.0;
			for(unsigned int i = 0; i < tries; ++i){
				for(unsigned int j = 0, end = (unsigned int) vec.rows(); j < end; ++j){
					res += vec.coeff(j);
				}
			}
			return res;
		}

		Real testAccess(const std::vector<Real>& vec, const unsigned int tries){
			Real res = 0.0;
			for(unsigned int i = 0; i < tries; ++i){
				for(unsigned int j = 0, end = (unsigned int) vec.size(); j < end; ++j){
					res += vec[j];
				}
			}
			return res;
		}

		Real testIterAccess(const std::vector<Real>& vec, const unsigned int tries){
			Real res = 0.0;
			for(unsigned int i = 0; i < tries; ++i){
				for(const auto val : vec){
					res += val;
				}
			}
			return res;
		}

		void testEigenVsStdVector(){
			const unsigned int vecSize = 2048;
			Real* array = new Real[2048];
			VectorX eigenVec(vecSize);
			std::vector<Real> stdVec(vecSize);
			GeneratorType gen;
			RandomDistributionReal dist;
			for(unsigned int i = 0; i < 2048; ++i){
				eigenVec.coeffRef(i) = dist(gen);
				stdVec[i] = eigenVec.coeff(i);
				array[i] = stdVec[i];
			}

			const unsigned int tries = 3000000;
			Real val = 0.0;
			StopWatch sw;
			val = testAccess(eigenVec, tries);
			std::cout << "Eigen: " << sw.elapsedAsTimeFrame() << ", val" << val << std::endl;
			StopWatch sw2;
			val = testAccess(stdVec, tries);
			std::cout << "Std: " << sw2.elapsedAsTimeFrame() << ", val" << val << std::endl;
			StopWatch sw3;
			val = testIterAccess(stdVec, tries);
			std::cout << "Std: " << sw3.elapsedAsTimeFrame() << ", val" << val << std::endl;
			StopWatch sw4;
			val = testAccess(array, vecSize, tries);
			std::cout << "Eigen: " << sw4.elapsedAsTimeFrame() << ", val" << val << std::endl;


		};

		void runAll(){
			testEigenVsStdVector();
		}

	}

}


#endif //RANDOMFORESTWITHGPS_EIGENVECTORX_H
