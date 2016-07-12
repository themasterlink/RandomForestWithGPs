/*
 * DataTests.h
 *
 *  Created on: 12.07.2016
 *      Author: Max
 */

#ifndef UNITTESTS_DATATESTS_H_
#define UNITTESTS_DATATESTS_H_

#include "../Data/DataConverter,h"

class Tests {

	static void testForUniformRandDataConverter(){
		Data dataTest;
		Labels labelTest;
		std::vector<int> countClass(4,0);
		Eigen::VectorXd p(2);
		p << 1,1;
		for(int i = 0; i < 30; ++i){
			dataTest.push_back(p);
			if(i % 2 == 0){
				labelTest.push_back(0);
				countClass[0]++;
			}else if(i % 3 == 0){
				labelTest.push_back(1);
				countClass[1]++;
			}else if(i % 5 == 0){
				labelTest.push_back(2);
				countClass[2]++;
			}else{
				labelTest.push_back(3);
				countClass[3]++;
			}
		}
		Eigen::MatrixXd result2;
		Eigen::VectorXd y2;
		DataConverter::toRandUniformDataMatrix(dataTest, labelTest, countClass, result2, y2, 12, 0);
		std::cout << "result2: \n" << result2 << std::endl;
		std::cout << "\n\ny2: \n" << y2 << std::endl;
	}

private:
	Tests(){};
	~Tests(){};

};

#endif /* UNITTESTS_DATATESTS_H_ */
