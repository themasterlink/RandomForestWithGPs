/*
 * GaussianKernelRandomGenerator.h
 *
 *  Created on: 14.12.2016
 *      Author: Max
 */

#ifndef GAUSSIANPROCESS_KERNEL_GAUSSIANKERNELOPTIMIZER_H_
#define GAUSSIANPROCESS_KERNEL_GAUSSIANKERNELOPTIMIZER_H_

#include <boost/random.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include "../../Data/OnlineStorage.h"
#include "../../Utility/Util.h"
#include "GaussianKernel.h"

class GaussianKernelOptimizer {
public:
	using base_generator_type = GeneratorType; // generator type
	using uniform_distribution_int = boost::random::uniform_int_distribution<int>; // generator type
	using uniform_distribution_real = boost::uniform_real<Real>; // generator type

	GaussianKernelOptimizer(const int maxTriesPerSolution, const int amountOfUsedClasses, const std::vector<Vector2>& minAndMaxValues, const int seed);
	virtual ~GaussianKernelOptimizer();

	void init();

//	void getNextPoint(GaussianKernelParams& params, const int iClassNr){
//		m_mutex.lock();
//		//TODO find way of deterministiv handling of this:
//		int neighbourId = getRandomNrForClass(iClassNr);
//		while(neighbourId == iClassNr){ // find a random id which is not equal to the current iClassNr
//			neighbourId = getRandomNrForClass(iClassNr);
//		}
//		const Real randDist = getRandDist(iClassNr);
//		params.m_length.changeAmountOfDims(m_solutions[iClassNr].back().m_length.hasMoreThanOneDim());
//		for(unsigned int k = 0; k < GaussianKernelParams::paramsAmount; ++k){
//			if(m_solutions[iClassNr].back().m_params[k]->hasMoreThanOneDim()){
//				for(unsigned int ele = 0; ele < ClassKnowledge::amountOfDims(); ++ele){
//					params.m_params[k]->getValues()[ele] = m_solutions[iClassNr].back().m_params[k]->getValues()[ele] + randDist *
//							(m_solutions[neighbourId].back().m_params[k]->getValues()[ele] - m_solutions[iClassNr].back().m_params[k]->getValues()[ele]);
//				}
//			}else{
//				params.m_params[k]->getValues()[0] = m_solutions[iClassNr].back().m_params[k]->getValue() + randDist *
//						(m_solutions[neighbourId].back().m_params[k]->getValue() - m_solutions[iClassNr].back().m_params[k]->getValue());
//			}
//		}
//		// is the same as the above just without the need of calling the operators: params = m_solutions[iClassNr].back() +  * (m_solutions[neighbourId].back() - m_solutions[iClassNr].back());
//		m_mutex.unlock();
//	}
//
//	void addPoint(GaussianKernelParams params, const int iClassNr, const Real fitness){
//		if()
//		m_solutions[m_usedClasses[iClassNr]].push_back(params);
//	}

	// calc probability to improve own solution or use another solution if the value could be better

private:

	// returns a value between 0 ..< amountOfUsedClass - 1, param needed to specify correct generator (thread safe)
	unsigned int getRandomNrForClass(const int iClassNr);

	Real getRandDist(const int iClassNr);

	std::vector<Vector2 > m_minAndMaxValues; // should have the dimension of the used params amount

	uniform_distribution_int m_randomDistrubtionInts; // gives a value between 0 and the amount of used classes - 1

	uniform_distribution_real m_randomDistrubtionDist; // gives a value between -1 and 1

	std::vector<base_generator_type> m_generators; // each used class gets its own generator to ensure that each single thread per class gets its own random numbers

	using ParamList = std::list<GaussianKernelParams>;
	using SolutionVector = std::vector<ParamList>;

	SolutionVector m_solutions;

	std::vector<unsigned int> m_usedClasses; // converts an outside class nr to an internal used class nr (for example class 100 -> is mapped to 5)

	std::vector<unsigned int> m_trialCounter; // has the dimension of the amount of used classes

	boost::mutex m_mutex;
};

#endif /* GAUSSIANPROCESS_KERNEL_GAUSSIANKERNELOPTIMIZER_H_ */
