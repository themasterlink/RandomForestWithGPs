/*
 * GaussianKernelOptimizer.cc
 *
 *  Created on: 14.12.2016
 *      Author: Max
 */

#ifdef BUILD_OLD_CODE

#include "GaussianKernelOptimizer.h"

//GaussianKernelOptimizer::GaussianKernelRandomGenerator(){
//	std::vector<Real> means = {Settings::instance().getDirectRealValue("KernelParam.lenMean"),
//			Settings::instance().getDirectRealValue("KernelParam.fNoiseMean"),
//			Settings::instance().getDirectRealValue("KernelParam.sNoiseMean")};
//	std::vector<Real> sds = {Settings::instance().getDirectRealValue("KernelParam.lenVar"),
//			Settings::instance().getDirectRealValue("KernelParam.fNoiseVar"),
//			Settings::instance().getDirectRealValue("KernelParam.sNoiseVar")};
//	setMeansAndVars(means, sds);
//}

GaussianKernelOptimizer::~GaussianKernelOptimizer(){
}

//void GaussianKernelOptimizer::addPointWithMeansAndVars(const std::vector<Real>& means, const std::vector<Real>& vars){
//	if(means.size() == vars.size() && means.size() == GaussianKernelParams::paramsAmount){
//		m_means.m_length.setAllValuesTo(means[0]);
//		m_means.m_fNoise.setAllValuesTo(means[1]);
//		m_means.m_sNoise.setAllValuesTo(means[2]);
//		m_vars.m_length.setAllValuesTo(vars[0]);
//		m_vars.m_fNoise.setAllValuesTo(vars[1]);
//		m_vars.m_sNoise.setAllValuesTo(vars[2]);
//	}else if(means.size() == vars.size() && means.size() == GaussianKernelParams::paramsAmount - 1 + ClassKnowledge::amountOfDims()){
//		m_means.m_length.changeAmountOfDims(true);
//		for(unsigned int i = 0; i < ClassKnowledge::amountOfDims(); ++i){
//			m_means.m_length.getValues()[i] = means[i];
//			m_vars.m_length.getValues()[i] = vars[i];
//		}
//		m_means.m_fNoise.setAllValuesTo(means[vars.size() - 2]);
//		m_means.m_sNoise.setAllValuesTo(means[vars.size() - 1]);
//		m_vars.m_fNoise.setAllValuesTo(vars[vars.size() - 2]);
//		m_vars.m_sNoise.setAllValuesTo(vars[vars.size() - 1]);
//	}else{
//		printError("This means and vars are incorrect!");
//	}
//}

//void GaussianKernelOptimizer::randParams(GaussianKernelParams& params){
//
//}


#endif // BUILD_OLD_CODE