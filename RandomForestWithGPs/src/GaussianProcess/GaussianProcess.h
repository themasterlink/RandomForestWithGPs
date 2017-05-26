/*
 * GaussianProcessBinaryClass.h
 *
 *  Created on: 15.06.2016
 *      Author: Max
 */

#ifndef GAUSSIANPROCESS_GAUSSIANPROCESS_H_
#define GAUSSIANPROCESS_GAUSSIANPROCESS_H_

#include <Eigen/Cholesky>
#include "../Utility/Util.h"
#include "../Data/LabeledVectorX.h"
#include <cmath>
#include "Kernel/GaussianKernel.h"
#include "../Base/Predictor.h"

class GaussianProcessWriter;
class BayesOptimizer;

class GaussianProcess : public PredictorBinaryClass {
	friend GaussianProcessWriter;
	friend BayesOptimizer;
public:

	enum class Status {
		NANORINFERROR = -1,
		ALLFINE = 0
	};

	GaussianProcess();
	virtual ~GaussianProcess();

	void init(const Matrix& dataMat, const VectorX& y);

	void train();

	void trainWithoutKernelOptimize();

	Real predict(const VectorX& newPoint, const int sampleSize) const;

	Real predict(const VectorX& point) const;

	void setKernelParams(const Real len, const Real fNoise, const Real sNoise);

	void setKernelParams(const std::vector<Real>& lens, const Real fNoise, const Real sNoise);

	GaussianKernel& getKernel(){ return m_kernel; };

	const StopWatch& getTrainFWatch(){return m_sw;};

	void resetFastPredict(){ m_fastPredict = false; };

	GaussianProcess::Status trainBayOpt(Real& logZ, const Real lambda); // TODO should be private again
private:

	Status trainLM(Real& logZ, std::vector<Real>& dLogZ);

	Status trainF(const Matrix& K);

	Status train(const int dataPoints, const Matrix& dataMat, const VectorX& y);

	void updatePis();

	Real m_repetitionStepFactor;

	Matrix m_dataMat;
	VectorX m_a;
	VectorX m_y;
	VectorX m_t;
	VectorX m_f;
	VectorX m_pi;
	VectorX m_dLogPi;
	VectorX m_ddLogPi;
	VectorX m_sqrtDDLogPi;
	Matrix m_innerOfLLT;
	Eigen::LLT<Matrix> m_choleskyLLT;
	int m_dataPoints; // amount of data points
	StopWatch m_sw;

	bool m_init;
	bool m_trained;
	mutable bool m_fastPredict;
	mutable Real m_fastPredictVFStar;
	GaussianKernel m_kernel;


};

#endif /* GAUSSIANPROCESS_GAUSSIANPROCESS_H_ */
