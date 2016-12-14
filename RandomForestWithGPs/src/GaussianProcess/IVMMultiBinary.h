/*
 * IVMMultiBinary.h
 *
 *  Created on: 17.11.2016
 *      Author: Max
 */

#ifndef GAUSSIANPROCESS_IVMMULTIBINARY_H_
#define GAUSSIANPROCESS_IVMMULTIBINARY_H_

#include "IVM.h"
#include "../Base/Predictor.h"
#include "../Base/ThreadMaster.h"
#include "../Data/OnlineStorage.h"

class IVMMultiBinary  : public PredictorMultiClass, public Observer {
public:
	IVMMultiBinary(OnlineStorage<ClassPoint*>& storage,
			const unsigned int numberOfInducingPointsPerIVM,
			const bool doEPUpdate);

	virtual ~IVMMultiBinary();

	void train();

	int predict(const DataPoint& point) const;

	int predict(const ClassPoint& point) const;

	void predict(const DataPoint& point, std::vector<double>& probabilities) const;

	void predictData(const Data& points, Labels& labels) const;

	void predictData(const Data& points, Labels& labels, std::vector< std::vector<double> >& probabilities) const;

	int amountOfClasses() const;

	void update(Subject* caller, unsigned int event);

private:

	void trainInParallel(const int usedIvm, const double trainTime, InformationPackage* package);

	void predictDataInParallel(const Data& points, const int usedIvm,
			std::vector< std::vector<double> >* probabilities, InformationPackage* package) const;

	void initInParallel(const int startOfKernel, const int endOfKernel, Eigen::MatrixXd* differenceMatrix);

	void retrainIvmIfNeeded(InformationPackage* package, const int iClassNr);

	OnlineStorage<ClassPoint*>& m_storage;

	std::vector<IVM*> m_ivms;

	std::vector<bool> m_isClassUsed;

	std::vector<unsigned int> m_classOfIVMs;

	int m_numberOfInducingPointsPerIVM;

	bool m_doEpUpdate;

	bool m_init;

	bool m_firstTraining;

	OnlineRandomForest* m_orfForKernel;

};

#endif /* GAUSSIANPROCESS_IVMMULTIBINARY_H_ */
