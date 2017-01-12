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
#include "../Base/InformationPackage.h"
#include "../Data/OnlineStorage.h"

class IVMMultiBinary  : public PredictorMultiClass, public Observer {
public:
	IVMMultiBinary(OnlineStorage<ClassPoint*>& storage,
			const unsigned int numberOfInducingPointsPerIVM,
			const bool doEPUpdate, const int orfClassLabel = UNDEF_CLASS_LABEL);

	virtual ~IVMMultiBinary();

	void train();

	unsigned int predict(const DataPoint& point) const;

	unsigned int predict(const ClassPoint& point) const;

	void predict(const DataPoint& point, std::vector<double>& probabilities) const;

	void predictData(const Data& points, Labels& labels) const;

	void predictData(const ClassData& points, Labels& labels) const;

	void predictData(const Data& points, Labels& labels, std::vector< std::vector<double> >& probabilities) const;

	void predictData(const ClassData& points, Labels& labels, std::vector< std::vector<double> >& probabilities) const;

	unsigned int amountOfClasses() const;

	void update(Subject* caller, unsigned int event);

private:

	void retrainAllIvmsIfNeeded(InformationPackage* wholePackage);

	void trainInParallel(IVM* ivm, const int usedIvm, InformationPackage* package);

	void predictDataInParallel(IVM* ivm, const Data& points, const int usedIvm,
			std::vector< std::vector<double> >* probabilities, InformationPackage* package) const;

	void predictClassDataInParallel(IVM* ivm, const ClassData& points, const int usedIvm,
			std::vector< std::vector<double> >* probabilities, InformationPackage* package) const;

	void initInParallel(const int startOfKernel, const int endOfKernel, Eigen::MatrixXd* differenceMatrix, InformationPackage* package);

	void retrainIvmIfNeeded(IVM* ivm, InformationPackage* package, const int iClassNr);

	unsigned int getLabelFrom(const std::vector<double>& probs) const;

	OnlineStorage<ClassPoint*>& m_storage;

	std::vector<IVM*> m_ivms;

	std::vector<bool> m_isClassUsed;

	std::vector<unsigned int> m_classOfIVMs;

	std::vector<unsigned int> m_generalClassesToIVMs; // converts from all classes to the ordering in m_ivms and m_isClassUsed

	int m_numberOfInducingPointsPerIVM;

	bool m_doEpUpdate;

	bool m_init;

	bool m_firstTraining;

	double m_correctAmountForTrainingData;

	OnlineRandomForest* m_orfForKernel;

	boost::thread_group m_group;

	std::list<InformationPackage*> m_packages;

	std::vector<double> m_correctAmountForTrainingDataForClasses;

	const int m_orfClassLabel;

	unsigned int m_amountOfAllClasses;

	mutable RandomUniformNr m_randClass;
};

#endif /* GAUSSIANPROCESS_IVMMULTIBINARY_H_ */
