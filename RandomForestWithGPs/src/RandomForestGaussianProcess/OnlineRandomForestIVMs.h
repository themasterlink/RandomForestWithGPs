/*
 * OnlineRandomForestIVMs.h
 *
 *  Created on: 17.11.2016
 *      Author: Max
 */

#ifndef RANDOMFORESTGAUSSIANPROCESS_ONLINERANDOMFORESTIVMS_H_
#define RANDOMFORESTGAUSSIANPROCESS_ONLINERANDOMFORESTIVMS_H_

#include "../RandomForests/OnlineRandomForest.h"
#include "../GaussianProcess/IVMMultiBinary.h"
#include "../Base/Observer.h"

class OnlineRandomForestIVMs : public PredictorMultiClass, public Observer {
public:
	OnlineRandomForestIVMs(OnlineStorage<LabeledVectorX*>& storage, const int maxDepth, const int amountOfUsedClasses);
	virtual ~OnlineRandomForestIVMs();

	void update(Subject* caller, unsigned int event) override;

	unsigned int predict(const VectorX& point) const override;

	unsigned int predict(const LabeledVectorX& point) const;

	void predictData(const Data& points, Labels& labels) const override;

	void predictData(const LabeledData& points, Labels& labels) const;

	void predictData(const Data& points, Labels& labels, std::vector< std::vector<real> >& probabilities) const override;

	void predictData(const LabeledData& points, Labels& labels, std::vector< std::vector<real> >& probabilities) const;

	unsigned int amountOfClasses() const override;

private:

	void trainIvm(const int usedIvm, const int nrOfInducingPoints, const bool doEpUpdate, LabeledData* data, const int orfClass);

	void update();

	OnlineStorage<LabeledVectorX*>& m_storage;

	OnlineRandomForest m_orf;

	std::vector<IVMMultiBinary*> m_ivms;

	std::vector<OnlineStorage<LabeledVectorX*>* > m_onlineStoragesForIvms;

	int m_amountOfUsedClasses;

	bool m_firstTrainedDone;

};

#endif /* RANDOMFORESTGAUSSIANPROCESS_ONLINERANDOMFORESTIVMS_H_ */
