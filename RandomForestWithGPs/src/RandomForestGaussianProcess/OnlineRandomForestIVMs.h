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
	OnlineRandomForestIVMs(OnlineStorage<ClassPoint*>& storage, const int maxDepth, const int amountOfUsedClasses);
	virtual ~OnlineRandomForestIVMs();

	void update(Subject* caller, unsigned int event);

	int predict(const DataPoint& point) const;

	void predictData(const Data& points, Labels& labels) const;

	void predictData(const Data& points, Labels& labels, std::vector< std::vector<double> >& probabilities) const;

	int amountOfClasses() const;

private:

	void update();

	OnlineStorage<ClassPoint*>& m_storage;

	OnlineRandomForest m_orf;

	int m_amountOfUsedClasses;

	bool m_firstTrainedDone;

};

#endif /* RANDOMFORESTGAUSSIANPROCESS_ONLINERANDOMFORESTIVMS_H_ */
