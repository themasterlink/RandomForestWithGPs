/*
 * tests.h
 *
 *  Created on: 15.09.2016
 *      Author: Max
 */

#ifndef TESTS_TESTS_H_
#define TESTS_TESTS_H_

#include "../Utility/Util.h"
#include "../Data/TotalStorage.h"
#include "../Base/Settings.h"

int readAllData(){
	int firstPoints; // all points
	Settings::getValue("TotalStorage.amountOfPointsUsedForTraining", firstPoints);
	const real share = Settings::getDirectRealValue("TotalStorage.shareForTraining");
	firstPoints /= share;
	printOnScreen("Read " << firstPoints << " points per class");
	TotalStorage::readData(firstPoints);
	printOnScreen("TotalStorage::getSmallestClassSize(): " << TotalStorage::getSmallestClassSize() << " with " << TotalStorage::getAmountOfClass() << " classes");
	const auto trainAmount = (int) (share * (std::min((int) TotalStorage::getSmallestClassSize(), firstPoints) * (real) TotalStorage::getAmountOfClass()));
	return trainAmount;
}

#include "multiClassRFGPTest.h"
#include "binaryClassRFTest.h"
#include "binaryClassGPTest.h"
#include "binaryClassIVMTest.h"
#include "multiClassIVMTest.h"
#include "binaryClassORFTest.h"
#include "multiClassGPTest.h"
#include "multiClassORFIVMTest.h"
#include "performanceMeasurement.h"

#endif /* TESTS_TESTS_H_ */
