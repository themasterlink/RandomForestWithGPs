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
	Settings::instance().getValue("TotalStorage.amountOfPointsUsedForTraining", firstPoints);
	const Real share = Settings::instance().getDirectRealValue("TotalStorage.shareForTraining");
	firstPoints /= share;
	printOnScreen("Read " << firstPoints << " points per class");
	TotalStorage::instance().readData(firstPoints);
	printOnScreen("TotalStorage::instance().getSmallestClassSize(): " << TotalStorage::instance().getSmallestClassSize()
																	  << " with "
																	  << TotalStorage::instance().getAmountOfClass()
																	  << " classes");
	const auto trainAmount = (int) (share *
									(std::min((int) TotalStorage::instance().getSmallestClassSize(), firstPoints) *
									 (Real) TotalStorage::instance().getAmountOfClass()));
	return trainAmount;
}


#ifdef BUILD_OLD_CODE

#include "multiClassRFGPTest.h"
#include "binaryClassRFTest.h"
#include "binaryClassGPTest.h"
#include "multiClassGPTest.h"

#endif // BUILD_OLD_CODE

#include "binaryClassIVMTest.h"
#include "multiClassIVMTest.h"
#include "binaryClassORFTest.h"
#include "multiClassORFIVMTest.h"
#include "performanceMeasurement.h"

#endif /* TESTS_TESTS_H_ */
