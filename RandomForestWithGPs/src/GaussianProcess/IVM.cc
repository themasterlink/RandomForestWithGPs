/*
 * IVM.cc
 *
 *  Created on: 27.09.2016
 *      Author: Max
 */

#include "IVM.h"
#include <boost/math/special_functions/erf.hpp>
#include "../Base/CommandSettings.h"
#include "../Base/Settings.h"
#include "../Data/DataWriterForVisu.h"
#include "../Utility/Util.h"
#include "../Data/DataConverter.h"
#include "../Data/DataBinaryWriter.h"

boost::mutex IVM::m_listMutex;
#define LOG2   0.69314718055994528623
#define LOG2PI 1.8378770664093453391
#define SQRT2  1.4142135623730951455

IVM::IVM(OnlineStorage<ClassPoint*>& storage, const bool isPartOfMultiIvm):
	m_logZ(0), m_derivLogZ(),
	m_storage(storage),m_dataPoints(0),
	m_numberOfInducingPoints(0), m_bias(0), m_lambda(0),
	m_doEPUpdate(false), m_splitOfClassOneInData(0.5), m_desiredPoint(0.5), m_desiredMargin(0.05),
	m_calcLogZ(false), m_calcDerivLogZ(false), m_trained(false),
	m_gaussKernel(nullptr),
	m_rfKernel(nullptr),
	m_uniformNr(0, 10, 0),
	m_useNeighbourComparison(false),
	m_package(nullptr),
	m_isPartOfMultiIvm(isPartOfMultiIvm),
	m_className("undefined"),
	m_arFunvals(nullptr),
	m_hyperParamsValues(nullptr){
	int kernelType = 0;
	Settings::getValue("IVM.kernelType", kernelType);
	if(kernelType == 0){
		m_kernelType = GAUSS;
		bool hasLengthMoreThanParam;
		Settings::getValue("IVM.hasLengthMoreThanParam", hasLengthMoreThanParam);
		m_gaussKernel = new GaussianKernel();
		m_gaussKernel->changeKernelConfig(hasLengthMoreThanParam);
		m_gaussKernel->newRandHyperParams();
	}else if(kernelType == 1){
		m_kernelType = RF;
		int samplingAmount, maxDepth;
		Settings::getValue("RandomForestKernel.samplingAmount", samplingAmount);
		Settings::getValue("RandomForestKernel.maxDepth", maxDepth);
		const bool createOrf = !isPartOfMultiIvm;
		m_rfKernel = new RandomForestKernel(m_storage, maxDepth, samplingAmount, ClassKnowledge::amountOfClasses(), createOrf);
	}else{
		printError("This kernel type is not supported here!");
	}
}

IVM::~IVM(){
	if(Settings::getDirectBoolValue("IVM.useCmaes")){
		cmaes::cmaes_exit(&m_evo); /* release memory */
		cmaes::cmaes_boundary_transformation_exit(&m_cmaesBoundaries); /* release memory */
		free(m_hyperParamsValues);
	}
}

void IVM::setDerivAndLogZFlag(const bool doLogZ, const bool doDerivLogZ){
	m_calcDerivLogZ = doDerivLogZ;
	m_calcLogZ = doLogZ;
}

void IVM::setOnlineRandomForest(OnlineRandomForest* forest){
	if(m_rfKernel != nullptr){
		m_rfKernel->setOnlineRandomForest(forest);
	}else{
		printError("The type is not the right one!");
	}
}

void IVM::init(const unsigned int numberOfInducingPoints,
		const Eigen::Vector2i& labelsForClasses,
		const bool doEPUpdate, const bool calcDifferenceMatrixAlone){
	if(m_storage.size() == 0u){
		printError("No data in init given!"); return;
	}
	m_labelsForClasses = labelsForClasses;
	m_className = ClassKnowledge::getNameFor(getLabelForOne());
	m_uniformNr.setSeed((int)labelsForClasses.coeff(0) * 74739); // TODO better way if class is used multiple times
	m_uniformNr.setMinAndMax(1, m_storage.size() / 100);
	const bool oneVsAllCase = m_labelsForClasses.coeff(1) == UNDEF_CLASS_LABEL;
	if(m_labelsForClasses.coeff(0) == m_labelsForClasses.coeff(1)){
		printError("The labels for the two different classes are the same!");
	}
	m_dataPoints = m_storage.size();
	m_y = Vector(m_storage.size());
	int amountOfOneClass = 0;
	for(unsigned int i = 0; i < m_y.rows(); ++i){ // convert usuall mutli class labels in 1 and -1
		if(m_storage[i]->getLabel() == m_labelsForClasses.coeff(0)){
			m_y.coeffRef(i) = 1;
			++amountOfOneClass;
		}else if(((int) m_storage[i]->getLabel() == m_labelsForClasses.coeff(1)) || oneVsAllCase){
			m_y.coeffRef(i) = -1;
		}else{
			printError("This IVM contains data, which does not belong to one of the two classes!");
			return;
		}
	}
	m_doEPUpdate = doEPUpdate;
	setNumberOfInducingPoints(numberOfInducingPoints);
	//	StopWatch sw;
	if(m_kernelType == GAUSS){
		if(calcDifferenceMatrixAlone){
			const bool calcDifferenceMatrix = !m_gaussKernel->hasLengthMoreThanOneDim();
			m_gaussKernel->init(m_storage.storage(), calcDifferenceMatrix, false);
		}else{
			// in this case just init the connectin between the kernel and the data, but no calculation is performed!
			m_gaussKernel->init(m_storage.storage(), false, false);
		}
	}else if(m_kernelType == RF){
		if(m_rfKernel != nullptr){
			m_rfKernel->init();
			// to train the tree!
			if(!m_isPartOfMultiIvm){
				m_rfKernel->update(&m_storage, OnlineStorage<ClassPoint*>::APPENDBLOCK);
			}
		}
	}else{
		printError("This kernel type is not supported");
	}
//	printInPackageOnScreen(m_package, "Time: " << sw.elapsedAsPrettyTime());
	//printInPackageOnScreen(m_package, "Frac: " << (double) amountOfOneClass / (double) m_dataPoints);

	m_bias = boost::math::cdf(boost::math::complement(m_logisticNormal, (double) amountOfOneClass / (double) m_dataPoints));
	Settings::getValue("IVM.lambda", m_lambda);
	Settings::getValue("IVM.desiredMargin", m_desiredMargin);
	m_splitOfClassOneInData = (double) amountOfOneClass / (double) m_dataPoints;
	m_desiredPoint = (0.5 + m_splitOfClassOneInData) * 0.5; // middle between 0.5 and real margin
	Settings::getValue("IVM.useNeighbourComparison", m_useNeighbourComparison);

	// sampling method:
	if(Settings::getDirectBoolValue("IVM.useCmaes") && m_kernelType == GAUSS){

		double lowerBounds[] = {0.2,5.1, 0.03};
		lowerBounds[0] = Settings::getDirectDoubleValue("CMAES.lowerBoundLen");
		lowerBounds[1] = Settings::getDirectDoubleValue("CMAES.lowerBoundFNoise");
		lowerBounds[2] = Settings::getDirectDoubleValue("CMAES.lowerBoundSNoise");
		double upperBounds[] = {9., 24.0, 0.45};
		upperBounds[0] = Settings::getDirectDoubleValue("CMAES.upperBoundLen");
		upperBounds[1] = Settings::getDirectDoubleValue("CMAES.upperBoundFNoise");
		upperBounds[2] = Settings::getDirectDoubleValue("CMAES.upperBoundSNoise");
		const unsigned long nb_bounds = 3; /* numbers used from lower and upperBounds */
		unsigned long dimension;

		/* initialize boundaries, be sure that initialSigma is smaller than upper minus lower bound */
		cmaes::cmaes_boundary_transformation_init(&m_cmaesBoundaries, lowerBounds, upperBounds, nb_bounds);
		/* Initialize everything into the struct evo, 0 means default */
		const int seed = m_gaussKernel->getSeed() + 812737 * (1 + getLabelForOne());
		m_arFunvals = cmaes::cmaes_init(&m_evo, 0, NULL, NULL, seed, 0, "../Settings/cmaes_initials.par");
		dimension = (unsigned long) cmaes::cmaes_Get(&m_evo, "dimension");
		if(dimension != nb_bounds){
			printError("The dimension in the settings does not fit!");
		}
		m_hyperParamsValues = cmaes::cmaes_NewDouble(dimension); /* calloc another vector */
		// no values are here -> so read is uncessary
//		cmaes::cmaes_ReadSignals(&m_evo, "../Settings/cmaes_signals.par");  /* write header and initial values */
	}
}

void IVM::setNumberOfInducingPoints(unsigned int nr){
	m_numberOfInducingPoints = std::min(nr, m_dataPoints);
	m_nuTilde = Vector(m_numberOfInducingPoints);
	m_tauTilde = Vector(m_numberOfInducingPoints);
	m_eye = Matrix::Identity(m_numberOfInducingPoints, m_numberOfInducingPoints);
}

double IVM::cumulativeDerivLog(const double x){
	return -(LOG2PI + x * x) * 0.5;
}

double IVM::cumulativeLog(const double x){
	return boost::math::erfc(-x / SQRT2) - LOG2;
}

bool IVM::train(const bool doSampling, const int verboseLevel, const bool useKernelValuesAsBestParams){
//	if(m_kernel.calcDiagElement(0) == 0){
//		if(verboseLevel != 0)
//			printError("The kernel diagonal is 0, this kernel params are invalid:" << m_kernel.prettyString());
//		return false;
//	}
//	if(!m_kernel.isInit() || isnan(m_kernel.kernelFunc(0,0)) || m_kernel.calcDiagElement(0) != m_kernel.kernelFunc(0,0)){
//		if(verboseLevel != 0)
//			printError("The kernel was not initalized!");
//		return false;
//	}
	m_trained = false;

	const double minCorrectForWholeClassification = 70;
	if(m_package == nullptr){
		printError("The IVM has no set package set!");
		return false;
	}
	m_package->wait();
	if(m_numberOfInducingPoints <= 0){
		if(verboseLevel != 0)
			printError("The number of inducing points is equal or below zero: " << m_numberOfInducingPoints);
		return false;
	}
	if(m_kernelType == GAUSS){
		GaussianKernelParams bestParams;
		std::string folderLocation;
		if(CommandSettings::get_useFakeData()){
			Settings::getValue("TotalStorage.folderLocFake", folderLocation);
		}else{
			Settings::getValue("TotalStorage.folderLocReal", folderLocation);
		}
		const std::string kernelFilePath = folderLocation + "bestKernelParamsForClass" + number2String((int)m_labelsForClasses.coeff(0)) + ".binary";
		bool loadBestParams = false;
		double bestLogZ = NEG_DBL_MAX;
		double bestCorrectness = 0;
		if(boost::filesystem::exists(kernelFilePath) && Settings::getDirectBoolValue("IVM.Training.useSavedHyperParams")){
			bestParams.readFromFile(kernelFilePath);
			loadBestParams = true;
		}else if(useKernelValuesAsBestParams){
			bestParams = m_gaussKernel->getHyperParams();
			if(m_logZ == 0){
				printError("The logZ was not set correctly!");
			}
			bestLogZ = m_logZ; // should be set correctly on the last training with this params
			if(Settings::getDirectBoolValue("IVM.useCmaes")){
				std::stringstream str2;
				str2 << "Best: " << number2String(bestParams.m_length.getValue(),3) << ", "
						<< number2String(bestParams.m_fNoise.getValue(),4) << ", "
						<< number2String(bestParams.m_sNoise.getValue(),3) << ", "
						<< "e: " << number2String(100 - m_package->correctlyClassified(), 3) << " %%";
				m_package->setAdditionalInfo(str2.str());
				bestCorrectness = m_package->correctlyClassified();
			}else if(m_package->correctlyClassified() > minCorrectForWholeClassification){
				bestCorrectness = m_package->correctlyClassified();
			}
		}else{
			bestParams.m_length.setAllValuesTo(Settings::getDirectDoubleValue("KernelParam.len"));
			bestParams.m_fNoise.setAllValuesTo(Settings::getDirectDoubleValue("KernelParam.fNoise"));
			bestParams.m_sNoise.setAllValuesTo(Settings::getDirectDoubleValue("KernelParam.sNoise"));
		}
		if(doSampling){
			printInPackageOnScreen(m_package, "Start sampling");
			bool hasMoreThanOneLengthValue = Settings::getDirectBoolValue("IVM.hasLengthMoreThanParam");
			m_gaussKernel->changeKernelConfig(hasMoreThanOneLengthValue);
			std::vector<double> means = {Settings::getDirectDoubleValue("KernelParam.lenMean"),
					Settings::getDirectDoubleValue("KernelParam.fNoiseMean"),
					Settings::getDirectDoubleValue("KernelParam.sNoiseMean")};
			std::vector<double> sds = {Settings::getDirectDoubleValue("KernelParam.lenVar"),
					Settings::getDirectDoubleValue("KernelParam.fNoiseVar"),
					Settings::getDirectDoubleValue("KernelParam.sNoiseVar")};
			m_gaussKernel->setGaussianRandomVariables(means, sds);
			setDerivAndLogZFlag(true, false);
			StopWatch sw;
			StopWatch swAvg;
			if(!loadBestParams){
				m_uniformNr.setMinAndMax(0, m_dataPoints - 1);
				List<unsigned int> testPoints;
				int counter = 0;
				const int minAmountOfDataPoints = (std::min(m_dataPoints * m_splitOfClassOneInData, m_dataPoints * (1. - m_splitOfClassOneInData)) - 1) * 2;
				const int maxAmount = std::min((int) 100, minAmountOfDataPoints);
				while(counter < maxAmount){
					int value = -1;
					while(value == -1){
						int act = m_uniformNr();
						const unsigned int label = m_storage[act]->getLabel();
						if(counter % 2 == 0){ // check that always a different label is used for the test set
							if(label == getLabelForOne()){
								value = act;
							}
						}else{
							if(label != getLabelForOne()){
								value = act;
							}
						}
					}
					// check if not used until now
					bool alreadUsed = false;
					for(List<unsigned int>::const_iterator it = testPoints.begin(); it != testPoints.end(); ++it){
						if((int) *it == value){
							alreadUsed = true;
							break;
						}
					}
					if(!alreadUsed){
						++counter;
						testPoints.push_back(value);
					}
				}
				m_uniformNr.setMinAndMax(1, m_dataPoints / 100);
//				int iCounterSampling = 0;
				if(Settings::getDirectBoolValue("IVM.useCmaes")){
					/* Iterate until stop criterion holds */
					StopWatch sw;
					int iCounter = 0;
					double negBestLogZ = -bestLogZ;
					List<Eigen::Vector2d> points;
					List<double> values;
					double *const*pop;

					const int sampleLambda = cmaes::cmaes_Get(&m_evo, "lambda");
					const unsigned long dimension = (unsigned long) cmaes::cmaes_Get(&m_evo, "dimension");
					const int desiredAmountOfInducingsPoints = m_numberOfInducingPoints;

					while(m_package != nullptr){
						/* generate lambda new search points, sample population */

						pop = cmaes::cmaes_SamplePopulation(&m_evo); /* do not change content of pop */

						/* transform into bounds and evaluate the new search points */
						for(int i = 0; i < sampleLambda; ++i) {
//							const double corr = m_package->correctlyClassified();
//							const double probDiff = corr < 60. ? 0. : corr < 80 ? 0.1 : corr < 90 ? 0.2 : 0.3;
							cmaes::cmaes_boundary_transformation(&m_cmaesBoundaries, pop[i], m_hyperParamsValues, dimension);
							/* this loop can be omitted if is_feasible is invariably true */
//							while(!is_feasible(x_in_bounds, dimension)) { /* is_feasible needs to be user-defined, in case, and can change/repair x */
//								cmaes_ReSampleSingle(&evo, i);
//								cmaes_boundary_transformation(&boundaries, pop[i], x_in_bounds, dimension);
//							}
							m_gaussKernel->setHyperParams(m_hyperParamsValues[0], m_hyperParamsValues[1]); //, m_hyperParamsValues[2]);

							sw.startTime();
							const int diffInInducingPoints = desiredAmountOfInducingsPoints - m_numberOfInducingPoints;
							if(diffInInducingPoints > 0){
								setNumberOfInducingPoints(desiredAmountOfInducingsPoints);
							}
							const bool trained = internalTrain(true, 1);
							double error = NEG_DBL_MAX;
							if(trained){

								double oneError, minusOneError;
//								int amountOfOnesCorrect = 0, amountOfMinusOnesCorrect = 0;
//								int amountOfOneChecks = 0, amountOfMinusOneChecks = 0;
//								double correctness;
//								const bool onlyUseOnes = false;
//
//								testOnTrainingsData(amountOfOneChecks, amountOfOnesCorrect,
//										amountOfMinusOneChecks, amountOfMinusOnesCorrect,
//										correctness, probDiff, onlyUseOnes, false, testPoints); // false -> only testPoints
								error = calcErrorOnTrainingsData(false, testPoints, oneError, minusOneError);
								if(oneError > minusOneError){
									error = 0.25 * minusOneError + 0.75 * oneError;
								}else{
									error = 0.25 * oneError + 0.75 * minusOneError;
								}
//								double neededValue = 50.0;
								if(error <= 48.){
//									arFunvals[i] = - m_logZ / (double) m_numberOfInducingPoints + (-correctness + 100) * 2;
									const int diff = desiredAmountOfInducingsPoints - m_numberOfInducingPoints; // bad if not all inducing points were used
									m_arFunvals[i] = - m_logZ / (double) m_numberOfInducingPoints + error + diff / (double) desiredAmountOfInducingsPoints;
									if(m_arFunvals[i] < negBestLogZ * 1.2){
										error = calcErrorOnTrainingsData(true, testPoints, oneError, minusOneError);
										if(oneError > minusOneError){
											error = 0.25 * minusOneError + 0.75 * oneError;
										}else{
											error = 0.25 * oneError + 0.75 * minusOneError;
										}
										printInPackageOnScreen(m_package, "New full error of: " << error);
//										testOnTrainingsData(amountOfOneChecks, amountOfOnesCorrect,
//												amountOfMinusOneChecks, amountOfMinusOnesCorrect,
//												correctness, probDiff, onlyUseOnes, true, testPoints);
//										printInPackageOnScreen(m_package, "New full correcntess of: " << correctness);
//										arFunvals[i] = - m_logZ / (double) m_numberOfInducingPoints + (-correctness + 100) * 2; //fitfun(x_in_bounds, dimension); /* evaluate */
//										if(arFunvals[i] < negBestLogZ * 1.2 && correctness >= m_package->correctlyClassified()){
//											const double newProbDiff = correctness < 60. ? 0. : correctness < 80 ? 0.1 : correctness < 90 ? 0.2 : 0.3;
//											if(newProbDiff > probDiff){ // avoids overshooting in the beginning
//												// test again
//												testOnTrainingsData(amountOfOneChecks, amountOfOnesCorrect,
//														amountOfMinusOneChecks, amountOfMinusOnesCorrect,
//														correctness, newProbDiff, onlyUseOnes, true, testPoints);
//												printInPackageOnScreen(m_package, "New full on prob: " << newProbDiff << " correcntess of: " << correctness);
//											}

										// still better than take it
										const double correctness = 100. - error;
										if(correctness >= m_package->correctlyClassified()){
											m_gaussKernel->getCopyOfParams(bestParams);
											negBestLogZ = m_arFunvals[i];
											m_package->changeCorrectlyClassified(correctness);
											if(!m_gaussKernel->hasLengthMoreThanOneDim()){
												std::stringstream str2;
												str2 << "Best: " << number2String(bestParams.m_length.getValue(),3) << ", "
														<< number2String(bestParams.m_fNoise.getValue(),4) << ", "
														<< number2String(bestParams.m_sNoise.getValue(),3) << ", "
														<< "e: " << number2String(error, 3) << " %%, 1: "<< number2String(oneError, 3)
														<< " %%, -1: " << number2String(minusOneError, 3) << " %% logZ: " << negBestLogZ;
												//															<< "mc: " << number2String((amountOfMinusOnesCorrect / (double) amountOfMinusOneChecks) * 100., 2) << " %%, pc: "
												//															<<  number2String((amountOfOnesCorrect / (double) amountOfOneChecks) * 100, 2) << " %%, logZ: " << negBestLogZ << ", " << std::max(newProbDiff, probDiff);
												m_package->setAdditionalInfo(str2.str());
											}
											std::stringstream str;
											str << "New best params: " << bestParams << ", with correctness of: " << correctness;/*
																		<< " %%, ones: " << (amountOfOnesCorrect / (double) amountOfOneChecks) * 100.
																		<< " %%, minus ones: " << (amountOfMinusOnesCorrect / (double) amountOfMinusOneChecks) * 100.
																		<< ", amount of minues correct: " << amountOfMinusOnesCorrect << ", amount of minus ones: " << amountOfMinusOneChecks
																		<< " %%, for: " << m_dataPoints << " points";*/
											m_package->printLineToScreenForThisThread(str.str());
											//											}
											if(correctness > 95.){
												m_package->abortTraing();
												i = sampleLambda;
											}
										}
									}

								}else{
									m_arFunvals[i] = m_numberOfInducingPoints * 200;
								}
								values.push_back(m_arFunvals[i]);
							}else{
								m_arFunvals[i] = m_numberOfInducingPoints * 200;
								values.push_back(NEG_DBL_MAX);
							}
							points.push_back(Eigen::Vector2d(m_hyperParamsValues[0], m_hyperParamsValues[1]));
//							values.push_back(m_logZ);
							if(error > NEG_DBL_MAX){
							printInPackageOnScreen(m_package, "Test len: " << m_hyperParamsValues[0] << ", fNoise: " << m_hyperParamsValues[1] << ", has: " << -m_logZ / (double) m_numberOfInducingPoints
									<< ", logZ: " << -m_logZ << ", inducing: " << m_numberOfInducingPoints << " needed time: " << sw.elapsedAsTimeFrame() << ", error: " << error << " , c: " << iCounter);
							}else{
								printInPackageOnScreen(m_package, "Failed len: " << m_hyperParamsValues[0] << ", fNoise: " << m_hyperParamsValues[1]
										<< " needed time: " << sw.elapsedAsTimeFrame() << " , c: " << iCounter);
							}
							m_package->performedOneTrainingStep(); // adds a one to the counter
							if(m_package->shouldTrainingBeAborted()){
								break;
							}else if(m_package->shouldTrainingBePaused()){
								m_package->printLineToScreenForThisThread("Training has to wait!");
								m_package->wait(); // will hold this process
							}else if(m_package->correctlyClassified() > 99.0 && iCounter > cmaes::cmaes_Get(&m_evo, "lambda") * 2.0){
								m_package->abortTraing();
							}
							++iCounter;
						}
						if(m_package->shouldTrainingBeAborted()){
							m_package->printLineToScreenForThisThread("Training should be aborted!");
							break;
						}

						/* update the search distribution used for cmaes_SampleDistribution() */
						cmaes::cmaes_UpdateDistribution(&m_evo, m_arFunvals);  /* assumes that pop[i] has not been modified */

						/* read instructions for printing output or changing termination conditions */
//						cmaes_ReadSignals(&evo, "cmaes_signals.par");
					}
					/* get best estimator for the optimum, xmean */
//					cmaes::cmaes_boundary_transformation(&boundaries,
//						(double const *) cmaes_GetPtr(&m_evo, "xbestever"), /* "xbestever" might be used as well */
//						x_in_bounds, dimension);
					printOnScreen("Best params: " << bestParams);
//					bestParams.m_length.setAllValuesTo(x_in_bounds[0]);
//					bestParams.m_fNoise.setAllValuesTo(x_in_bounds[1]);
//					bestParams.m_sNoise.setAllValuesTo(m_gaussKernel->getHyperParams().m_sNoise.getValue());
					bestLogZ = -negBestLogZ;
					  /* and finally release memory */
					if((CommandSettings::get_visuRes() > 0. || CommandSettings::get_visuResSimple() > 0.) && Settings::getDirectBoolValue("VisuParams.visuHyperParamSampling2D")){
						DataWriterForVisu::writePointsIn2D("hp_params_"+m_className+".svg", points, values);
						openFileInViewer("hp_params_"+m_className+".svg");
					}
				}else{
				while(m_package != nullptr){ // equals a true
					//				if(m_uniformNr() % 10 == 0){
					//					printError("Just an test error!" << m_uniformNr() % 2);
					//				}
					m_gaussKernel->newRandHyperParams();
					std::stringstream str;
					str << "Try params: " << m_gaussKernel->getHyperParams();
					m_package->printLineToScreenForThisThread(str.str());
					const bool trained = internalTrain(true, 1);
					std::stringstream str2;
					if(trained){
						str2 << "Params: " << m_gaussKernel->getHyperParams() << " with success and logZ: " << m_logZ;
						m_package->overwriteLastLineToScreenForThisThread(str2.str());
					}else{
						str2 << "Params: " << m_gaussKernel->getHyperParams() << " failed";
						m_package->printLineToScreenForThisThread(str2.str());
					}
					//				if(!trained){
					//					printDebug("Hyperparams which not work: " << m_kernel.prettyString());
					//				}
					if(trained && bestLogZ < m_logZ * 0.98){ // even if the logZ is slightly above the value can be good enough -> perform simple check
						m_package->printLineToScreenForThisThread("Perform a simple test");
						// perform a simple test
						// go over a bunch of points to test it
						int amountOfOnesCorrect = 0, amountOfMinusOnesCorrect = 0;
						int amountOfOneChecks = 0, amountOfMinusOneChecks = 0;
						double correctness;
						double probDiff = 0;
						const bool onlyUseOnes = false;
						testOnTrainingsData(amountOfOneChecks, amountOfOnesCorrect,
								amountOfMinusOneChecks, amountOfMinusOnesCorrect,
								correctness, probDiff, onlyUseOnes, false, testPoints); // false -> only testPoints

						bool didCompleteCheck = false;
						if(correctness > minCorrectForWholeClassification){
							m_package->overwriteLastLineToScreenForThisThread("Perform a complex test");
							didCompleteCheck = true;
							// check all points
							testOnTrainingsData(amountOfOneChecks, amountOfOnesCorrect,
									amountOfMinusOneChecks, amountOfMinusOnesCorrect,
									correctness, probDiff, onlyUseOnes, true, testPoints); // false -> only testPoints
							if(correctness > 95){
								m_package->abortTraing();
							}
						}
						if(didCompleteCheck){
							// even for the first best correctness case is this valid, because here the simple correctness was above the threshold the first time
							if(correctness > bestCorrectness){
								// only take the params if the full check on the
								// trainingsdata provided a better value than the last full check
								// setting the means to the new bestParams
								if(!m_gaussKernel->hasLengthMoreThanOneDim()){
									std::vector<double> newMeans(3);
									for(unsigned int i = 0; i < 3; ++i){
										newMeans[i] = m_gaussKernel->getHyperParams().m_params[i]->getValue();
									}
									m_gaussKernel->setGaussianRandomVariables(newMeans, sds);
								}
								bestCorrectness = correctness;
								m_gaussKernel->getCopyOfParams(bestParams);
								bestLogZ = m_logZ;
								m_package->changeCorrectlyClassified(correctness);
								if(!m_gaussKernel->hasLengthMoreThanOneDim()){
									std::stringstream str2;
									str2 << "Best: " << number2String(bestParams.m_length.getValue(),3) << ", "
											<< number2String(bestParams.m_fNoise.getValue(),6) << ", "
											<< number2String(bestParams.m_sNoise.getValue(),3) << ", "
											<< "complex: " << number2String(correctness, 2) << " %%, logZ: " << bestLogZ;
									m_package->setAdditionalInfo(str2.str());
								}
								std::stringstream str;
								str << "New best params: " << bestParams << ", with correctness of: " << correctness;/*
									<< " %%, ones: " << (amountOfOnesCorrect / (double) amountOfOneChecks) * 100.
									<< " %%, minus ones: " << (amountOfMinusOnesCorrect / (double) amountOfMinusOneChecks) * 100.
									<< ", amount of minues correct: " << amountOfMinusOnesCorrect << ", amount of minus ones: " << amountOfMinusOneChecks
									<< " %%, for: " << m_dataPoints << " points";*/
								m_package->printLineToScreenForThisThread(str.str());
							}
						}else if(bestCorrectness == 0){ // for the starting cases	// in this case only the simple check was performed and the values
							// are not good enough to guarantee that these params are better
							// so always take the params with the lower logZ
							if(correctness > m_package->correctlyClassified() || bestLogZ < m_logZ){
								m_gaussKernel->getCopyOfParams(bestParams);
								bestLogZ = m_logZ;
								m_package->changeCorrectlyClassified(correctness);
								if(!m_gaussKernel->hasLengthMoreThanOneDim()){
									std::stringstream str2;
									str2 << "Best: " << number2String(bestParams.m_length.getValue(),3) << ", "
											<< number2String(bestParams.m_fNoise.getValue(),6) << ", "
											<< number2String(bestParams.m_sNoise.getValue(),3) << ", "
											<< "simple: " << number2String(correctness, 2) << " %%, logZ: " << bestLogZ;
									m_package->setAdditionalInfo(str2.str());
								}
								std::stringstream str;
								str << "New best params: " << bestParams << ", with simple correctness of: " << correctness;
								m_package->printLineToScreenForThisThread(str.str());
							}
						}
						//					printInPackageOnScreen(m_package, "\nBestParams: " << bestParams << ", with: " << bestLogZ);
					}
					swAvg.recordActTime();
					m_package->performedOneTrainingStep(); // adds a one to the counter
					if(m_package->shouldTrainingBeAborted()){
						m_package->printLineToScreenForThisThread("Training should be aborted!");
						break;
					}else if(m_package->shouldTrainingBePaused()){
						m_package->printLineToScreenForThisThread("Training has to wait!");
						m_package->wait(); // will hold this process
					}
//					if(iCounterSampling++ > 100){
//						break;
//					}
				}
				}
				if(Settings::getDirectBoolValue("IVM.Training.overwriteExistingHyperParams")){
					bestParams.writeToFile(kernelFilePath);
				}

//				m_gaussKernel->setHyperParamsWith(bestParams);
//				if(CommandSettings::get_visuRes() > 0. || CommandSettings::get_visuResSimple() > 0.){
//					setDerivAndLogZFlag(false, false);
//					m_uniformNr.setMinAndMax(1, 1); // final training with all points considered
//					internalTrain(true, verboseLevel);
//					DataWriterForVisu::writeSvg("before_ivm_"+number2String((int)m_labelsForClasses.coeff(0))+".svg", *this, m_I, m_storage.storage());
//					openFileInViewer("before_ivm_"+number2String((int)m_labelsForClasses.coeff(0))+".svg");
//				}
//				int iGradientCounter = 0;
//				List<double> logZs;
//				GaussianKernelParams eSquared;
//				eSquared.setAllValuesTo(0);
//				m_uniformNr.setMinAndMax(1, m_dataPoints / 100);
//				setDerivAndLogZFlag(true, true);
//				while(m_package != nullptr){ // equals a true
//					const bool trained = internalTrain(iGradientCounter % 10 == 0, 0);
//					if(trained){
//						printInPackageOnScreen(m_package, "Gradient is: " << m_derivLogZ << ", p: " << m_gaussKernel->getHyperParams() << ", l: " << m_logZ);
//						logZs.push_back(m_logZ);
//						if(!m_gaussKernel->hasLengthMoreThanOneDim()){
//							for(unsigned int j = 0; j < 3; ++j){
//								const double lastLearningRate = sqrt(EPSILON + eSquared.m_params[j]->getValue()); // 0,001
//								eSquared.m_params[j]->getValues()[0] = 0.9 * eSquared.m_params[j]->getValue() + 0.1 * m_derivLogZ.m_params[j]->getSquaredValue(); // 0,0000000099856
//								const double actLearningRate = sqrt(EPSILON + eSquared.m_params[j]->getValue());
//								m_gaussKernel->getHyperParams().m_params[j]->getValues()[0] += 0.000001 * (lastLearningRate / actLearningRate) * m_derivLogZ.m_params[j]->getValue();
//							}
//						}else{
//							m_gaussKernel->subGradient(m_derivLogZ, -0.00000005);
//						}
//					}else{
////						internalTrain(true, 0);
//					}
//					if(bestLogZ < m_logZ){
//						bestLogZ = m_logZ;
//						m_gaussKernel->getCopyOfParams(bestParams);
//						if(!m_gaussKernel->hasLengthMoreThanOneDim()){
//							std::stringstream str2;
//							str2 << "Best: " << number2String(bestParams.m_length.getValue(),3) << ", "
//									<< number2String(bestParams.m_fNoise.getValue(),6) << ", "
//									<< number2String(bestParams.m_sNoise.getValue(),3) << ", "
//									<< "logZ: " << bestLogZ;
//							m_package->setAdditionalInfo(str2.str());
//						}
//					}
//					if(iGradientCounter++ > 1000){
//						break;
//					}
//				}
//				DataWriterForVisu::writeSvg("logZ_"+number2String((int)m_labelsForClasses.coeff(0))+".svg", logZs);
			}
			if(bestLogZ == NEG_DBL_MAX){
				printError("This ivm could not find any parameter set in the given time, which could be trained without an error!");
				return false;
			}
			printOnScreen("For IVM: " << m_className << " logZ: " << bestLogZ << ", "<< bestParams << ", has: " << m_package->correctlyClassified() << ", with IPs: " << m_numberOfInducingPoints);
			m_gaussKernel->setHyperParamsWith(bestParams);
			setDerivAndLogZFlag(false, false);
			m_uniformNr.setMinAndMax(1, 1); // final training with all points considered
			const bool ret = internalTrain(true, verboseLevel);
			if(ret && !m_doEPUpdate){
				// train the whole active set again but in the oposite direction similiar to an ep step
				const bool ret2 = trainOptimizeStep(0);
				if(!ret2){
					printWarning("The optimization step could not be performed!");
				}
			}
			if((CommandSettings::get_visuRes() > 0. || CommandSettings::get_visuResSimple() > 0.) && CommandSettings::get_useFakeData() && Settings::getDirectBoolValue("VisuParams.visuFinalIvm")){
				DataWriterForVisu::writeSvg("ivm_"+m_className+".svg", *this, m_I, m_storage.storage());
				openFileInViewer("ivm_"+m_className+".svg");
			}
			m_uniformNr.setMinAndMax(1, m_dataPoints / 100);
			if(m_package != nullptr){
				m_package->finishedTask(); // tell thread master this thread is finished and will be done in just a second
			}
			return ret;
		}else{
			//		setDerivAndLogZFlag(false, false);
			m_gaussKernel->setHyperParamsWith(bestParams);
			m_uniformNr.setMinAndMax(1, 1);
			std::stringstream str;
			str << "Use hyperParams: " << bestParams;
			m_package->printLineToScreenForThisThread(str.str());
			const bool ret = internalTrain(true, verboseLevel);
			if(ret && !m_doEPUpdate){
				// train the whole active set again but in the oposite direction similiar to an ep step
				trainOptimizeStep(verboseLevel);
			}
			//		DataWriterForVisu::writeSvg("ivm_"+number2String((int)m_labelsForClasses.coeff(0))+".svg", *this, m_I, m_data);
			//		openFileInViewer("ivm_"+number2String((int)m_labelsForClasses.coeff(0))+".svg");
			m_uniformNr.setMinAndMax(1, m_dataPoints / 100);

			if(m_package != nullptr){
				m_package->finishedTask(); // tell thread master this thread is finished and will be done in just a second
			}
			return ret;
		}
	}else if(m_kernelType == RF){
		setDerivAndLogZFlag(false, false);
		m_uniformNr.setMinAndMax(1, 1);
		const bool ret = internalTrain(true, verboseLevel);
		printOnScreen("Training: " << ret);
		if(ret && !m_doEPUpdate){
			// train the whole active set again but in the oposite direction similiar to an ep step
//			trainOptimizeStep(verboseLevel);
			m_trained = true;
		}
		if(CommandSettings::get_visuRes() > 0. || CommandSettings::get_visuResSimple() > 0.){
			printInPackageOnScreen(m_package, "Training finished only visu has to be done!");
			DataWriterForVisu::writeSvg("ivm_"+m_className+".svg", *this, m_I, m_storage.storage());
			openFileInViewer("ivm_"+m_className+".svg");
		}
		//		DataWriterForVisu::writeSvg("ivm_"+number2String((int)m_labelsForClasses.coeff(0))+".svg", *this, m_I, m_data);
		//		openFileInViewer("ivm_"+number2String((int)m_labelsForClasses.coeff(0))+".svg");
		m_uniformNr.setMinAndMax(1, m_dataPoints / 100);
		if(m_package != nullptr){
			m_package->finishedTask(); // tell thread master this thread is finished and will be done in just a second
		}
		return ret;
	}else{
		printError("This kernel type is not supported!");
	}
	return false;
}

bool IVM::internalTrain(bool clearActiveSet, const int verboseLevel){
	if(m_kernelType == GAUSS){
		if(verboseLevel == 2 && m_gaussKernel->wasDifferenceCalced())
			printInPackageOnScreen(m_package, "Diff: " << m_gaussKernel->getDifferences(0,0) << ", " << m_gaussKernel->getDifferences(1,0));
	}
	Vector m = Vector::Zero(m_dataPoints);
	Vector beta = Vector::Zero(m_dataPoints);
	Vector mu = Vector::Zero(m_dataPoints);
	Vector zeta = Vector(m_dataPoints);
	if(clearActiveSet){
		m_I.clear();
	}else{
		if(m_I.size() != m_numberOfInducingPoints){
			printError("The active set size is not correct! Reset active set!");
			m_I.clear();
			clearActiveSet = true;
		}
	}
	m_J.clear();
	Eigen::Vector2i amountOfPointsPerClass;
	amountOfPointsPerClass[0] = amountOfPointsPerClass[1] = 0;
	for(unsigned int i = 0; i < m_dataPoints; ++i){
		if(m_kernelType == GAUSS){
			zeta.coeffRef(i) = m_gaussKernel->calcDiagElement(i);
		}else if(m_kernelType == RF){
			zeta.coeffRef(i) = m_rfKernel->calcDiagElement(i);
		}else{
			zeta.coeffRef(i) = 0;
		}
		m_J.push_back(i);
		++amountOfPointsPerClass[(m_y.coeff(i) == 1 ? 0 : 1)];
	}
	Vector g = Vector(m_numberOfInducingPoints);
	Vector nu = Vector(m_numberOfInducingPoints);
	Vector delta = Vector(m_numberOfInducingPoints);
	StopWatch updateMat, findPoints;
	findPoints.startTime();
	double fraction = 0.;
	//printInPackageOnScreen(m_package, "bias: " << m_bias);
	List<unsigned int>::const_iterator itOfActiveSet = m_I.begin();
	Vector s_nk = Vector(m_dataPoints), k_nk = Vector(m_dataPoints); // k_nk is not filled for k == 0!!!!
	const bool visuDeltas = !m_uniformNr.isUsed() && Settings::getDirectBoolValue("VisuParams.visuEntropyForFinalIvm");
	List<double> deltaValues;
	List<std::string> colors;
	int forceUsedOfClass = 0; // 1 or -1 are the possible classes (0 == non selected)
//	List<double> informationOfUsedValues;
	for(unsigned int k = 0; k < m_numberOfInducingPoints; ++k){

		if(m_kernelType == RF){
			printInPackageOnScreen(m_package, "Calculation of inducing point nr: " << k);
		}
		int argmax = -1;
		//List<Pair<int, double> > pointEntropies;
		delta.coeffRef(k) = NEG_DBL_MAX;
		if(clearActiveSet){
//			List<double> deltasValue;
//			List<std::string> colorForDeltas;

			unsigned int increaseValue = 1; // if no is activated this is the standart case
			List<unsigned int>::const_iterator itOfJ = m_J.begin();
			// do not jump over values in the first 10 iterations or
			// if the desired amount of points for one of the classes is below 0.35
			// and the actual fraction is below that desired point, this can be done,
			// because in this case only 0.35 of the data has to be searched and the rest is skipped
			if(m_uniformNr.isUsed() && k > 10 && !((m_desiredPoint < 0.35 && fraction < m_desiredPoint - m_desiredMargin)
					|| (m_desiredPoint > 0.65 && fraction > m_desiredPoint + m_desiredMargin))){
				increaseValue = m_uniformNr(); // returns a random value
				const unsigned int start = m_uniformNr();
				for(unsigned int i = 0; i < start; ++i){
					++itOfJ;
				}
			}
			while(itOfJ != m_J.end()){
				double gForJ, nuForJ;
				double deltaForJ = calcInnerOfFindPointWhichDecreaseEntropyMost(*itOfJ, zeta, mu, gForJ, nuForJ, fraction, amountOfPointsPerClass, forceUsedOfClass, verboseLevel);
				// deltaForJ == NEG_DBL_MAX means this class should not be used (fraction requirment not fullfilled!)

//				printInPackageOnScreen(m_package, "Point: " << *itOfJ << ", with: " << deltaForJ << " and nu: " << nuForJ);
				if(deltaForJ > NEG_DBL_MAX && nuForJ > 0.){ // if nuForJ is smaller 0 it shouldn't be considered at all
					if(m_useNeighbourComparison){
						unsigned int informationCounter = 0;
						const double labelOfJ = m_y.coeff(*itOfJ);
						for(List<unsigned int>::const_iterator itOfI = m_I.begin(); itOfI != m_I.end(); ++itOfI, ++informationCounter){
							if(labelOfJ == m_y.coeff(*itOfI)){ // only if they have the same class
								double similiarty = 0;
								if(m_kernelType == GAUSS){
									similiarty = m_gaussKernel->kernelFunc(*itOfI, *itOfJ);
								}else if(m_kernelType == RF){
									similiarty = m_rfKernel->kernelFunc(*itOfI, *itOfJ);
								}
								deltaForJ += similiarty * delta.coeff(informationCounter); // plus, because all values are negative, will decrease the information
							}
						}
					}
					if(deltaForJ > delta.coeff(k) && nuForJ > 0.){
						argmax = *itOfJ;
						delta.coeffRef(k) = deltaForJ;
						g.coeffRef(k) = gForJ;
						nu.coeffRef(k) = nuForJ;
					}
//					deltasValue.push_back(deltaForJ);
//					colorForDeltas.push_back(std::string(m_y[*itOfJ] == 1 ? "red" : "blue"));
					increaseValue = m_uniformNr(); // returns a random value
					for(unsigned int i = 0; i < increaseValue; ++i){
						++itOfJ; // increases the iterator
						if(itOfJ == m_J.end()){ // controls if the loop should be ended
							break;
						}
					}
				}else{
					// go only one further if it does not work
					++itOfJ;
				}
			}
//			if(k > 0){
////				printInPackageOnScreen(m_package, "g[k] is " << g[k]);
//				double min, max;
//				DataConverter::getMinMax(deltasValue, min, max);
//				if(min == max){
//					// problem
//					DataConverter::getMinMax(mu, min, max);
//					if(min == max){
//						printInPackageOnScreen(m_package, "the complete mu is the same!");
//					}
//				}
//			}
//			printInPackageOnScreen(m_package, "min: " << min << ", max: " << max);
//			printInPackageOnScreen(m_package, "mu: " << mu.transpose());
//			DataWriterForVisu::writeSvg("deltas1.svg", deltasValue, colorForDeltas);
//			openFileInViewer("deltas1.svg");
//			sleep(1);
			if(visuDeltas){
				deltaValues.push_back((double) delta[k]);
				colors.push_back(std::string(m_y[argmax] == 1 ? "red" : "blue"));
			}
		}else{
			argmax = *itOfActiveSet;
			double gForArgmax, nuForArgmax;
			// TODO remove calc of Inner not needed here
			delta.coeffRef(k) = calcInnerOfFindPointWhichDecreaseEntropyMost(argmax, zeta, mu, gForArgmax, nuForArgmax, fraction, amountOfPointsPerClass, 0, verboseLevel);
			g.coeffRef(k) = gForArgmax;
			nu.coeffRef(k) = nuForArgmax;
			++itOfActiveSet;
		}
//		printInPackageOnScreen(m_package, "New point was found: " << argmax);
		if(argmax == -1 && m_J.size() > 0){
			if(fraction < m_desiredPoint - m_desiredMargin && forceUsedOfClass == 0){
				forceUsedOfClass = 1;
			}else if(fraction > m_desiredPoint - m_desiredMargin && forceUsedOfClass == 0){
				forceUsedOfClass = -1;
			}else{
				return false;
				if(m_numberOfInducingPoints * 0.1 < m_I.size() && m_I.size() > 5){
					// change the nr of inducing points:
					setNumberOfInducingPoints(m_I.size());
					int l = 0;
					for(List<unsigned int>::const_iterator itOfI = m_I.begin(); itOfI != m_I.end(); ++itOfI, ++l){
						m_nuTilde.coeffRef(l) = m.coeff(*itOfI) * beta.coeff(*itOfI);
						m_tauTilde.coeffRef(l) = beta.coeff(*itOfI);
					}
					m_muTildePlusBias = m_nuTilde.cwiseQuotient(m_tauTilde) + (m_bias * Vector::Ones(m_numberOfInducingPoints));
					// calc m_L
					Eigen::MatrixXd copyOfL = m_L;
					m_L.resize(m_numberOfInducingPoints, m_numberOfInducingPoints);
					for(unsigned int q = 0; q < m_numberOfInducingPoints; ++q){
						for(unsigned int p = 0; p < m_numberOfInducingPoints; ++p){
							m_L.coeffRef(p,q) = copyOfL.coeff(p,q);
						}
					}
					m_choleskyLLT.compute(m_L);
					if(m_calcLogZ){
						calcLogZ();
					}else if(m_calcDerivLogZ){
						printError("The derivative can not be calculated without the log!");
					}
					std::cout << "New inducing points: " << m_numberOfInducingPoints << std::endl;
					return true;
				}else{
					printError("No new inducing point found and the nr of inducing points is to low");
					return false;
				}
			}
			--k;
			continue;
//			if(verboseLevel != 0){
//				for(List<unsigned int>::const_iterator it = m_I.begin(); it != m_I.end(); ++it){
//					printInPackageOnScreen(m_package, "(" << *it << ", " << (double) m_y[*it] << ")");
//				}
//				std::string classRes = "";
//				if(fraction < m_desiredPoint - m_desiredMargin){
//					classRes = "1";
//				}else if(fraction > m_desiredPoint - m_desiredMargin){
//					classRes = "-1";
//				}else{
//					classRes = "1 or -1";
//				}
//				printInPackageOnScreen(m_package, "m_desiredPoint: " << m_desiredPoint << ", fraction: " << fraction);
//				printError("No new inducing point was found and there are still points over and next point should be from class: " << classRes << "!");
//			}
//			return false;
		}else if(argmax == -1){
			if(verboseLevel != 0)
				printError("No new inducing point was found, because no points are left to process, number of inducing points: "
						<< m_numberOfInducingPoints << ", size: " << m_dataPoints);
			return false;
		}
		forceUsedOfClass = 0;

		fraction = ((fraction * k) + (m_y.coeff(argmax) == 1 ? 1 : 0)) / (double) (k + 1);
		if(verboseLevel == 2)
			printDebug("Next i is: " << argmax << " has label: " << (double) m_y.coeff(argmax));
		// refine site params, posterior params & M, L, K
		if(fabs((double)g.coeff(k)) < EPSILON){
			m.coeffRef(argmax) = mu.coeff(argmax);
		}else if(fabs((double)nu.coeff(k)) > EPSILON){
			m.coeffRef(argmax) = g.coeff(k) / nu.coeff(k) + mu.coeff(argmax);
		}else{
			printError("G is zero and nu is not!");
			return false;
		}
		beta.coeffRef(argmax) = nu.coeff(k) / (1.0 - nu.coeff(k) * zeta.coeff(argmax));
		if(beta.coeff(argmax) < EPSILON){
			beta.coeffRef(argmax) = EPSILON;
		}
//		Vector a_nk;
		if(k != 0){
			for(unsigned int i = 0; i < m_dataPoints; ++i){
				if(m_kernelType == GAUSS){
					k_nk.coeffRef(i) = m_gaussKernel->kernelFunc(i, argmax);
				}else if(m_kernelType == RF){
					k_nk.coeffRef(i) = m_rfKernel->kernelFunc(i, argmax);
				}
			}
			for(unsigned int i = 0; i < m_dataPoints; ++i){ // TODO for known active set only the relevant values have to been updated!
				double temp = 0.;
				for(unsigned int j = 0; j < k; ++j){
					temp += m_M.coeff(j, argmax) * m_M.coeff(j,i);
				}
				s_nk.coeffRef(i) = k_nk.coeff(i) - temp; // s_nk = k_nk - temp;
			}
			/*Vector colVec = m_M.col(argmax);
			s_nk = k_nk - (colVec.transpose() * m_M).transpose();*/
		}else{
			for(unsigned int i = 0; i < m_dataPoints; ++i){
				if(m_kernelType == GAUSS){
					s_nk.coeffRef(i) = m_gaussKernel->kernelFunc(i, argmax);
				}else if(m_kernelType == RF){
					s_nk.coeffRef(i) = m_rfKernel->kernelFunc(i, argmax);
				}
			}
		}
		if(verboseLevel == 2){
			printInPackageOnScreen(m_package, "Next: " << argmax);
			printInPackageOnScreen(m_package, "zeta: " << zeta.transpose());
			printInPackageOnScreen(m_package, "mu: " << mu.transpose());
			//	printInPackageOnScreen(m_package, "k_nk: " << k_nk.transpose());
			printInPackageOnScreen(m_package, "s_nk: " << s_nk.transpose());
		}
		//zeta -= ((double) nu[k]) * s_nk.cwiseProduct(s_nk);
		//mu += ((double) g[k]) * s_nk; // <=> mu += g[k] * s_nk;
//		printInPackageOnScreen(m_package, "s_nk: " << min1 << ", " << max1);
		for(unsigned int i = 0; i < m_dataPoints; ++i){ // TODO for known active set only the relevant values have to been updated!
			zeta.coeffRef(i) -= nu.coeff(k) * (s_nk.coeff(i) * s_nk.coeff(i)); // <=> zeta -= nu[k] * s_nk.cwiseProduct(s_nk); // <=> diag(A^new) = diag(A) - (u^2)_j
			mu.coeffRef(i) += g.coeff(k) * s_nk.coeff(i); // <=> mu += g[k] * s_nk; // h += alpha_i * ( K_.,i - M_.,i^T * M_.,i) <=> alpha_i * (k_nk - s_nk)
		}
		/* IVM script:
		 * h += alpha_i * l / sqrt(p_i) * ->mu
		 * h += alpha_i * l / sqrt(p_i) * (1 / l * (sqrt(p_i) * K_.,i - sqrt(p_i) * M_.,i^T * M_.,i))
		 * h += alpha_i * l / sqrt(p_i) * (sqrt(p_i) / l * (K_.,i - M_.,i^T * M_.,i))
		 * h += alpha_i * (K_.,i - M_.,i^T * M_.,i)
		 * diag(A) -= (u_j^2)_j
		 * diag(A) -= l^-2 * (sqrt(p_i) / l * (K_.,i - M_.,i^T * M_.,i))^2
		 * diag(A) -= l^-2 * (p_i / l^2 * (K_.,i - M_.,i^T * M_.,i)^2)
		 * diag(A) -= p_i * (K_.,i - M_.,i^T * M_.,i)^2
		 * for: s_nk = (K_.,i - M_.,i^T * M_.,i)
		 * diag(A) -= p_i * s_nk.cwiseProduct(s_nk)
		 */
		if(nu.coeff(k) < 0.0){
			if(verboseLevel != 0){
				printError("The actual nu is below zero: " <<  (double) nu.coeff(k));
				for(List<unsigned int>::const_iterator it = m_I.begin(); it != m_I.end(); ++it){
					printInPackageOnScreen(m_package, "(" << *it << ", " << (double) m_y.coeff(*it) << ")");
				}
			}
			return false;
		}
		const double sqrtNu = sqrt((double)nu.coeff(k));
		// update K and L
		/*
		if(k == 0){
			m_K = Matrix(1,1);
			m_K(0,0) = m_kernel.calcDiagElement();
			m_L = Matrix(1,1);
			m_L(0,0) = 1.0 / sqrtNu;
		}else{
			Vector k_vec = Vector(m_I.size());
			unsigned int t = 0;
			for(List<unsigned int>::const_iterator itOfI = m_I.begin(); itOfI != m_I.end(); ++itOfI, ++t){
				k_vec[t] = m_kernel.kernelFunc(*itOfI, argmax);
			}
			Matrix D(m_K.rows() + 1, m_K.cols() + 1);
			D << m_K, k_vec,
			     k_vec.transpose(), m_kernel.calcDiagElement();
			m_K = D;
			// update L
			a_nk = m_M.col(argmax);
			Matrix D2(m_L.rows() + 1, m_L.cols() + 1);
			D2 << m_L, Vector::Zero(k),
					a_nk.transpose(), 1. / sqrtNu;
			m_L = D2;
		}*/
		if(k==0){
			if(m_doEPUpdate){
				m_K = Matrix(m_numberOfInducingPoints, m_numberOfInducingPoints); // init at beginning to avoid realloc
				if(m_kernelType == GAUSS){
					m_K.coeffRef(0,0) = m_gaussKernel->calcDiagElement(0);
				}else if(m_kernelType == RF){
					m_K.coeffRef(0,0) = m_rfKernel->calcDiagElement(0);
				}
			}
			m_L = Matrix::Zero(m_numberOfInducingPoints, m_numberOfInducingPoints);
			m_L.coeffRef(0,0) = 1.0 / sqrtNu;
			m_M = Matrix(m_numberOfInducingPoints, m_dataPoints);
		}else{
			if(m_doEPUpdate){
				unsigned int t = 0;
				const unsigned int lastRowAndCol = k;
				for(List<unsigned int>::const_iterator itOfI = m_I.begin(); itOfI != m_I.end() && t < lastRowAndCol; ++itOfI, ++t){
					// uses the kernel matrix from the actual element with all other elements in the active set
					const double temp = k_nk.coeff(*itOfI); // <=> is the same: m_kernel.kernelFunc(*itOfI, argmax); saves recalc
					m_K.coeffRef(lastRowAndCol, t) = temp;
					m_K.coeffRef(t, lastRowAndCol) = temp;
				}
				if(m_kernelType == GAUSS){
					m_K.coeffRef(lastRowAndCol, lastRowAndCol) = m_gaussKernel->calcDiagElement(lastRowAndCol);
				}else if(m_kernelType == RF){
					m_K.coeffRef(lastRowAndCol, lastRowAndCol) = m_rfKernel->calcDiagElement(lastRowAndCol);
				}
			}
			// update L
			if(argmax < m_M.cols()){
				for(unsigned int i = 0; i < k; ++i){
					m_L.coeffRef(k,i) = m_M.coeff(i, argmax); // a_nk[i]; with a_nk = m_M.col(argmax);
				}
				m_L.coeffRef(k, k) = 1. / sqrtNu;
			}else{
				printError("The argmax value is bigger than the amount of columns in M!"); return false;
			}
		}
		// update M
		/*if(k == 0){
			m_M = Matrix(m_numberOfInducingPoints, m_dataPoints);
			for(unsigned int i = 0; i < m_dataPoints; ++i){
				m_M(0,i) = sqrtNu * s_nk[i];
			}
		}else{
			Matrix D(m_M.rows() + 1, m_M.cols());
			D << m_M,
				(sqrtNu * s_nk).transpose();
			m_M = D;
		}*/
		for(unsigned int i = 0; i < m_dataPoints; ++i){
			m_M.coeffRef(k,i) = sqrtNu * s_nk.coeff(i);
		}
		m_listMutex.lock();
		const unsigned int uiArgmax = argmax;
		if(uiArgmax < m_dataPoints){ // check that is not -1 == UINT_MAX
			if(clearActiveSet){
				m_I.push_back(argmax);
			}
			m_J.remove(argmax);
		}
		m_listMutex.unlock();
		amountOfPointsPerClass.coeffRef(m_y.coeff(argmax) == 1 ? 0 : 1) -= 1;
	}


	if(verboseLevel == 2){
		int classOneCounter = 0;
		for(List<unsigned int>::const_iterator itOfI = m_I.begin(); itOfI != m_I.end(); ++itOfI){
			if(m_y.coeff(*itOfI) == 1){
				++classOneCounter;
			}
		}
		printInPackageOnScreen(m_package, "Fraction in including points is: " << classOneCounter / (double) m_I.size() * 100. << " %");
		printInPackageOnScreen(m_package, "Find " << m_numberOfInducingPoints << " points: " << findPoints.elapsedAsPrettyTime());
	}
	if(visuDeltas){
		DataWriterForVisu::writeSvg("deltas_" + m_className + ".svg", deltaValues, colors);
		openFileInViewer("deltas_" + m_className + ".svg");
	}
	if(m_I.size() != m_numberOfInducingPoints){
		if(verboseLevel != 0)
			printError("The active set has not the desired amount of points");
		return false;
	}
	unsigned int l = 0;
	Vector muSqueezed(m_numberOfInducingPoints);
	for(List<unsigned int>::const_iterator itOfI = m_I.begin(); itOfI != m_I.end(); ++itOfI, ++l){
		m_nuTilde.coeffRef(l) = m.coeff(*itOfI) * beta.coeff(*itOfI);
		m_tauTilde.coeffRef(l) = beta.coeff(*itOfI);
		muSqueezed.coeffRef(l) = mu.coeff(*itOfI);
	}
	// calc m_L
//	printInPackageOnScreen(m_package, "m_L: \n" << m_L);
	//printInPackageOnScreen(m_package, "m_K: \n" << m_K);
	//printInPackageOnScreen(m_package, "m_M: \n" << m_M);
	m_choleskyLLT.compute(m_L);

//	printInPackageOnScreen(m_package, "L: \n" << m_choleskyLLT.matrixL().toDenseMatrix());
//	printInPackageOnScreen(m_package, "llt: \n" << m_choleskyLLT.matrixLLT());
//	printInPackageOnScreen(m_package, "before m_L: \n" << m_L);
//	m_muTildePlusBias = m_nuTilde.cwiseQuotient(m_tauTilde) + (m_bias * Vector::Ones(m_numberOfInducingPoints));
//	printInPackageOnScreen(m_package, "mu tilde before: " << m_muTildePlusBias.transpose());
	if(m_doEPUpdate){ // EP update
		Matrix Sigma = m_K * (m_eye - m_choleskyLLT.solve(m_K));
		//Matrix controlSigma = m_K * (I - m_choleskyLLT.solve(m_K));
		double deltaMax = 1.0;
		const unsigned int maxEpCounter = 100;
		double epThreshold = 1e-7;
		List<double> listToPrint;
		//double minDelta = DBL_MAX;
		StopWatch updateEP;
		StopWatch sigmaUp, sigmaUpNew;
		unsigned int counter = 0;
//		printInPackageOnScreen(m_package, "Sigma: \n" << Sigma);
//		printInPackageOnScreen(m_package, "m_K: \n" << m_K);
//		printInPackageOnScreen(m_package, "(m_eye - m_choleskyLLT.solve(m_K)): \n" << (m_eye - m_choleskyLLT.solve(m_K)));
		for(; counter < maxEpCounter && deltaMax > epThreshold; ++counter){
			updateEP.startTime();
			Vector deltaTau(m_numberOfInducingPoints);
//			printInPackageOnScreen(m_package, "<<< " << counter << " <<<");
			unsigned int i = 0;
			for(List<unsigned int>::const_iterator itOfI = m_I.begin(); itOfI != m_I.end(); ++itOfI, ++i){
				const double tauMin = 1. / Sigma.coeff(i,i) - m_tauTilde.coeff(i);
				const double nuMin  = muSqueezed.coeff(i) / Sigma.coeff(i,i) - m_nuTilde.coeff(i);
				const unsigned int index = (*itOfI);
				const double label = m_y.coeff(index);

				const std::complex<double> tau_c(tauMin, 0);
				//double denom = std::max(abs(sqrt(tau_c * (tau_c / (lambda * lambda) + 1.))), EPSILON);
				double denom = std::max(std::abs((sqrt(tau_c * (tau_c / (m_lambda * m_lambda) + 1.0)))), EPSILON);
				const double c = label * tauMin / denom;
				double u;
				if(fabs(nuMin) < EPSILON){
					u = c * m_bias;
				}else{
					u = label * nuMin / denom + c * m_bias;
				}
				const double dlZ = c * exp(cumulativeDerivLog(u) - cumulativeLog(u));
				const double d2lZ  = dlZ * (dlZ + u * c);

				const double oldTauTilde = m_tauTilde.coeff(i);
				denom = 1.0 - d2lZ / tauMin;
				m_tauTilde.coeffRef(i) = std::max(d2lZ / denom, 0.);
				m_nuTilde.coeffRef(i)  = (dlZ + nuMin / tauMin * d2lZ) / denom;
				deltaTau.coeffRef(i)  = m_tauTilde.coeff(i) - oldTauTilde;
//				printInPackageOnScreen(m_package, "Label of " << (*itOfI) << " is: " << label
//						<< ", has " << m_muTildePlusBias[i] << ", old tau: "
//						<< oldTauTilde << ", new tau: " << m_tauTilde[i]
//						<< ", new new: " << m_nuTilde[i] << ", Sigma(i,i): " << Sigma(i,i));
													  /*<< ", dlZ: " << dlZ
						<< ", d2lZ: " << d2lZ << ", c: " << c << ", tauMin: " << tauMin <<", inner denom: "
						<< std::abs((sqrt(tau_c * (tau_c / (m_lambda * m_lambda) + 1.0)))));*/

				// update approximate posterior
				/*
				sigmaUpNew.startTime();
				Vector si = Sigma.col(i);
				denom = 1.0 + deltaTau[i] * si[i];
				//if(fabs(denom) > EPSILON)
				Sigma -= (deltaTau[i] / denom) * (si * si.transpose());
				sigmaUpNew.recordActTime();
				 */
				sigmaUpNew.startTime();
				const Vector oldSigmaCol = Sigma.col(i);
				denom = 1.0 + deltaTau.coeff(i) * oldSigmaCol.coeff(i); // <=> 1.0 + deltaTau[i] * si[i] for si = Sigma.col(i)
				const double fac = deltaTau.coeff(i) / denom;
				// is the same as Sigma -= (deltaTau[i] / denom) * (si * si.transpose()); but faster
				for(int p = 0; p < m_I.size(); ++p){
					Sigma.coeffRef(p,p) -= fac * oldSigmaCol.coeff(p) * oldSigmaCol.coeff(p);
					for(int q = p + 1; q < m_I.size(); ++q){
						const double sub = fac * oldSigmaCol.coeff(p) * oldSigmaCol.coeff(q);
						Sigma.coeffRef(p,q) -= sub;
						Sigma.coeffRef(q,p) -= sub;
					}
				}
				sigmaUpNew.recordActTime();

				/*for(int p = 0; p < m_I.size(); ++p){
			 	for(int q = 0; q < m_I.size(); ++q){
			 		if(fabs(controlSigma(p,q) - Sigma(p,q)) > fabs(Sigma(p,q)) * 1e-7){
			 			printError("Calc is wrong!");
			 		}
			 	}
			 }*/
				//else
				//Sigma -= delta_tau[i] / EPSILON * GP_Matrix::OutProd(si);
				muSqueezed = Sigma * m_nuTilde;
			}
			/*Vector _s_sqrt = Vector(n); // not used in the moment
		 const double sqrtEps = sqrt(EPSILON);
		 for(unsigned int i=0; i<m_numberOfInducingPoints; i++){
			 if(m_tauTilde[i] > EPSILON)
				 _s_sqrt[i] = sqrt(m_tauTilde[i]);
			 else
				 _s_sqrt[i] = sqrtEps;
		 }*/
			deltaMax = deltaTau.cwiseAbs().maxCoeff();
			listToPrint.push_back(deltaMax);
			//minDelta = std::min(minDelta, deltaMax);
			updateEP.recordActTime();
		}
		printInPackageOnScreen(m_package, "new sigma up time: " << sigmaUpNew.elapsedAvgAsPrettyTime());
		printInPackageOnScreen(m_package, "total new sigma up time: " << sigmaUpNew.elapsedAvgAsTimeFrame() * ((double) counter * m_I.size()));
		printInPackageOnScreen(m_package, "Ep time: " << updateEP.elapsedAvgAsPrettyTime());
		printInPackageOnScreen(m_package, "Total ep time: " << updateEP.elapsedAvgAsTimeFrame() * (double) counter);
		//printInPackageOnScreen(m_package, "Min delta: " << minDelta);

		DataWriterForVisu::writeSvg("deltas" + m_className + ".svg", listToPrint, true);
		openFileInViewer("deltas" + m_className + ".svg");
	/*	Matrix temp = m_K;
		for(unsigned int i = 0; i < m_tauTilde.rows(); ++i){
			temp(i,i) += 1. / (double) m_tauTilde[i];
		}
		m_choleskyLLT.compute(temp);
		m_L = temp;*/
		m_L = m_K + DiagMatrixXd(m_tauTilde.cwiseInverse()).toDenseMatrix();
		m_choleskyLLT.compute(m_L);
		// compute log z
	}
	if(m_calcLogZ){
		calcLogZ();
	}else if(m_calcDerivLogZ){
		printError("The derivative can not be calculated without the log!");
	}
	m_muTildePlusBias = m_nuTilde.cwiseQuotient(m_tauTilde) + (m_bias * Vector::Ones(m_numberOfInducingPoints));
//	printInPackageOnScreen(m_package, "m_muTildePlusBias: " << m_muTildePlusBias.transpose());
//	printInPackageOnScreen(m_package, "after m_L: \n" << m_choleskyLLT.matrixL().toDenseMatrix());
//	printInPackageOnScreen(m_package, "mu tilde before flipping: " << m_muTildePlusBias.transpose());
	//unsigned int t = 0;
	/*for(List<unsigned int>::const_iterator itOfI = m_I.begin(); itOfI != m_I.end(); ++itOfI, ++t){
		if(m_y[*itOfI] < 0 ? m_muTildePlusBias[t] > 0 : m_muTildePlusBias[t] < 0){
			m_muTildePlusBias[t] *= -1.;
		}
	}*/
//	printInPackageOnScreen(m_package, "mu tilde after: " << m_muTildePlusBias.transpose());
	m_trained = true;
	return true;
}

bool IVM::trainOptimizeStep(const int verboseLevel){
	if(m_I.size() > 0){
//		List<Vector> vecs;
//		vecs.push_back(m_muTildePlusBias);
		Vector oldMuTildeBias(m_muTildePlusBias);
		m_I.reverse(); // flip order!
		const bool ret = internalTrain(false, verboseLevel);
		if(ret){
//			vecs.push_back(m_muTildePlusBias);
//			vecs.rbegin()->reverse();
//			Vector diff(m_muTildePlusBias);
			for(unsigned int i = 0; i < m_numberOfInducingPoints; ++i){
				m_muTildePlusBias.coeffRef(i) += oldMuTildeBias.coeff(m_numberOfInducingPoints - i - 1);
				m_muTildePlusBias.coeffRef(i) *= 0.5;
//				diff[i] -= oldMuTildeBias[m_numberOfInducingPoints - i - 1];
//				diff[i] = fabs(diff[i]);
			}
//			DataWriterForVisu::writeSvg("muTildeBias.svg", vecs, false);
//			openFileInViewer("muTildeBias.svg");
//			DataWriterForVisu::writeSvg("muTildeBiasDiff.svg", diff, false);
//			openFileInViewer("muTildeBiasDiff.svg");
			return true;
		}else{
			m_I.reverse(); // flip order!
			return false;
		}

	}else{
		printError("This function can only be called, if the initial training was performed!");
		return false;
	}
}

void IVM::calcLogZ(){
	m_logZ = 0.0;
	const Matrix& llt = m_choleskyLLT.matrixLLT();
	//printInPackageOnScreen(m_package, "llt: \n" << llt);
	for(unsigned int i = 0; i < m_numberOfInducingPoints; ++i){
		m_logZ -= log((double) llt.coeff(i,i));
	}
	if(false){
		const Vector muTilde = m_nuTilde.cwiseQuotient(m_tauTilde);
		Vector muL0 = Vector::Zero(m_numberOfInducingPoints);
		Vector muL0_2 = Vector::Zero(m_numberOfInducingPoints);
		List<unsigned int>::const_iterator it = m_I.begin();
		for(uint i=0; i<m_numberOfInducingPoints; ++i, ++it){
			if(m_y.coeff(*it) == 1){
				double sum = muTilde.coeff(i);
				for(int k = (int)i-1; k >= 0; --k){
					sum -= (double)llt.coeff(i,k) * muL0.coeff(k);
				}
				muL0.coeffRef(i) = sum / (double) llt.coeff(i,i);
			}else{
				double sum = muTilde.coeff(i);
				for(int k = (int)i-1; k >= 0; --k){
					sum -= (double)llt.coeff(i,k) * muL0_2.coeff(k);
				}
				muL0_2.coeffRef(i) = sum / (double) llt.coeff(i,i);
			}
		}
		Vector muL1 = Vector::Zero(m_numberOfInducingPoints);
		Vector muL1_2 = Vector::Zero(m_numberOfInducingPoints);
		List<unsigned int>::const_reverse_iterator itr = m_I.rbegin();
		for(int i= (int) m_numberOfInducingPoints - 1; i >= 0; --i, ++itr){
			if(m_y.coeff(*itr) == 1){
				double sum = muL0.coeff(i);
				for(int k = i+1; k < m_numberOfInducingPoints; ++k){
					sum -= (double)llt.coeff(k,i) * muL1.coeff(k);
				}
				muL1.coeffRef(i) = sum / (double)llt.coeff(i,i);
			}else{
				double sum = muL0.coeff(i);
				for(int k = i+1; k < m_numberOfInducingPoints; ++k){
					sum -= (double)llt.coeff(k,i) * muL1_2.coeff(k);
				}
				muL1_2.coeffRef(i) = sum / (double)llt.coeff(i,i);
			}
		}
		m_logZ -= (1. - m_desiredPoint) * (0.5 * muTilde.dot(muL1)) + m_desiredPoint * (0.5 * muTilde.dot(muL1_2));
		if(m_calcDerivLogZ){
			calcDerivatives(muL1);
		}
	}else{
		const Vector muTilde = m_nuTilde.cwiseQuotient(m_tauTilde);
		Vector muL0 = Vector::Zero(m_numberOfInducingPoints);
		for(uint i=0; i<m_numberOfInducingPoints; ++i){
			double sum = muTilde.coeff(i);
			for(int k = (int)i-1; k >= 0; --k){
				sum -= (double)llt.coeff(i,k) * muL0.coeff(k);
			}
			muL0.coeffRef(i) = sum / (double) llt.coeff(i,i);
		}
		Vector muL1 = Vector::Zero(m_numberOfInducingPoints);
		for(int i= (int) m_numberOfInducingPoints - 1; i >= 0; --i){
			double sum = muL0.coeff(i);
			for(int k = i+1; k < m_numberOfInducingPoints; ++k){
				sum -= (double)llt.coeff(k,i) * muL1.coeff(k);
			}
			muL1.coeffRef(i) = sum / (double)llt.coeff(i,i);
		}
		m_logZ -= 0.5 * muTilde.dot(muL1);
		if(m_calcDerivLogZ){
			calcDerivatives(muL1);
		}
	}
}

void IVM::calcDerivatives(const Vector& muL1){
	if(m_kernelType == GAUSS){
		m_derivLogZ.m_length.changeAmountOfDims(m_gaussKernel->hasLengthMoreThanOneDim());
		m_derivLogZ.setAllValuesTo(0);
		if(!m_gaussKernel->hasLengthMoreThanOneDim()){
			std::vector<Matrix> CMatrix(m_gaussKernel->getHyperParams().paramsAmount);
			Matrix Z2 = (muL1 * muL1.transpose()) - m_choleskyLLT.solve(m_eye) * 0.5;
			int i = 0;
			for(std::vector<unsigned int>::const_iterator it = m_gaussKernel->getHyperParams().usedParamTypes.begin();
					it != m_gaussKernel->getHyperParams().usedParamTypes.end(); ++it, ++i){
				const GaussianKernelElement* type = (const GaussianKernelElement*) KernelTypeGenerator::getKernelFor(*it);
				if(!type->isDerivativeOnlyDiag()){
					m_gaussKernel->calcCovarianceDerivativeForInducingPoints(CMatrix[i], m_I, type);
				}
				delete type;
			}
			for(unsigned int i = 0; i < m_numberOfInducingPoints; ++i){
				for(unsigned int j = 0; j < m_numberOfInducingPoints; ++j){
					const double z2Value = Z2(i,j);
					for(unsigned int u = 0; u < m_gaussKernel->getHyperParams().paramsAmount; ++u){ // for every kernel param
						if(!m_gaussKernel->getHyperParams().m_params[u]->isDerivativeOnlyDiag()){
							m_derivLogZ.m_params[u]->getValues()[0] += z2Value * CMatrix[u].coeff(i,j);
						}
					}
				}
			}
		}else{
			std::vector<Matrix> cMatrix(ClassKnowledge::amountOfDims() + m_gaussKernel->getHyperParams().paramsAmount - 1);
			const Matrix Z2 = (muL1 * muL1.transpose()) - m_choleskyLLT.solve(m_eye) * 0.5;
			int i = 0;
			for(unsigned int u = 0; u < m_gaussKernel->getHyperParams().paramsAmount; ++u){ // for every kernel param
				if(!m_gaussKernel->getHyperParams().m_params[u]->isDerivativeOnlyDiag()){
					if(m_gaussKernel->getHyperParams().m_params[u]->hasMoreThanOneDim()){
						for(unsigned int k = 0; k < ClassKnowledge::amountOfDims(); ++k){
							m_gaussKernel->calcCovarianceDerivativeForInducingPoints(cMatrix[i], m_I, m_gaussKernel->getHyperParams().m_params[u], k);
							++i;
						}
					}else{
						m_gaussKernel->calcCovarianceDerivativeForInducingPoints(cMatrix[i], m_I, m_gaussKernel->getHyperParams().m_params[u]);
						++i;
					}
				}
			}
			for(unsigned int i = 0; i < m_numberOfInducingPoints; ++i){
				for(unsigned int j = 0; j < m_numberOfInducingPoints; ++j){
					const double z2Value = Z2.coeff(i,j);
					int t = 0;
					for(unsigned int u = 0; u < m_gaussKernel->getHyperParams().paramsAmount; ++u){ // for every kernel param
						if(!m_gaussKernel->getHyperParams().m_params[u]->isDerivativeOnlyDiag()){
							if(m_gaussKernel->getHyperParams().m_params[u]->hasMoreThanOneDim()){
								for(unsigned int k = 0; k < ClassKnowledge::amountOfDims(); ++k){
									m_derivLogZ.m_params[u]->getValues()[k] += z2Value * cMatrix[t].coeff(i,j);
									++t;
								}
							}else{
								m_derivLogZ.m_params[u]->getValues()[0] += z2Value * cMatrix[t].coeff(i,j);
								++t;
							}
						}
					}
				}
			}
		}
	}else if(m_kernelType == RF){
		printError("This type has no deriviative");
	}
}

double IVM::calcInnerOfFindPointWhichDecreaseEntropyMost(const unsigned int j, const Vector& zeta, const Vector& mu, double& g_kn, double& nu_kn, const double fraction, const Eigen::Vector2i& amountOfPointsPerClassLeft, const int useThisLabel, const int verboseLevel){
	const double label = m_y.coeff(j);
	if(useThisLabel != 0 && label != useThisLabel){ // use this label if it is not return NEG_DBL_MAX, useThisLabel == 0 -> no forced us
		return NEG_DBL_MAX;
	}else if(amountOfPointsPerClassLeft.coeff(0) > 0 && amountOfPointsPerClassLeft.coeff(1) > 0){
		if((fraction < m_desiredPoint - m_desiredMargin && label == -1) || (fraction > m_desiredPoint - m_desiredMargin && label == 1)){
			// => only less than 20 % of data is 1 choose 1
			return NEG_DBL_MAX; // or only less than 20 % of data is -1 choose -1
		}
	}
	const double tau = 1.0 / zeta.coeff(j);
	const std::complex<double> tau_c(tau, 0);
	//double denom = std::max(abs(sqrt(tau_c * (tau_c / (lambda * lambda) + 1.))), EPSILON);
	const double denom = std::max(std::abs((sqrt(tau_c * (tau_c / (m_lambda * m_lambda) + 1.0)))), EPSILON);
	const double c = label * tau / denom;
	nu_kn = mu.coeff(j) / zeta.coeff(j);
	double u;
	if(fabs(nu_kn) < EPSILON){
		u = c * m_bias;
	}else{
		u = label * nu_kn / denom + c * m_bias;
	}
	g_kn = c * exp(cumulativeDerivLog(u) - cumulativeLog(u));
	nu_kn = g_kn * (g_kn + u * c);
	const double delta_kn = log(1.0 - nu_kn * (double) zeta.coeff(j)) / (2.0 * LOG2);
	//const double delta_kn = zeta[j] * nu_kn;
	// pointEntropies.append( (j, delta_ln));
	if(verboseLevel == 2){
		printInPackageOnScreen(m_package, (label == 1 ? RED : CYAN) << "j: " << j << ", is: " << label << ", with: "
				<< delta_kn << ", g: " << g_kn << ", nu: " << nu_kn << ", zeta: " << (double) zeta.coeff(j) << ", c: " << c << ", u: " << u<< RESET); }
	/*if(delta_kn > delta[k]){ // nu_kn > EPSILON avoids that the ivm is not trained
		//if(k == j){
		//if(nu_kn < 0.)
		delta[k] = delta_kn;
		nu[k] = nu_kn;
		g[k] = g_kn;
		argmax = j;
	}*/
	return delta_kn;
}

double IVM::calcErrorOnTrainingsData(const bool wholeDataSet, const List<unsigned int>& testPoints, double& oneError, double& minusOneError){
	double plusError = 0;
	double minusError = 0;
	int oneCounter = 0;
	if(wholeDataSet){
		for(unsigned int i = 0; i < m_dataPoints; ++i){
			const int label = m_storage[i]->getLabel();
			const double prob = predictOnTraining(i);
			if(label == getLabelForOne()){
				plusError += 1.0 - prob;
				++oneCounter;
			}else{
				minusError += prob;
			}
		}
		oneError = plusError / (double) oneCounter * 100.;
		minusOneError = (minusError / (double) (m_dataPoints - oneCounter)) * 100.;
		return (oneError + minusError) * 0.5;
	}else{
		for(List<unsigned int>::const_iterator it = testPoints.cbegin(); it != testPoints.cend(); ++it){
			const int label = m_storage[*it]->getLabel();
			const double prob = predictOnTraining(*it);
			if(label == getLabelForOne()){
				plusError += 1.0 - prob;
				++oneCounter;
			}else{
				minusError += prob;
			}
		}
		oneError = plusError / (double) oneCounter * 100.;
		minusOneError = (minusError / (double) (testPoints.size() - oneCounter)) * 100.;
		return (oneError + minusError) * 0.5;
	}
}

void IVM::testOnTrainingsData(int & amountOfOneChecks, int& amountOfOnesCorrect, int& amountOfMinusOneChecks,
		int& amountOfMinusOnesCorrect, double& correctness, const double probDiff,
		const bool onlyUseOnes, const bool wholeDataSet, const List<unsigned int>& testPoints){
	amountOfOneChecks = amountOfOnesCorrect = 0;
	amountOfMinusOneChecks = amountOfMinusOnesCorrect = 0;
	if(wholeDataSet){
		for(unsigned int i = 0; i < m_dataPoints; ++i){
			const int label = m_storage[i]->getLabel();
			const double prob = predictOnTraining(i);
			if(label == getLabelForOne()){
				if(prob > 0.5 + probDiff){
					++amountOfOnesCorrect;
				}
				++amountOfOneChecks;
			}else{
				if(prob < 0.5 - probDiff){
					++amountOfMinusOnesCorrect;
				}
				++amountOfMinusOneChecks;
			}
		}
	}else{
		for(List<unsigned int>::const_iterator it = testPoints.cbegin(); it != testPoints.cend(); ++it){
			const int label = m_storage[*it]->getLabel();
			const double prob = predictOnTraining(*it);
			if(label == getLabelForOne()){
				if(prob > 0.5 + probDiff){
					++amountOfOnesCorrect;
				}
				++amountOfOneChecks;
			}else{
				if(prob < 0.5 - probDiff){
					++amountOfMinusOnesCorrect;
				}
				++amountOfMinusOneChecks;
			}
		}
	}
	if(onlyUseOnes){
		correctness = (amountOfOnesCorrect / (double) amountOfOneChecks) * 100.;
	}else{
		// both classes are equally important, therefore the combination of the correctnes gives a good indiciation how good we are at the moment
		correctness = ((amountOfMinusOnesCorrect / (double) amountOfMinusOneChecks) * 0.5 + (amountOfOnesCorrect / (double) amountOfOneChecks) * 0.5) * 100.;
	}
}

double IVM::predict(const Vector& input) const{
	const unsigned int n = m_I.size();
	Vector k_star(n);
	unsigned int i = 0;
	double diagEle = 0;
	if(m_kernelType == GAUSS){
		for(List<unsigned int>::const_iterator itOfI = m_I.cbegin(); itOfI != m_I.cend(); ++itOfI, ++i){
			k_star[i] = m_gaussKernel->kernelFuncVec(input, *m_storage[*itOfI]);
		}
		diagEle = m_gaussKernel->calcDiagElement(0);
	}else if(m_kernelType == RF){
		for(List<unsigned int>::const_iterator itOfI = m_I.begin(); itOfI != m_I.end(); ++itOfI, ++i){
			k_star.coeffRef(i) = m_rfKernel->kernelFuncVec(input, *m_storage[*itOfI]);
		}
		diagEle = m_rfKernel->calcDiagElement(0);
	}

//	printInPackageOnScreen(m_package, "L: \n" << m_choleskyLLT.matrixL().toDenseMatrix());
//	printInPackageOnScreen(m_package, "llt: \n" << m_choleskyLLT.matrixLLT());
	const Vector v = m_choleskyLLT.solve(k_star);
	/*
	const Vector mu_tilde = m_nuTilde.cwiseQuotient(m_tauTilde);
	double mu_star = (mu_tilde + (m_bias * Vector::Ones(n))).dot(v);*/
	double mu_star = m_muTildePlusBias.dot(v);
	double sigma_star = (diagEle - k_star.dot(v));
	//printInPackageOnScreen(m_package, "mu_start: " << mu_star);
	//printInPackageOnScreen(m_package, "sigma_star: " << sigma_star);
	double contentOfSig = 0;
	if(1.0 / (m_lambda * m_lambda) + sigma_star < 0){
		contentOfSig = mu_star;
	}else{
		contentOfSig = (mu_star / sqrt(1.0 / (m_lambda * m_lambda) + sigma_star));
	}
	return boost::math::erfc(-contentOfSig / SQRT2) / 2.0;
}

double IVM::predictOnTraining(const unsigned int id){
	const unsigned int n = m_I.size();
	Vector k_star(n);
	double diagEle = 0;
	unsigned int i = 0;
	if(m_kernelType == GAUSS){
		for(List<unsigned int>::const_iterator itOfI = m_I.begin(); itOfI != m_I.end(); ++itOfI, ++i){
			k_star.coeffRef(i) = m_gaussKernel->kernelFunc(id, *itOfI);
		}
		diagEle = m_gaussKernel->calcDiagElement(0);
	}else if(m_kernelType == RF){
		for(List<unsigned int>::const_iterator itOfI = m_I.begin(); itOfI != m_I.end(); ++itOfI, ++i){
			k_star.coeffRef(i) = m_rfKernel->kernelFunc(id, *itOfI);
		}
		diagEle = m_rfKernel->calcDiagElement(0);
	}

	//	printInPackageOnScreen(m_package, "L: \n" << m_choleskyLLT.matrixL().toDenseMatrix());
	//	printInPackageOnScreen(m_package, "llt: \n" << m_choleskyLLT.matrixLLT());
	const Vector v = m_choleskyLLT.solve(k_star);
	/*
		const Vector mu_tilde = m_nuTilde.cwiseQuotient(m_tauTilde);
		double mu_star = (mu_tilde + (m_bias * Vector::Ones(n))).dot(v);*/
	double mu_star = m_muTildePlusBias.dot(v);
	double sigma_star = (diagEle - k_star.dot(v));
	//printInPackageOnScreen(m_package, "mu_start: " << mu_star);
	//printInPackageOnScreen(m_package, "sigma_star: " << sigma_star);
	double contentOfSig = 0;
	if(1.0 / (m_lambda * m_lambda) + sigma_star < 0){
		contentOfSig = mu_star;
	}else{
		contentOfSig = (mu_star / sqrt(1.0 / (m_lambda * m_lambda) + sigma_star));
	}
	return boost::math::erfc(-contentOfSig / SQRT2) / 2.0;
}

double IVM::predictMu(const Vector& input) const{
	const unsigned int n = m_I.size();
	Vector k_star(n);
	unsigned int i = 0;
	if(m_kernelType == GAUSS){
		for(List<unsigned int>::const_iterator itOfI = m_I.begin(); itOfI != m_I.end(); ++itOfI, ++i){
			k_star.coeffRef(i) = m_gaussKernel->kernelFuncVec(input, *m_storage[*itOfI]);
		}
	}else if(m_kernelType == RF){
		for(List<unsigned int>::const_iterator itOfI = m_I.begin(); itOfI != m_I.end(); ++itOfI, ++i){
			k_star.coeffRef(i) = m_rfKernel->kernelFuncVec(input, *m_storage[*itOfI]);
		}
	}
	const Vector v = m_choleskyLLT.solve(k_star);
	/*
		const Vector mu_tilde = m_nuTilde.cwiseQuotient(m_tauTilde);
		double mu_star = (mu_tilde + (m_bias * Vector::Ones(n))).dot(v);*/
	return m_muTildePlusBias.dot(v);
}

double IVM::predictSigma(const Vector& input) const{
	const unsigned int n = m_I.size();
	Vector k_star(n);
	unsigned int i = 0;
	double diagEle = 0;
	if(m_kernelType == GAUSS){
		for(List<unsigned int>::const_iterator itOfI = m_I.begin(); itOfI != m_I.end(); ++itOfI, ++i){
			k_star.coeffRef(i) = m_gaussKernel->kernelFuncVec(input, *m_storage[*itOfI]);
		}
		diagEle = m_gaussKernel->calcDiagElement(0);
	}else if(m_kernelType == RF){
		for(List<unsigned int>::const_iterator itOfI = m_I.begin(); itOfI != m_I.end(); ++itOfI, ++i){
			k_star.coeffRef(i) = m_rfKernel->kernelFuncVec(input, *m_storage[*itOfI]);
		}
		diagEle = m_rfKernel->calcDiagElement(0);
	}
	const Vector v = m_choleskyLLT.solve(k_star);
	/*
		const Vector mu_tilde = m_nuTilde.cwiseQuotient(m_tauTilde);
		double mu_star = (mu_tilde + (m_bias * Vector::Ones(n))).dot(v);*/
	return (diagEle - k_star.dot(v));
}

unsigned int IVM::getLabelForOne() const{
	return (unsigned int) m_labelsForClasses.coeff(0);
}

unsigned int IVM::getLabelForMinusOne() const{
	return (unsigned int) m_labelsForClasses.coeff(1);
}

void IVM::setKernelSeed(unsigned int seed){
	if(m_kernelType == GAUSS){
		m_gaussKernel->setSeed(seed);
	}else if(m_kernelType == RF){
		m_rfKernel->setSeed(seed);
	}
}
