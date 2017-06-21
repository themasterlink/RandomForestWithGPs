/*
 * DataWriterForVisu.cc
 *
 *  Created on: 04.06.2016
 *      Author: Max
 */

#include "DataWriterForVisu.h"
#include "../Utility/ColorConverter.h"
#include "DataConverter.h"
#include <opencv2/core/core.hpp>
#include <math.h>
#include "../Base/CommandSettings.h"

DataWriterForVisu::DataWriterForVisu()
{
	// TODO Auto-generated constructor stub

}

DataWriterForVisu::~DataWriterForVisu(){
}

void DataWriterForVisu::writeData(const std::string& fileName, const LabeledData& data,
		const int x, const int y){
	if(data.size() > 0){
		if(!(data[0]->rows() > x && data[0]->rows() > y && x != y && y >= 0 && x >= 0)){
			printError("These axis x: " << x << ", y: " << y << " aren't printable!");
			return;
		}
		std::ofstream file;
		file.open(Logger::getActDirectory() + fileName);
		if(file.is_open()){
			for(LabeledDataConstIterator it = data.cbegin(); it != data.cend(); ++it){
				file << (**it)[x] << " " << (**it)[y] << " " << (*it)->getLabel() << "\n";
			}
		}
		file.close();
	}else{
		printError("No data -> no writting!");
	}
}

void DataWriterForVisu::generateGrid(const std::string& fileName,
		const PredictorMultiClass* predictor,
		const Real amountOfPointsOnOneAxis,
		const LabeledData& data,
		const int x, const int y){
	if(data.size() == 0){
		printError("No data is given, this data is needed to find min and max!");
		return;
	}
	const int dim = (int) data[0]->rows();
	Vector2i dimVec;
	dimVec << x,y;
	Vector2 min, max;
	DataConverter::getMinMaxIn2D(data, min, max, dimVec);
	Vector2 stepSize = (1. / amountOfPointsOnOneAxis) * (max - min);
	std::ofstream file;
	file.open(fileName);
	Data points;
	points.reserve(amountOfPointsOnOneAxis * (amountOfPointsOnOneAxis + 1));
	int amount = 0;
	for(Real xVal = max[0]; xVal >= min[0]; xVal -= stepSize[0]){
		for(Real yVal = min[1]; yVal < max[1]; yVal+= stepSize[1]){
			VectorX ele(dim);
			for(int i = 0; i < dim; ++i){
				if(i == x){
					ele[i] = xVal;
				}else if(i == y){
					ele[i] = yVal;
				}else{
					ele[i] = 0;
				}
			}
			file << xVal << " " << yVal << " " << predictor->predict(ele) << "\n";
			++amount;
		}
	}
	file.close();
}

void DataWriterForVisu::generateGrid(const std::string& fileName,
		const PredictorBinaryClass* predictor,
		const Real amountOfPointsOnOneAxis,
		const LabeledData& data,
		const int x, const int y){
	if(data.size() == 0){
		printError("No data is given, this data is needed to find min and max!");
		return;
	}
	const int dim = data[0]->rows();
	Vector2i dimVec;
	dimVec << x,y;
	Vector2 min, max;
	DataConverter::getMinMaxIn2D(data, min, max, dimVec);
	Vector2 stepSize = (1. / amountOfPointsOnOneAxis) * (max - min);
	std::ofstream file;
	file.open(Logger::getActDirectory() + fileName);
	Data points;
	points.reserve(amountOfPointsOnOneAxis * (amountOfPointsOnOneAxis + 1));
	int amount = 0;
	for(Real xVal = max[0]; xVal >= min[0]; xVal -= stepSize[0]){
		for(Real yVal = min[1]; yVal < max[1]; yVal+= stepSize[1]){
			VectorX ele(dim);
			for(int i = 0; i < dim; ++i){
				if(i == x){
					ele[i] = xVal;
				}else if(i == y){
					ele[i] = yVal;
				}else{
					ele[i] = 0;
				}
			}
			file << xVal << " " << yVal << " " << predictor->predict(ele) << "\n";
			++amount;
		}
	}
	file.close();
}

void DataWriterForVisu::writeSvg(const std::string& fileName, const PredictorBinaryClass* predictor, const LabeledData& data, const int x, const int y){
	if(data.size() == 0){
		printError("No data is given, this data is needed to find min and max!");
		return;
	}
	const int dim = data[0]->rows();
	Vector2i dimVec;
	dimVec << x,y;
	Vector2 min, max;
	DataConverter::getMinMaxIn2D(data, min, max, dimVec);
	const Vector2 diff = max - min;
	max[0] += diff[0] * 0.2;
	max[1] += diff[1] * 0.2;
	min[0] -= diff[0] * 0.2;
	min[1] -= diff[1] * 0.2;
	const Real amountOfPointsOnOneAxis = std::max(CommandSettings::get_visuRes(), CommandSettings::get_visuResSimple());
	Vector2 stepSize = (1. / amountOfPointsOnOneAxis) * (max - min);
	int amount = 0;
	std::vector<Real> labels;
	const Real elementInX = (int)((max[0] - min[0]) / stepSize[0]);
	const Real elementInY = (int)((max[1] - min[1]) / stepSize[1]);
	int iX = 0, iY;
	std::ofstream file;
	openSvgFile(fileName, (Real) 820., (Real) (max[0] - min[0]), (Real) (max[1] - min[1]), file);
	const bool complex = CommandSettings::get_visuRes() > 0;
	for(Real xVal = min[0]; xVal < max[0]; xVal += stepSize[0]){
		iY = 0;
		for(Real yVal = min[1]; yVal < max[1]; yVal+= stepSize[1]){
			VectorX ele(dim);
			for(int i = 0; i < dim; ++i){
				if(i == x){
					ele[i] = xVal;
				}else if(i == y){
					ele[i] = yVal;
				}else{
					ele[i] = 0;
				}
			}
			//gp.resetFastPredict();
			Real val = std::max(std::min((Real) 1.0, (Real) predictor->predict(ele)), (Real) 0.0);
			if(!complex){
				val = val >= (Real) 0.5 ? (Real) 1 : (Real) 0.;
			}
			Real r, g, b;
			ColorConverter::getProbColorForBinaryRGB(val, r, g, b);
			drawSvgRect(file, iX / elementInX * (Real) 100., iY / elementInY  * (Real) 100.,
						(Real) 100.0 / elementInX, (Real) 100.0 / elementInY, r * (Real) 100, g * (Real) 100, b * (Real) 100);
			++amount;
			++iY;
		}
		++iX;
	}
	std::list<unsigned int> empty;
	drawSvgDataPoints(file, data, min, max, dimVec, empty);
	closeSvgFile(file);
}

void DataWriterForVisu::writeSvg(const std::string& fileName, const PredictorMultiClass* predictor, const LabeledData& data, const int x, const int y){
	if(data.size() == 0){
		printError("No data is given, this data is needed to find min and max!");
		return;
	}
	const auto dim = (int) data[0]->rows();
	Vector2i dimVec;
	dimVec << x,y;
	Vector2 min, max;
	DataConverter::getMinMaxIn2D(data, min, max, dimVec);
	const Vector2 diff = max - min;
	max[0] += diff[0] * 0.2;
	max[1] += diff[1] * 0.2;
	min[0] -= diff[0] * 0.2;
	min[1] -= diff[1] * 0.2;
	const Real amountOfPointsOnOneAxis = std::max(CommandSettings::get_visuRes(), CommandSettings::get_visuResSimple());
	Vector2 stepSize = (1. / amountOfPointsOnOneAxis) * (max - min);
	int amount = 0;
	std::vector<Real> labels;
	const Real elementInX = (int)((max[0] - min[0]) / stepSize[0]);
	const Real elementInY = (int)((max[1] - min[1]) / stepSize[1]);
	std::ofstream file;
	const unsigned int classAmount = predictor->amountOfClasses();
	openSvgFile(fileName, 820., (Real) (max[0] - min[0]), (Real) (max[1] - min[1]), file);
	Data points;
	for(Real xVal = min[0]; xVal < max[0]; xVal += stepSize[0]){
	//for(Real xVal = max[0]; xVal >= min[0]; xVal -= stepSize[0]){
		for(Real yVal = min[1]; yVal < max[1]; yVal+= stepSize[1]){
			VectorX* ele = new VectorX(dim);
			for(int i = 0; i < dim; ++i){
				if(i == x){
					(*ele)[i] = xVal;
				}else if(i == y){
					(*ele)[i] = yVal;
				}else{
					(*ele)[i] = 0;
				}
			}
			points.push_back(ele);
			++amount;
		}
	}
	const bool complex = CommandSettings::get_visuRes() > 0;
	Labels result;
	std::vector< std::vector<Real> > probs;
	if(complex){
		predictor->predictData(points, result, probs);
	}else{
		predictor->predictData(points, result);
	}
	int counter = 0;
	int iX = 0, iY;
	std::vector< std::vector<Real> > colors(classAmount, std::vector<Real>(3));
	for(unsigned int i = 0; i < classAmount; ++i){ // save all colors
		ColorConverter::HSV2LAB((i + 1) / (Real) (classAmount) * 360., 1.0, 1.0, colors[i][0], colors[i][1], colors[i][2]);
	}
	for(Real xVal = min[0]; xVal < max[0]; xVal += stepSize[0]){
		iY = 0;
		for(Real yVal = min[1]; yVal < max[1]; yVal+= stepSize[1]){
				//Real val = fmax(fmin(1.0,), 0.0);
			Real r, g, b_;
			if(complex){
				Real l = 0, a = 0, b = 0;
				Real sum = 0;
				for(unsigned int i = 0; i < classAmount; ++i){
					sum += probs[counter][i];
				}
				if(sum > 0){
					for(unsigned int i = 0; i < classAmount; ++i){
						l += probs[counter][i] / sum * colors[i][0];
						a += probs[counter][i] / sum * colors[i][1];
						b += probs[counter][i] / sum * colors[i][2];
					}
				}
				ColorConverter::LAB2RGB(l, a, b, r, g, b_);
			}else{
				Real l = 0, a = 0, b = 0;
				l = colors[result[counter]][0];
				a = colors[result[counter]][1];
				b = colors[result[counter]][2];
				ColorConverter::LAB2RGB(l, a, b, r, g, b_);
			}
			drawSvgRect(file, iX / elementInX * 100., iY / elementInY  * 100.,
					100.0 / elementInX, 100.0 / elementInY, r * 100., g * 100., b_ * 100.);
			++counter;
			++iY;
		}
		++iX;
	}
	for(unsigned int i = 0; i < points.size(); ++i){
		SAVE_DELETE(points[i]);
	}
	std::list<unsigned int> empty;
	drawSvgDataPoints(file, data, min, max, dimVec, empty, classAmount);
	closeSvgFile(file);
}

void DataWriterForVisu::writeHisto(const std::string&fileName, const std::list<Real>& list, const unsigned int nrOfBins, const Real minValue, const Real maxValue){
	if(list.size() == 0){
		printError("No data is given!");
		return;
	}
	Real min, max;
	if(minValue == maxValue && minValue == -1){
		DataConverter::getMinMax(list, min, max);
	}else{
		min = minValue;
		max = maxValue;
	}
	if(min == max){
		printError("The min and max value are the same, the calculation of the histogram is not possible!"); return;
	}
	VectorX counter = VectorX::Zero(nrOfBins);
	for(auto it = list.begin(); it != list.end(); ++it){
		++counter[(*it - min) / (max - min) * (nrOfBins - 1)];
	}
	Real minCounter, maxCounter;
	DataConverter::getMinMax(counter, minCounter, maxCounter);
	std::ofstream file;
	openSvgFile(fileName, 820., 1., 1., file);
	const Real startOfData = 7.5;
	drawSvgCoords(file, 5., 5., 7.5, 7.5, nrOfBins + 1, maxCounter, 0, 100., 820, 820, true, 0., 1.);
	const Real dataWidth = (100.0 - 2. * startOfData);
	const Real width = (1. / (Real) nrOfBins *  (dataWidth * 0.95));
	const Real offset = (1. / (Real) nrOfBins * (dataWidth * 0.05));
	for(unsigned int i = 0; i < nrOfBins; ++i){
		const Real height = counter[i] / (Real) maxCounter * dataWidth;
		drawSvgRect(file, i / (Real) nrOfBins * dataWidth + offset + startOfData, startOfData, width, height == 0. ? 1. : height, 76, 5, 78);
	}
	closeSvgFile(file);
}

void DataWriterForVisu::writeSvg(const std::string& fileName, const IVM& ivm, const std::list<unsigned int>& selectedInducingPoints,
		const LabeledData& data, const int x, const int y, const int type){
	if(data.size() == 0){
		printError("No data is given, this data is needed to find min and max!");
		return;
	}
	const int dim = data[0]->rows();
	Vector2i dimVec;
	dimVec << x,y;
	Vector2 min, max;
	DataConverter::getMinMaxIn2D(data, min, max, dimVec);
	const Vector2 diff = max - min;
	if(diff[0] <= EPSILON || diff[1] <= EPSILON){
		printError("The min and max of the desired axis is equal for " << fileName << "!"); return;
	}
	max[0] += diff[0] * 0.2;
	max[1] += diff[1] * 0.2;
	min[0] -= diff[0] * 0.2;
	min[1] -= diff[1] * 0.2;
	const Real amountOfPointsOnOneAxis = std::max(CommandSettings::get_visuRes(), CommandSettings::get_visuResSimple());
	Vector2 stepSize = (1. / amountOfPointsOnOneAxis) * (max - min);
	int amount = 0;
	const Real elementInX = (int)((max[0] - min[0]) / stepSize[0]);
	const Real elementInY = (int)((max[1] - min[1]) / stepSize[1]);
	int iX = 0, iY;
	std::ofstream file;
	openSvgFile(fileName, 820., (Real) (max[0] - min[0]), (Real) (max[1] - min[1]), file);
	const bool complex = CommandSettings::get_visuRes() > 0;
	std::vector< std::vector<Real> > colors(2, std::vector<Real>(3));
	ColorConverter::RGB2LAB(1,0,0, colors[0][0], colors[0][1], colors[0][2]);
	ColorConverter::RGB2LAB(0,0,1, colors[1][0], colors[1][1], colors[1][2]);
	Real minMu = REAL_MAX, minSigma = REAL_MAX, maxMu = NEG_REAL_MAX, maxSigma = NEG_REAL_MAX;
	if(type == 1 || type == 2){
		for(Real xVal = min[0]; xVal < max[0]; xVal += stepSize[0]){
			for(Real yVal = min[1]; yVal < max[1]; yVal+= stepSize[1]){
				VectorX ele(dim);
				for(int i = 0; i < dim; ++i){
					if(i == x){
						ele[i] = xVal;
					}else if(i == y){
						ele[i] = yVal;
					}else{
						ele[i] = 0;
					}
				}
				Real prob = 0;
				if(type == 1){
					prob = ivm.predictMu(ele);
					if(prob < minMu){
						minMu = prob;
					}
					if(prob > maxMu){
						maxMu = prob;
					}
				}else if(type == 2){
					prob = ivm.predictSigma(ele);
					if(prob < minSigma){
						minSigma = prob;
					}
					if(prob > maxSigma){
						maxSigma = prob;
					}
				}
			}
		}
	}
	for(Real xVal = min[0]; xVal < max[0]; xVal += stepSize[0]){
	//for(Real xVal = max[0]; xVal >= min[0]; xVal -= stepSize[0]){
		iY = 0;
		for(Real yVal = min[1]; yVal < max[1]; yVal+= stepSize[1]){
			VectorX ele(dim);
			for(int i = 0; i < dim; ++i){
				if(i == x){
					ele[i] = xVal;
				}else if(i == y){
					ele[i] = yVal;
				}else{
					ele[i] = 0;
				}
			}
			Real prob = 0;
			if(type == 0){
				prob = ivm.predict(ele);
			}else if(type == 1){
				prob = (ivm.predictMu(ele) - minMu) / (maxMu - minMu);
			}else if(type == 2){
				prob = (ivm.predictSigma(ele) - minSigma) / (maxSigma - minSigma);
			}
			if(!complex){
				prob = prob > 0.5 ? 1 : 0;
			}
			Real l = prob * colors[0][0] + (1-prob) * colors[1][0];
			Real a = prob * colors[0][1] + (1-prob) * colors[1][1];
			Real b = prob * colors[0][2] + (1-prob) * colors[1][2];
			Real r, g, b_;
			ColorConverter::LAB2RGB(l, a, b, r, g, b_);

			//Real val = fmax(fmin(1.0,), 0.0);
			drawSvgRect(file, iX / elementInX * 100., iY / elementInY  * 100.,
					100.0 / elementInX, 100.0 / elementInY, r* 100., g* 100., b_ * 100.);
			++amount;
			++iY;
		}
		++iX;
	}
	drawSvgDataPoints(file, data, min, max, dimVec, selectedInducingPoints, 2, &ivm);
	closeSvgFile(file);
}

void DataWriterForVisu::writeSvg(const std::string& fileName, const LabeledData& data, const int x, const int y){
	Vector2i dimVec;
	dimVec << x,y;
	Vector2 min, max;
	DataConverter::getMinMaxIn2D(data, min, max, dimVec);
	const Vector2 diff = max - min;
	max[0] += diff[0] * 0.2;
	max[1] += diff[1] * 0.2;
	min[0] -= diff[0] * 0.2;
	min[1] -= diff[1] * 0.2;
	std::list<unsigned int> empty;
	std::ofstream file;
	std::map<unsigned int, unsigned int> classCounter;
	for(const auto& pos : data){
//	for(LabeledData::const_iterator it = data.begin(); it != data.end(); ++it){
		auto itClass = classCounter.find(pos->getLabel());
		if(itClass == classCounter.end()){
			classCounter.emplace(pos->getLabel(), 0);
		}
	}
	openSvgFile(fileName, 820., (Real) (max[0] - min[0]), (Real) (max[1] - min[1]), file);
	drawSvgDataPoints(file, data, min, max, dimVec, empty, classCounter.size());
	closeSvgFile(file);
}

void DataWriterForVisu::writePointsIn2D(const std::string& fileName, const std::list<Vector2>& points, const std::list<Real>& values){
	if(points.size() != values.size()){
		printError("The size of the values and the points does not match!");
	}else if(points.size() < 3){
		printError("There must be at least 3 points");
	}
	Vector2 min, max;
	DataConverter::getMinMaxIn2D(points, min, max);
	Real minVal, maxVal;
	DataConverter::getMinMax(values, minVal, maxVal, true);
	const Vector2 diff = max - min;
	max[0] += diff[0] * 0.2;
	max[1] += diff[1] * 0.2;
	min[0] -= diff[0] * 0.2;
	min[1] -= diff[1] * 0.2;
	std::list<int> empty;
	std::ofstream file;
	openSvgFile(fileName, 820., 1., 1.,  file);
	const Real height = 820;
	const int amountOfSeg = 10;
	drawSvgCoords2D(file, 7.5, 7.5, 10, 10, min, max, amountOfSeg, 820., height);
	Real minUsedValue = minVal;
	const int k = CommandSettings::get_visuRes() > 0 ? 20 : CommandSettings::get_visuResSimple() > 0 ? 1 : 0;
	const Real amountOfPointsOnOneAxis = std::max(CommandSettings::get_visuRes(), CommandSettings::get_visuResSimple());
	const Vector2 stepSize = (1. / amountOfPointsOnOneAxis) * (max - min);
	Real red[3];
	Real blue[3];
	ColorConverter::HSV2LAB(180., 1, 1, red[0], red[1], red[2]);
	ColorConverter::HSV2LAB(0., 1, 1, blue[0], blue[1], blue[2]);
	if(points.size() > 10){
		std::list<Real> sortedValues;
		Real mean = 0;
		for(auto it = values.cbegin(); it != values.cend(); ++it){
			if(*it > NEG_REAL_MAX){
				mean += *it;
			}
			bool found = false;
			for(auto itSort = sortedValues.cbegin(); itSort != sortedValues.cend() && !found; ++itSort){
				if(*itSort > *it){
					sortedValues.insert(itSort, *it);
					found = true;
				}
			}
			if(!found){
				sortedValues.push_back(*it);
			}
		}
		mean /= values.size();
		for(auto itSort = sortedValues.cbegin(); itSort != sortedValues.cend(); ++itSort){
			if(*itSort > mean - (maxVal - mean) && *itSort > NEG_REAL_MAX){
				minUsedValue = *itSort;
				break;
			}
		}
	}
	const Real elementInX = (int)((max[0] - min[0]) / stepSize[0]);
	const Real elementInY = (int)((max[1] - min[1]) / stepSize[1]);
	std::list<std::pair<Real, std::pair<Real, Real> > > imgPixel;
	for(Real dX = min[0]; dX < max[0]; dX += stepSize[0]){
		for(Real dY = min[1]; dY < max[1]; dY += stepSize[1]){
			std::list<std::pair<Real, Real> > closestsPoints;
			const Real newDX = (Real) (dX + stepSize[0] * 0.5);
			const Real newDY = (Real) (dY + stepSize[1] * 0.5);
			auto itValue = values.cbegin();
			for(auto it = points.cbegin(); it != points.cend(); ++it, ++itValue){
				const Real dist = ((*it).coeff(0) - newDX) * ((*it).coeff(0) - newDX) + ((*it).coeff(1) - newDY) * ((*it).coeff(1) - newDY);
				bool addNew = false;
				for(auto it = closestsPoints.begin(); it != closestsPoints.end(); ++it){
					if(dist < it->second){
						closestsPoints.emplace(it, *itValue, dist);
						addNew = true;
						break;
					}
				}
				if(!addNew && (int) closestsPoints.size() < k){
					closestsPoints.emplace_back(*itValue, dist);
				}else if((int) closestsPoints.size() > k){
					closestsPoints.pop_back();
				}
			}
			Real val = 0;
			Real totalDist = 0;
			for(const auto& point : closestsPoints){
				totalDist += point.second;
			}
			for(const auto& point : closestsPoints){
				if(point.first > NEG_REAL_MAX){
					val += point.first * (point.second / totalDist);
				}else{
					val = NEG_REAL_MAX;
					break;
				}
			}
			const Real xPos = (Real) ((dX - min[0]) / (max[0] - min[0]) * 80. + 10.); // see 10. in drawSvgCoords2D
			const Real yPos = (Real) ((dY - min[1]) / (max[1] - min[1]) * 80. + 10.);
			imgPixel.push_back(std::pair<Real, std::pair<Real, Real> >(val,  std::pair<Real, Real>(xPos, yPos)));
		}
	}
	Real newMinVal = REAL_MAX;
	Real newMaxVal = -REAL_MAX;
	for(const auto& pix : imgPixel){
		if(pix.first < newMinVal){
			newMinVal = pix.first;
		}
		if(pix.first > newMaxVal){
			newMaxVal = pix.first;
		}
	}
	for(const auto& pix : imgPixel){
		Real val = pix.first;
		const Real xPos = pix.second.first;
		const Real yPos = pix.second.second;
		if(val >= minVal){
			if(val > minUsedValue){
				val = (val - newMinVal) / (newMaxVal - newMinVal);
			}else{
				val = (Real) 0.;
			}
		}
		Real r,g,b;
		ColorConverter::getProbColorForBinaryRGB(val, r,g,b);
		drawSvgRect(file, xPos, yPos, 100. / elementInX, 100. / elementInY, r * 100,g * 100,b * 100);
	}
	auto itValues = values.cbegin();

	for(auto it = points.cbegin(); it != points.cend(); ++it, ++itValues){
		const Real dx = (((*it)[0] - min[0]) / (max[0] - min[0]) * 80. + 10.) / 100. * 820.; // see 10. in drawSvgCoords2D
		const Real dy = (((*it)[1] - min[1]) / (max[1] - min[1]) * 80. + 10.) / 100. * height;
		Real r = 0, g = 0, b = 0;
		//std::cout << "(*it)->getLabel(): " << (*it)->getLabel() + 1 << " " << amountOfClasses << ",";
		if(*itValues > NEG_REAL_MAX){ // ignores NEG_REAL_MAX values make them black
			Real val;
			if(*itValues > minUsedValue){
				val = (*itValues - minUsedValue) / (maxVal - minUsedValue);
			}else{
				val = 0.;
			}
			ColorConverter::getProbColorForBinaryRGB(val, r,g,b);
		}
		file << "<circle cx=\"" << dx << "\" cy=\"" << dy << "\" r=\"8\" fill=\"rgb(" << r * 100. << "%," << g * 100. << "%," << b * 100. << "%)\" stroke=\"black\" stroke-width=\"2\"/> \n";
	}
	closeSvgFile(file);
}

void DataWriterForVisu::writeImg(const std::string& fileName, const PredictorBinaryClass* predictor,
		const LabeledData& data, const int x, const int y){
	if(data.size() == 0){
		printError("No data is given, this data is needed to find min and max!");
		return;
	}
	const int dim = data[0]->rows();
	Vector2i dimVec;
	dimVec << x,y;
	Vector2 min, max;
	DataConverter::getMinMaxIn2D(data, min, max, dimVec);
	const Vector2 diff = max - min;
	max[0] += diff[0] * 0.2;
	max[1] += diff[1] * 0.2;
	min[0] -= diff[0] * 0.2;
	min[1] -= diff[1] * 0.2;

	const Real amountOfPointsOnOneAxis = std::max(CommandSettings::get_visuRes(), CommandSettings::get_visuResSimple());
	Vector2 stepSize = (1. / amountOfPointsOnOneAxis) * (max - min);
//	stepSize[1] *= 0.5;
	const Real elementInX = ceil((Real)((max[0] - min[0]) / stepSize[0])) + 1;
	const Real elementInY = ceil((Real)((max[1] - min[1]) / stepSize[1])) + 1;
	const bool complex = CommandSettings::get_visuRes() > 0;
	int counter = 0;
	std::vector< std::vector<Real> > colors(2, std::vector<Real>(3));
	const bool modeLab = false;
	if(modeLab){
		ColorConverter::RGB2LAB(1,0,0, colors[0][0], colors[0][1], colors[0][2]);
		ColorConverter::RGB2LAB(0,1,1, colors[1][0], colors[1][1], colors[1][2]);
	}
	unsigned int fac = 4;
	cv::Mat img(elementInY * fac, elementInX * fac, CV_8UC3, cv::Scalar(0, 0, 0));
	int iX = 0, iY;
	if(predictor != nullptr){
		for(Real xVal = min[0]; xVal < max[0]; xVal += stepSize[0]){
			if(iX >= elementInX){
				printError("This should not happen for x: " << iX);
				continue;
			}
			iY = 0;
			for(Real yVal = min[1]; yVal < max[1]; yVal += stepSize[1]){
				if(iY >= elementInY){
					printError("This should not happen for y: " << iY);
					continue;
				}
				VectorX* ele = new VectorX(dim);
				for(int i = 0; i < dim; ++i){
					if(i == x){
						(*ele)[i] = xVal;
					}else if(i == y){
						(*ele)[i] = yVal;
					}else{
						(*ele)[i] = 0;
					}
				}
				Real prob = predictor->predict(*ele);
				delete ele;
				//Real val = fmax(fmin(1.0,), 0.0);
				if(!complex){
					prob = prob > 0.5 ? 1 : 0;
				}
				Real r, g, b_;
				if(modeLab){
					Real l = prob * colors[0][0] + (1-prob) * colors[1][0];
					Real a = prob * colors[0][1] + (1-prob) * colors[1][1];
					Real b = prob * colors[0][2] + (1-prob) * colors[1][2];
					ColorConverter::LAB2RGB(l, a, b, r, g, b_);
				}else{
					ColorConverter::getProbColorForBinaryRGB(prob, r, g, b_);
				}
				const unsigned int xInPic = iX * fac;
				const unsigned int yInPic = (elementInY - iY - 1) * fac;
				for(unsigned int t = 0; t < fac; ++t){
					for(unsigned int l = 0; l < fac; ++l){
						cv::Vec3b& color = img.at<cv::Vec3b>(yInPic + t, xInPic + l);
						color[0] = b_ * 255; // from rgb to bgr
						color[1] = g * 255;
						color[2] = r * 255;
					}
				}
				++counter;
				++iY;
			}
			++iX;
		}
	}else{
		cv::Scalar white(255,255,255);
		img.setTo(white);
	}
	cv::Scalar black(0,0,0);
	unsigned int oldFac = fac;
	fac = 3;
	for(LabeledDataConstIterator it = data.cbegin(); it != data.cend(); ++it, ++counter){
		const int dx = ((**it)[x] - min[0]) / (max[0] - min[0]) * elementInX * oldFac;
		const int dy = (1. - ((**it)[y] - min[1]) / (max[1] - min[1])) * elementInY * oldFac;
		const int label = (*it)->getLabel();
		Real r, g, b;
		if(modeLab){
			ColorConverter::LAB2RGB(colors[label][0], colors[label][1], colors[label][2], r, g, b);
		}else{
			ColorConverter::getProbColorForBinaryRGB((Real) (label == 1 ? 0.0 : 1.0), r, g, b);
		}
		cv::Scalar actColor(b * 255,g * 255,r * 255);// from rgb to bgr
		cv::circle(img, cv::Point(dx, dy), (Real) fac * amountOfPointsOnOneAxis / 50.0, actColor, CV_FILLED, CV_AA);
		cv::circle(img, cv::Point(dx, dy), (Real) fac * amountOfPointsOnOneAxis / 50.0, black, (Real) fac / 4.0 * (Real) amountOfPointsOnOneAxis / 50.0, CV_AA);
	}
	cv::imwrite(Logger::getActDirectory() + fileName, img);
}

void DataWriterForVisu::writeImg(const std::string& fileName, const IVM* predictor,
		const LabeledData& data, const int x, const int y){
	if(data.size() == 0){
		printError("No data is given, this data is needed to find min and max!");
		return;
	}
	const int dim = data[0]->rows();
	Vector2i dimVec;
	dimVec << x,y;
	Vector2 min, max;
	DataConverter::getMinMaxIn2D(data, min, max, dimVec);
	const Vector2 diff = max - min;
	max[0] += diff[0] * 0.2;
	max[1] += diff[1] * 0.2;
	min[0] -= diff[0] * 0.2;
	min[1] -= diff[1] * 0.2;

	const Real amountOfPointsOnOneAxis = std::max(CommandSettings::get_visuRes(), CommandSettings::get_visuResSimple());
	Vector2 stepSize = (1. / amountOfPointsOnOneAxis) * (max - min);
//	stepSize[1] *= 0.5;
	const Real elementInX = ceil((Real)((max[0] - min[0]) / stepSize[0])) + 1;
	const Real elementInY = ceil((Real)((max[1] - min[1]) / stepSize[1])) + 1;
	const bool complex = CommandSettings::get_visuRes() > 0;
	int counter = 0;
	std::vector< std::vector<Real> > colors(4, std::vector<Real>(3));
	const bool modeLab = false;
	if(modeLab){
		ColorConverter::RGB2LAB(1,0,0, colors[0][0], colors[0][1], colors[0][2]);
		ColorConverter::RGB2LAB(0,1,1, colors[1][0], colors[1][1], colors[1][2]);
		ColorConverter::RGB2LAB(0,0,1, colors[2][0], colors[2][1], colors[2][2]);
		ColorConverter::RGB2LAB(1,1,0, colors[3][0], colors[3][1], colors[3][2]);
	}
	unsigned int fac = 4;
	cv::Mat img(elementInX * fac, elementInY * fac, CV_8UC3, cv::Scalar(0, 0, 0));
	int iX = 0, iY;
	cv::Scalar white(255,255,255);
	if(predictor != nullptr){
		for(Real xVal = min[0]; xVal < max[0]; xVal += stepSize[0]){
			if(iX >= elementInX){
				printError("This should not happen for x: " << iX);
				continue;
			}
			iY = 0;
			for(Real yVal = min[1]; yVal < max[1]; yVal += stepSize[1]){
				if(iY >= elementInY){
					printError("This should not happen for y: " << iY);
					continue;
				}
				VectorX* ele = new VectorX(dim);
				for(int i = 0; i < dim; ++i){
					if(i == x){
						(*ele)[i] = xVal;
					}else if(i == y){
						(*ele)[i] = yVal;
					}else{
						(*ele)[i] = 0;
					}
				}
				Real prob = predictor->predict(*ele);
				delete ele;
				//Real val = fmax(fmin(1.0,), 0.0);
				if(!complex){
					prob = prob > 0.5 ? 1 : 0;
				}
				Real r, g, b_;
				if(modeLab){
					Real l = prob * colors[0][0] + (1-prob) * colors[1][0];
					Real a = prob * colors[0][1] + (1-prob) * colors[1][1];
					Real b = prob * colors[0][2] + (1-prob) * colors[1][2];
					ColorConverter::LAB2RGB(l, a, b, r, g, b_);
				}else{
					ColorConverter::getProbColorForBinaryRGB(prob, r, g, b_);
				}
				const unsigned int xInPic = iX * fac;
				const unsigned int yInPic = (elementInY - iY - 1) * fac;
				for(unsigned int t = 0; t < fac; ++t){
					for(unsigned int l = 0; l < fac; ++l){
						cv::Vec3b& color = img.at<cv::Vec3b>(xInPic + l, yInPic + t);
						color[0] = b_ * 255; // from rgb to bgr
						color[1] = g * 255;
						color[2] = r * 255;
					}
				}
				++counter;
				++iY;
			}
			++iX;
		}
	}else{
		img.setTo(white);
	}
	cv::Scalar black(0,0,0);
	unsigned int oldFac = fac;
	fac = 3;
	for(LabeledDataConstIterator it = data.cbegin(); it != data.cend(); ++it, ++counter){
		const int dx = ((**it)[x] - min[0]) / (max[0] - min[0]) * elementInX * oldFac;
		const int dy = (1. - ((**it)[y] - min[1]) / (max[1] - min[1])) * elementInY * oldFac;
		const int label = (*it)->getLabel();
		Real r, g, b;
		if(modeLab){
			ColorConverter::LAB2RGB(colors[label][0], colors[label][1], colors[label][2], r, g, b);
		}else{
			ColorConverter::getProbColorForBinaryRGB(label == 1 ? 0.0 : 1.0, r, g, b);
		}
		cv::Scalar actColor(b * 255,g * 255,r * 255);// from rgb to bgr
		cv::circle(img, cv::Point(dy, dx), (Real) fac * amountOfPointsOnOneAxis / 50.0, actColor, CV_FILLED, CV_AA);
		cv::circle(img, cv::Point(dy, dx), (Real) fac * amountOfPointsOnOneAxis / 50.0, black, (Real) fac / 4.0 * (Real) amountOfPointsOnOneAxis / 50.0, CV_AA);
	}
	for(const auto& indx : predictor->getSelectedInducingPoints()){
		LabeledVectorX* it =  data[indx];
		const int dx = (it->coeff(x) - min[0]) / (max[0] - min[0]) * elementInX * oldFac;
		const int dy = (1. - (it->coeff(y) - min[1]) / (max[1] - min[1])) * elementInY * oldFac;
		const int label = it->getLabel();
		Real r, g, b;
		Real ir, ig, ib;
		if(modeLab){
			ColorConverter::LAB2RGB(colors[label][0], colors[label][1], colors[label][2], r, g, b);
		}else{
			ColorConverter::getProbColorForBinaryRGB(label == 1 ? 0.0 : 1.0, r, g, b);
			ColorConverter::getProbColorForBinaryRGB(label == 0 ? 0.0 : 1.0, ir, ig, ib);
		}
		cv::Scalar actColor(b * 255,g * 255,r * 255);// from rgb to bgr
		cv::Scalar induActColor(ib * 255,ig * 255,ir * 255);// from rgb to bgr
		cv::circle(img, cv::Point(dy, dx), (Real) fac * amountOfPointsOnOneAxis / 50.0, actColor, CV_FILLED, CV_AA);
		cv::circle(img, cv::Point(dy, dx), (Real) fac * amountOfPointsOnOneAxis / 50.0, black, (Real) fac / 4.0 * (Real) amountOfPointsOnOneAxis / 50.0, CV_AA);
		cv::circle(img, cv::Point(dy, dx), (Real) fac * amountOfPointsOnOneAxis / 200.0, induActColor, CV_FILLED, CV_AA);
	}
	cv::imwrite(Logger::getActDirectory() + fileName, img);
}


void DataWriterForVisu::writeImg(const std::string& fileName, const PredictorMultiClass* predictor, const LabeledData& data, const int x, const int y){
	if(data.size() == 0){
		printError("No data is given, this data is needed to find min and max!");
		return;
	}
	const int dim = data[0]->rows();
	Vector2i dimVec;
	dimVec << x,y;
	Vector2 min, max;
	DataConverter::getMinMaxIn2D(data, min, max, dimVec);
	const Vector2 diff = max - min;
	max[0] += diff[0] * 0.2;
	max[1] += diff[1] * 0.2;
	min[0] -= diff[0] * 0.2;
	min[1] -= diff[1] * 0.2;
	const Real amountOfPointsOnOneAxis = std::max(CommandSettings::get_visuRes(), CommandSettings::get_visuResSimple());
	Vector2 stepSize = (1. / amountOfPointsOnOneAxis) * (max - min);
	int amount = 0;
	std::vector<Real> labels;
	const Real elementInX = ceil((Real)((max[0] - min[0]) / stepSize[0])) + 1;
	const Real elementInY = ceil((Real)((max[1] - min[1]) / stepSize[1])) + 1;
	const unsigned int classAmount = predictor->amountOfClasses();
	Data points;
	for(Real xVal = min[0]; xVal < max[0]; xVal += stepSize[0]){
		//for(Real xVal = max[0]; xVal >= min[0]; xVal -= stepSize[0]){
		for(Real yVal = min[1]; yVal < max[1]; yVal+= stepSize[1]){
			VectorX* ele = new VectorX(dim);
			for(int i = 0; i < dim; ++i){
				if(i == x){
					(*ele)[i] = xVal;
				}else if(i == y){
					(*ele)[i] = yVal;
				}else{
					(*ele)[i] = 0;
				}
			}
			points.push_back(ele);
			++amount;
		}
	}
	const bool complex = CommandSettings::get_visuRes() > 0;
	Labels result;
	std::vector< std::vector<Real> > probs;
	if(complex){
		predictor->predictData(points, result, probs);
	}else{
		predictor->predictData(points, result);
	}
	int counter = 0;
	int iX = 0, iY;
	std::vector< std::vector<Real> > colors(classAmount, std::vector<Real>(3));
	const bool justTwoClasses = classAmount == 2;
	if(!justTwoClasses){
		for(unsigned int i = 0; i < classAmount; ++i){ // save all colors
			ColorConverter::HSV2LAB((i + 1) / (Real) (classAmount) * (Real) 360., 1.0, 1.0, colors[i][0], colors[i][1], colors[i][2]);
		}
	}
	const unsigned int fac = 32;
	cv::Mat img(elementInY * fac, elementInX * fac, CV_8UC3, cv::Scalar(0, 0, 0));
	for(Real xVal = min[0]; xVal < max[0]; xVal += stepSize[0]){
		if(iX >= elementInX){
			printError("This should not happen for x: " << iX);
			continue;
		}
		iY = 0;
		for(Real yVal = min[1]; yVal < max[1]; yVal += stepSize[1]){
			if(iY >= elementInY){
				printError("This should not happen for y: " << iY);
				continue;
			}
			//Real val = fmax(fmin(1.0,), 0.0);
			Real r, g, b_;
			if(complex){
				Real l = 0, a = 0, b = 0;
				Real sum = 0;
				for(unsigned int i = 0; i < classAmount; ++i){
					sum += probs[counter][i];
				}
				if(!justTwoClasses){
					if(sum > 0){
						for(unsigned int i = 0; i < classAmount; ++i){
							probs[counter][i] /= sum;
							//						probs[counter][i] = -(probs[counter][i] * probs[counter][i]) + 2. * probs[counter][i]; // strech more
						}
						for(unsigned int i = 0; i < classAmount; ++i){
							l += probs[counter][i] * colors[i][0];
							a += probs[counter][i] * colors[i][1];
							b += probs[counter][i] * colors[i][2];
						}
					}
					ColorConverter::LAB2RGB(l, a, b, r, g, b_);
				}else{
					const Real prob = probs[counter][0] / sum;
					ColorConverter::getProbColorForBinaryRGB(prob, r, g, b_);
				}
			}else{
				Real l = 0, a = 0, b = 0;
				l = colors[result[counter]][0];
				a = colors[result[counter]][1];
				b = colors[result[counter]][2];
				ColorConverter::LAB2RGB(l, a, b, r, g, b_);
			}
			for(unsigned int t = 0; t < fac; ++t){
				for(unsigned int l = 0; l < fac; ++l){
					cv::Vec3b& color = img.at<cv::Vec3b>((elementInY - iY - 1) * fac + t,iX * fac + l);
					color[0] = (uchar) (b_ * 255);
					color[1] = (uchar) (g * 255);
					color[2] = (uchar) (r * 255);
				}
			}
			++counter;
			++iY;
		}
		++iX;
	}
	cv::Scalar black(0,0,0);
	for(LabeledDataConstIterator it = data.cbegin(); it != data.cend(); ++it, ++counter){
		const int dx = ((**it)[x] - min[0]) / (max[0] - min[0]) * elementInX * fac;
		const int dy = (1. - ((**it)[y] - min[1]) / (max[1] - min[1])) * elementInY * fac;
		const int label = (*it)->getLabel();
		Real r, g, b;
		if(!justTwoClasses){
			ColorConverter::LAB2RGB(colors[label][0], colors[label][1], colors[label][2], r, g, b);
		}else{
			ColorConverter::getProbColorForBinaryRGB(label == 1 ? 0.0 : 1.0, r, g, b);
		}
		cv::Scalar actColor(b * 255,g * 255,r * 255);
		cv::circle(img, cv::Point(dx, dy), fac / 2 * amountOfPointsOnOneAxis / 50, actColor, CV_FILLED, CV_AA);
		cv::circle(img, cv::Point(dx, dy), fac / 2 * amountOfPointsOnOneAxis / 50, black, fac / 6 * amountOfPointsOnOneAxis / 50, CV_AA);
	}
	cv::imwrite(Logger::getActDirectory() + fileName, img);
	for(unsigned int i = 0; i < points.size(); ++i){
		SAVE_DELETE(points[i]);
	}
}

void DataWriterForVisu::drawSvgDataPoints(std::ofstream& file, const LabeledData& points,
		const Vector2& min, const Vector2& max, const Vector2i& dim, const std::list<unsigned int>& selectedInducingPoints, const int amountOfClasses, const IVM* ivm){
	unsigned int counter = 0;
	if(selectedInducingPoints.size() > 0){
		std::list<LabeledDataConstIterator> inducedPoints;
		for(LabeledDataConstIterator it = points.cbegin(); it != points.cend(); ++it, ++counter){
			bool isInduced = false;
			for(auto itI = selectedInducingPoints.cbegin(); itI != selectedInducingPoints.cend(); ++itI){
				if((*itI) == counter){
					isInduced = true; break;
				}
			}
			if(isInduced){
				inducedPoints.push_back(it);
				continue;
			}
			const Real dx = ((**it)[dim[0]] - min[0]) / (max[0] - min[0]) * 100.;
			const Real dy = ((**it)[dim[1]] - min[1]) / (max[1] - min[1]) * 100.;
			std::string color = "blue";
			if(ivm != nullptr){
				if((*it)->getLabel() == ivm->getLabelForOne()){
					color = "red";
				}
			}else{
				if((*it)->getLabel() == 0){
					color = "red";
				}
			}
			file << "<circle cx=\"" << dx << "%\" cy=\"" << dy << "%\" r=\"8\" fill=\"" << color << "\" stroke=\"black\" stroke-width=\"2\"/> \n";
		}
		// draw after all points, to make them more visible
		for(auto it = inducedPoints.cbegin(); it != inducedPoints.cend(); ++it){
			const Real dx = ((***it)[dim[0]] - min[0]) / (max[0] - min[0]) * 100.;
			const Real dy = ((***it)[dim[1]] - min[1]) / (max[1] - min[1]) * 100.;
			std::string color = "blue";
			if(ivm != nullptr){
				if((**it)->getLabel() == ivm->getLabelForOne()){
					color = "red";
				}
			}else{
				if((**it)->getLabel() == 0){
					color = "red";
				}
			}
			file << "<circle cx=\"" << dx << "%\" cy=\"" << dy << "%\" r=\"8\" fill=\"" << color << "\" stroke=\"black\" stroke-width=\"2\"/> \n";
			file << "<circle cx=\"" << dx << "%\" cy=\"" << dy << "%\" r=\"3\" fill=\"white\" /> \n";
		}
	}else{
		for(LabeledDataConstIterator it = points.cbegin(); it != points.cend(); ++it, ++counter){
			const Real dx = ((**it)[dim[0]] - min[0]) / (max[0] - min[0]) * 100.;
			const Real dy = ((**it)[dim[1]] - min[1]) / (max[1] - min[1]) * 100.;
			if(amountOfClasses == 2){
				Real prob = 0;
				if(ivm != nullptr){
					if((*it)->getLabel() == ivm->getLabelForOne()){
						prob = 1;
					}
				}else{
					if((*it)->getLabel() == 0){
						prob = 1;
					}
				}
				Real r, g, b;
				ColorConverter::getProbColorForBinaryRGB(prob, r,g,b);
				std::stringstream color;
				color << "rgb(" << r * 100. << "%," << g * 100. << "%," << b * 100. << "%)";
				file << "<circle cx=\"" << dx << "%\" cy=\"" << dy << "%\" r=\"8\" fill=\"" << color.str() << "\" stroke=\"black\" stroke-width=\"2\"/> \n";
			}else{
				Real r, g, b;
				//std::cout << "(*it)->getLabel(): " << (*it)->getLabel() + 1 << " " << amountOfClasses << ",";
				ColorConverter::HSV2RGB(((*it)->getLabel() + 1) / (Real) (amountOfClasses) * 360., 1.0, 1.0, r, g, b);
				file << "<circle cx=\"" << dx << "%\" cy=\"" << dy << "%\" r=\"8\" fill=\"rgb(" << r * 100. << "%," << g * 100. << "%," << b * 100. << "%)\" stroke=\"black\" stroke-width=\"2\"/> \n";
			}
		}
	}
}

void DataWriterForVisu::writeSvg(const std::string& fileName, const Matrix& mat){
	std::ofstream file;
	openSvgFile(fileName, 1920, mat.cols(), mat.rows(), file);
	Real min, max;
	const bool ignoreDBLMAXNEG = true;
	DataConverter::getMinMax(mat, min, max, ignoreDBLMAXNEG);
	for(int iX = 0; iX < mat.rows(); ++iX){
		for(int iY = 0; iY < mat.cols(); ++iY){
			if(mat(iX,iY) > NEG_REAL_MAX){
				const Real prob = (mat(iX,iY)  - min) / (max - min);
				Real r,g,b;
				ColorConverter::HSV2RGB(prob * 360.,1.0, 1.0,r,g,b);
				drawSvgRect(file, iX / (Real)mat.rows() * 100., iY / (Real)mat.cols()  * 100., 100.0 / mat.rows(), 100.0 / mat.cols(), (r * 100.0), (g * 100.0), (b * 100.0));
			}else{
				drawSvgRect(file, iX / (Real)mat.rows() * 100., iY / (Real)mat.cols()  * 100., 100.0 / mat.rows(), 100.0 / mat.cols(), 10, 10, 10);
			}
		}
	}
	closeSvgFile(file);
}

void DataWriterForVisu::writeSvg(const std::string& fileName, const std::list<Real>& list, const std::list<std::string>& colors){
	if(list.size() == 0 || list.size() != colors.size()){
		printError("The lists are unequal or have no elements!");
		return;
	}
	VectorX vec(list.size());
	unsigned int t = 0;
	for(auto it = list.begin(); it != list.end(); ++it, ++t){
		vec[t] = *it;
	}
	Real min, max;
	DataConverter::getMinMax(vec, min, max, true);
	if(max == min){
		printError("Min and max are equal!");
		return;
	}
	std::ofstream file;
	const bool open = openSvgFile(fileName, 820., 1.0, 1.0, file);
	if(open){
		drawSvgCoords(file, 7.5, 7.5, 10, 10, vec.size(), max - min, min, max, 820., 820.);
		drawSvgDots(file, vec, 10., 10., min, max, 820., 820., colors);
		closeSvgFile(file);
	}
}

void DataWriterForVisu::writeSvg(const std::string& fileName, const VectorX& vec, const bool drawLine){
	if(vec.rows() == 0){
		return;
	}
	Real min, max;
	DataConverter::getMinMax(vec, min, max);
	std::ofstream file;
	openSvgFile(fileName, 820., 1.0, 1.0, file);
	drawSvgCoords(file, 7.5, 7.5, 10, 10, vec.size(), max - min, min, max, 820., 820.);
	if(drawLine){
		drawSvgLine(file, vec, 10., 10., min, max, 820., 820., "black");
	}else{
		drawSvgDots(file, vec, 10., 10., min, max, 820., 820., "black");
	}
	closeSvgFile(file);
}

void DataWriterForVisu::writeSvg(const std::string& fileName, const std::list<VectorX>& vecs, const bool drawLine){
	if(vecs.size() == 0){
		return;
	}
	const auto size = vecs.begin()->rows();
	for(auto it = vecs.cbegin(); it != vecs.cend(); ++it){
		if(size != it->rows()){
			return;
		}
	}
	Real min = REAL_MAX, max = NEG_REAL_MAX;
	for(auto it = vecs.cbegin(); it != vecs.cend(); ++it){
		Real minAct, maxAct;
		DataConverter::getMinMax(*it, minAct, maxAct);
		if(minAct < min){
			min = minAct;
		}
		if(maxAct > max){
			max = maxAct;
		}
	}
	std::ofstream file;
	openSvgFile(fileName, 820., 1.0, 1.0, file);
	drawSvgCoords(file, 7.5, 7.5, 10, 10, size, max - min, min, max, 820., 820.);
	std::vector<std::string> colors(4);
	colors[0] = "red";
	colors[1] = "blue";
	colors[2] = "green";
	colors[3] = "black";
	auto itColor = colors.cbegin();
	if(drawLine){
		for(auto it = vecs.cbegin(); it != vecs.cend(); ++it){
			drawSvgLine(file, *it, 10., 10., min, max, 820., 820., *itColor);
			++itColor;
			if(itColor == colors.end())
				itColor = colors.begin();
		}
	}else{
		for(auto it = vecs.cbegin(); it != vecs.cend(); ++it){
			drawSvgDots(file, *it, 10., 10., min, max, 820., 820.,  *itColor);
			++itColor;
			if(itColor == colors.end())
				itColor = colors.begin();
		}
	}
	closeSvgFile(file);
}

void DataWriterForVisu::writeSvg(const std::string& fileName, const std::list<Real>& list, const bool drawLine){
	if(list.size() == 0){
		return;
	}
	VectorX vec(list.size());
	unsigned int i = 0;
	for(auto it = list.begin(); it != list.end(); ++it, ++i){
		vec.coeffRef(i) = *it;
	}
	writeSvg(fileName, vec, drawLine);
}

void DataWriterForVisu::drawSvgCoords(std::ofstream& file,
		const Real startX, const Real startY, const Real startXForData, const Real startYForData, const Real xSize,
		const Real ySize, const Real min, const Real max, const Real width, const Real heigth, const bool useAllXSegments
		, const Real minX, const Real maxX){
	UNUSED(ySize);
	file << "<path d=\"M " << startX / 100. * width << " "<< startY / 100. * heigth
		 << " l " << (100. - 2. * startX) / 100. * width  << " 0"
		 << " M " << startX / 100. * width << " "<< startY / 100. * heigth
		 << " l 0 " << (100. - 2. * startY) / 100. * heigth
		 << "\" fill=\"transparent\" stroke=\"black\"/> \n";
	const Real widthOfMarks = 8;
	unsigned int amountOfSegm = useAllXSegments ? xSize - 1 : std::min(10, (int) xSize - 1);
	Real segmentWidth = ((100 - startXForData - startXForData) / 100. * width) / amountOfSegm;
	file << "<path d=\"";//M " << startXForData / 100. * width << " "  << startY / 100. *heigth - widthOfMarks / 2;
	for(unsigned int i = 0; i <= amountOfSegm; ++i){
		file << " M " << startXForData / 100. * width + i * segmentWidth
			 << " " << startY / 100. * heigth - widthOfMarks / 2
			 << " l " << "0 " << widthOfMarks;
	}
	file << "\" fill=\"transparent\" stroke=\"black\"/> \n";
	if(minX == maxX){
		for(unsigned int i = 0; i <= amountOfSegm; ++i){ // transform=\"translate(0,10) scale(1,-1) translate(0,-10)\"
			file << "<text x=\"" << startXForData / 100. * width + i * segmentWidth
					<< "\" y=\"" << -(startY / 100. * heigth - widthOfMarks / 2. - 20)<< "\" transform=\"scale(1,-1)\" "
					<< "font-family=\"sans-serif\" font-size=\"10px\" text-anchor=\"middle\" fill=\"black\">"
					<< (int)((xSize - 1) / amountOfSegm * i) + 1 << "</text>\n";
		}
	}else{
		for(unsigned int i = 0; i <= amountOfSegm; ++i){ // transform=\"translate(0,10) scale(1,-1) translate(0,-10)\"
			file << "<text x=\"" << startXForData / 100. * width + i * segmentWidth
					<< "\" y=\"" << -(startY / 100. * heigth - widthOfMarks / 2. - 20)<< "\" transform=\"scale(1,-1)\" "
					<< "font-family=\"sans-serif\" font-size=\"10px\" text-anchor=\"middle\" fill=\"black\">"
					<< StringHelper::number2String(((maxX - minX) * i / (Real) amountOfSegm + minX), 3) << "</text>\n";
		}
	}
	amountOfSegm = 10;
	segmentWidth = ((100 - startXForData - startXForData) / 100. * width) / amountOfSegm;
	file << "<path d=\"";//M " << startXForData / 100. * width << " "  << startY / 100. *heigth - widthOfMarks / 2;
	for(unsigned int i = 0; i <= amountOfSegm; ++i){
		file << " M " << startY / 100. * heigth - widthOfMarks / 2
		  	 << " " << startYForData / 100. * width + i * segmentWidth
			 << " l " << widthOfMarks << " 0 ";
	}
	file << "\" fill=\"transparent\" stroke=\"black\"/> \n";
	for(unsigned int i = 0; i <= amountOfSegm; ++i){ // transform=\"translate(0,10) scale(1,-1) translate(0,-10)\"
		file << "<text x=\"" << (startY / 100. * heigth - widthOfMarks / 2. - 20)
			 << "\" y=\"" << -(startXForData / 100. * width + i * segmentWidth) << "\" transform=\"scale(1,-1)\" "
			 << "font-family=\"sans-serif\" font-size=\"10px\" text-anchor=\"middle\" fill=\"black\">"
			 << StringHelper::number2String(((max - min) * i / (Real) amountOfSegm + min), 3) << "</text>\n";
	}
	file << "<path d=\"M " << startX / 100. * heigth - widthOfMarks / 2 << " " << (100. - startY) / 100. * width
		 << " l " << widthOfMarks << " " << 0 << " l " << - widthOfMarks / 2.0 << " " << widthOfMarks * 1.5
		 << " l " << - widthOfMarks / 2.0 << " " << -widthOfMarks * 1.5
		 << "\" fill=\"black\" stroke=\"black\"/> \n";
	file << "<path d=\"M " << (100. - startX) / 100. * width << " " << startY / 100. * heigth - widthOfMarks / 2
		 << " l " << 0 << " " << widthOfMarks << " l " << widthOfMarks * 1.5 << " " << - widthOfMarks / 2.0
		 << " l " << -widthOfMarks * 1.5 << " " << - widthOfMarks / 2.0
		 << "\" fill=\"black\" stroke=\"black\"/> \n";
}

void DataWriterForVisu::drawSvgCoords2D(std::ofstream& file,
		const Real startX, const Real startY, const Real startXForData, const Real startYForData, const Vector2& min,
		const Vector2& max, const unsigned int amountOfSegm, const Real width, const Real heigth){
	file << "<path d=\"M " << startX / 100. * width << " "<< startY / 100. * heigth
			<< " l " << (100. - 2. * startX) / 100. * width  << " 0"
			<< " M " << startX / 100. * width << " "<< startY / 100. * heigth
			<< " l 0 " << (100. - 2. * startY) / 100. * heigth
			<< "\" fill=\"transparent\" stroke=\"black\"/> \n";
	const Real widthOfMarks = 8;
	Real segmentWidth = ((100 - startXForData - startXForData) / 100. * width) / amountOfSegm;
	file << "<path d=\"";//M " << startXForData / 100. * width << " "  << startY / 100. *heigth - widthOfMarks / 2;
	for(unsigned int i = 0; i <= amountOfSegm; ++i){
		file << " M " << startXForData / 100. * width + i * segmentWidth
				<< " " << startY / 100. * heigth - widthOfMarks / 2
				<< " l " << "0 " << widthOfMarks;
	}
	file << "\" fill=\"transparent\" stroke=\"black\"/> \n";
	for(unsigned int i = 0; i <= amountOfSegm; ++i){ // transform=\"translate(0,10) scale(1,-1) translate(0,-10)\"
		file << "<text x=\"" << startXForData / 100. * width + i * segmentWidth
				<< "\" y=\"" << -(startY / 100. * heigth - widthOfMarks / 2. - 20)<< "\" transform=\"scale(1,-1)\" "
				<< "font-family=\"sans-serif\" font-size=\"10px\" text-anchor=\"middle\" fill=\"black\">"
				<< StringHelper::number2String((Real) ((i / (Real) (amountOfSegm - 1)) * (max[0] - min[0]) + min[0]), 4)  << "</text>\n";
	}
	segmentWidth = ((100 - startXForData - startXForData) / 100. * width) / amountOfSegm;
	file << "<path d=\"";//M " << startXForData / 100. * width << " "  << startY / 100. *heigth - widthOfMarks / 2;
	for(unsigned int i = 0; i <= amountOfSegm; ++i){
		file << " M " << startY / 100. * heigth - widthOfMarks / 2
				<< " " << startYForData / 100. * width + i * segmentWidth
				<< " l " << widthOfMarks << " 0 ";
	}
	file << "\" fill=\"transparent\" stroke=\"black\"/> \n";
	for(unsigned int i = 0; i <= amountOfSegm; ++i){ // transform=\"translate(0,10) scale(1,-1) translate(0,-10)\"
		file << "<text x=\"" << (startY / 100. * heigth - widthOfMarks / 2. - 20)
						 << "\" y=\"" << -(startXForData / 100. * width + i * segmentWidth) << "\" transform=\"scale(1,-1)\" "
						 << "font-family=\"sans-serif\" font-size=\"10px\" text-anchor=\"middle\" fill=\"black\">"
						 << StringHelper::number2String((Real) ((i / (Real) (amountOfSegm - 1)) * (max[1] - min[1]) + min[1]), 4) << "</text>\n";
	}
	file << "<path d=\"M " << startX / 100. * heigth - widthOfMarks / 2 << " " << (100. - startY) / 100. * width
			<< " l " << widthOfMarks << " " << 0 << " l " << - widthOfMarks / 2.0 << " " << widthOfMarks * 1.5
			<< " l " << - widthOfMarks / 2.0 << " " << -widthOfMarks * 1.5
			<< "\" fill=\"black\" stroke=\"black\"/> \n";
	file << "<path d=\"M " << (100. - startX) / 100. * width << " " << startY / 100. * heigth - widthOfMarks / 2
			<< " l " << 0 << " " << widthOfMarks << " l " << widthOfMarks * 1.5 << " " << - widthOfMarks / 2.0
			<< " l " << -widthOfMarks * 1.5 << " " << - widthOfMarks / 2.0
			<< "\" fill=\"black\" stroke=\"black\"/> \n";
}

void DataWriterForVisu::drawSvgLine(std::ofstream& file, const VectorX vec,
		const Real startX, const Real startY, const Real min,
		const Real max, const Real width, const Real heigth, const std::string& color){
	if(vec.rows() > 0){
		const Real diff = max == min ? 1 : max - min;
		file << "<path d=\"M " << (0 / (Real) (vec.rows() - 1) * (100. - 2. * startX) + startX) / 100. * width << " "
				 	 	 	   << ((vec[0] - min) / diff * (100. - 2. * startY) + startY) / 100. * heigth << " \n";
		for(unsigned int i = 1; i < vec.rows(); ++i){
			file << " L "<< (i / (Real) (vec.rows() - 1) * (100. - 2. * startX) + startX) / 100. * width << " "
						 << ((vec[i] - min) / diff * (100. - 2. * startY) + startY) / 100. * heigth << " \n";
		}
		file << "\" fill=\"transparent\" stroke=\"" << color << "\"/> \n";
	}
}

void DataWriterForVisu::drawSvgDots(std::ofstream& file, const VectorX vec,
		const Real startX, const Real startY, const Real min,
		const Real max, const Real width, const Real heigth, const std::string& color){
	const Real diff = max == min ? 1 : max - min;
	for(unsigned int i = 0; i < vec.rows(); ++i){
		if(vec[i] > NEG_REAL_MAX){ // don't use NEG_REAL_MAX Values they should be ignored
			file << "<circle cx=\"" << (i / (Real) vec.rows() * (100. - 2. * startX) + startX) / 100. * width
					<< "\" cy=\"" << ((vec[i] - min) / diff * (100. - 2. * startY) + startY) / 100. * heigth
					<< "\" r=\"3\" fill=\"transparent\" stroke=\"" << color << "\" /> \n";
		}
	}
}

void DataWriterForVisu::drawSvgDots(std::ofstream& file, const VectorX vec,
		const Real startX, const Real startY, const Real min,
		const Real max, const Real width, const Real heigth, const std::list<std::string>& colors){
	const Real diff = max == min ? 1 : max - min;
	auto it = colors.cbegin();
	for(unsigned int i = 0; i < vec.rows(); ++i){
		if(vec[i] > NEG_REAL_MAX){ // don't use NEG_REAL_MAX Values they should be ignored
			file << "<circle cx=\"" << (i / (Real) vec.rows() * (100. - 2. * startX) + startX) / 100. * width
					<< "\" cy=\"" << ((vec[i] - min) / diff * (100. - 2. * startY) + startY) / 100. * heigth
					<< "\" r=\"3\" fill=\"transparent\" stroke=\"" << *it << "\" /> \n";
		}
		++it;
	}
}

bool DataWriterForVisu::openSvgFile(const std::string& fileName, const Real width, const Real problemWidth, const Real problemHeight, std::ofstream& file){
	file.open(Logger::getActDirectory() + fileName);
	const Real height = (width / problemHeight *  problemWidth);
	file << "<svg version=\"1.1\" " <<
			"\nbaseProfile=\"full\"" <<
			"\nwidth=\"" << width << "\" height=\""<< (int) height << "\"\n" <<
			"xmlns=\"http://www.w3.org/2000/svg\">" << "\n";
	file << "<g transform=\"translate(0," << height / 2.0 <<") scale(1,-1) translate(0,-" << height / 2.0 <<")\">\n";
	return file.is_open();
}

void DataWriterForVisu::closeSvgFile(std::ofstream& file){
	file << "</g></svg>\n";
	file.close();
}

void DataWriterForVisu::drawSvgRect(std::ofstream& file, const Real xPos, const Real yPos,
			const Real width, const Real height,
			const int r, const int g, const int b){
	file << "<rect x=\""<< xPos <<"%\" y=\""<< yPos << "%\" width=\"" << width << "%\" height=\""
						<< height << "%\" fill=\"rgb(" << r << "%," << g << "%," << b << "%)\" /> \n";
}

