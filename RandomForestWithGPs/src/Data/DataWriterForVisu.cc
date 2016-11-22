/*
 * DataWriterForVisu.cc
 *
 *  Created on: 04.06.2016
 *      Author: Max
 */

#include "DataWriterForVisu.h"
#include "../Utility/ColorConverter.h"
#include "DataConverter.h"
#include <opencv2/core.hpp>
#include <math.h>
#include "../Base/CommandSettings.h"

DataWriterForVisu::DataWriterForVisu()
{
	// TODO Auto-generated constructor stub

}

DataWriterForVisu::~DataWriterForVisu(){
}

void DataWriterForVisu::writeData(const std::string& fileName, const ClassData& data,
		const int x, const int y){
	if(data.size() > 0){
		if(!(data[0]->rows() > x && data[0]->rows() > y && x != y && y >= 0 && x >= 0)){
			printError("These axis x: " << x << ", y: " << y << " aren't printable!");
			return;
		}
		std::ofstream file;
		file.open(fileName);
		if(file.is_open()){
			for(ClassDataConstIterator it = data.cbegin(); it != data.cend(); ++it){
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
		const double amountOfPointsOnOneAxis,
		const ClassData& data,
		const int x, const int y){
	if(data.size() == 0){
		printError("No data is given, this data is needed to find min and max!");
		return;
	}
	const int dim = data[0]->rows();
	Eigen::Vector2i dimVec;
	dimVec << x,y;
	Eigen::Vector2d min, max;
	DataConverter::getMinMaxIn2D(data, min, max, dimVec);
	Eigen::Vector2d stepSize = (1. / amountOfPointsOnOneAxis) * (max - min);
	std::ofstream file;
	file.open(fileName);
	Data points;
	points.reserve(amountOfPointsOnOneAxis * (amountOfPointsOnOneAxis + 1));
	int amount = 0;
	for(double xVal = max[0]; xVal >= min[0]; xVal -= stepSize[0]){
		for(double yVal = min[1]; yVal < max[1]; yVal+= stepSize[1]){
			DataPoint ele(dim);
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
		const double amountOfPointsOnOneAxis,
		const ClassData& data,
		const int x, const int y){
	if(data.size() == 0){
		printError("No data is given, this data is needed to find min and max!");
		return;
	}
	const int dim = data[0]->rows();
	Eigen::Vector2i dimVec;
	dimVec << x,y;
	Eigen::Vector2d min, max;
	DataConverter::getMinMaxIn2D(data, min, max, dimVec);
	Eigen::Vector2d stepSize = (1. / amountOfPointsOnOneAxis) * (max - min);
	std::ofstream file;
	file.open(fileName);
	Data points;
	points.reserve(amountOfPointsOnOneAxis * (amountOfPointsOnOneAxis + 1));
	int amount = 0;
	for(double xVal = max[0]; xVal >= min[0]; xVal -= stepSize[0]){
		for(double yVal = min[1]; yVal < max[1]; yVal+= stepSize[1]){
			DataPoint ele(dim);
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

void DataWriterForVisu::writeSvg(const std::string& fileName, const PredictorBinaryClass* predictor, const ClassData& data, const int x, const int y){
	if(data.size() == 0){
		printError("No data is given, this data is needed to find min and max!");
		return;
	}
	const int dim = data[0]->rows();
	Eigen::Vector2i dimVec;
	dimVec << x,y;
	Eigen::Vector2d min, max;
	DataConverter::getMinMaxIn2D(data, min, max, dimVec);
	const Eigen::Vector2d diff = max - min;
	max[0] += diff[0] * 0.2;
	max[1] += diff[1] * 0.2;
	min[0] -= diff[0] * 0.2;
	min[1] -= diff[1] * 0.2;
	const double amountOfPointsOnOneAxis = std::max(CommandSettings::get_visuRes(), CommandSettings::get_visuResSimple());
	Eigen::Vector2d stepSize = (1. / amountOfPointsOnOneAxis) * (max - min);
	int amount = 0;
	std::vector<double> labels;
	const double elementInX = (int)((max[0] - min[0]) / stepSize[0]);
	const double elementInY = (int)((max[1] - min[1]) / stepSize[1]);
	int iX = 0, iY;
	std::ofstream file;
	openSvgFile(fileName, 820., (double) (max[0] - min[0]), (double) (max[1] - min[1]), file);
	const bool complex = CommandSettings::get_visuRes() > 0;
	for(double xVal = max[0]; xVal >= min[0]; xVal -= stepSize[0]){
		iY = 0;
		for(double yVal = min[1]; yVal < max[1]; yVal+= stepSize[1]){
			DataPoint ele(dim);
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
			double val = fmax(fmin(1.0, predictor->predict(ele)), 0.0);
			if(!complex){
				val = val >= 0.5 ? 1 : 0.;
			}
			drawSvgRect(file, iX / elementInX * 100., iY / elementInY  * 100.,
					100.0 / elementInX, 100.0 / elementInY, val * 100, 0, (1.-val) * 100);
			++amount;
			++iY;
		}
		++iX;
	}
	std::list<int> empty;
	drawSvgDataPoints(file, data, min, max, dimVec, empty);
	closeSvgFile(file);
}

void DataWriterForVisu::writeSvg(const std::string& fileName, const PredictorMultiClass* predictor, const ClassData& data, const int x, const int y){
	if(data.size() == 0){
		printError("No data is given, this data is needed to find min and max!");
		return;
	}
	const int dim = data[0]->rows();
	Eigen::Vector2i dimVec;
	dimVec << x,y;
	Eigen::Vector2d min, max;
	DataConverter::getMinMaxIn2D(data, min, max, dimVec);
	const Eigen::Vector2d diff = max - min;
	max[0] += diff[0] * 0.2;
	max[1] += diff[1] * 0.2;
	min[0] -= diff[0] * 0.2;
	min[1] -= diff[1] * 0.2;
	const double amountOfPointsOnOneAxis = std::max(CommandSettings::get_visuRes(), CommandSettings::get_visuResSimple());
	Eigen::Vector2d stepSize = (1. / amountOfPointsOnOneAxis) * (max - min);
	int amount = 0;
	std::vector<double> labels;
	const double elementInX = (int)((max[0] - min[0]) / stepSize[0]);
	const double elementInY = (int)((max[1] - min[1]) / stepSize[1]);
	std::ofstream file;
	const int classAmount = predictor->amountOfClasses();
	openSvgFile(fileName, 820., (double) (max[0] - min[0]), (double) (max[1] - min[1]), file);
	Data points;
	for(double xVal = min[0]; xVal < max[0]; xVal += stepSize[0]){
	//for(double xVal = max[0]; xVal >= min[0]; xVal -= stepSize[0]){
		for(double yVal = min[1]; yVal < max[1]; yVal+= stepSize[1]){
			DataPoint* ele = new DataPoint(dim);
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
	std::vector< std::vector<double> > probs;
	if(complex){
		predictor->predictData(points, result, probs);
	}else{
		predictor->predictData(points, result);
	}
	int counter = 0;
	int iX = 0, iY;
	std::vector< std::vector<double> > colors(classAmount, std::vector<double>(3));
	for(unsigned int i = 0; i < classAmount; ++i){ // save all colors
		ColorConverter::HSV2LAB((i + 1) / (double) (classAmount) * 360., 1.0, 1.0, colors[i][0], colors[i][1], colors[i][2]);
	}
	for(double xVal = min[0]; xVal < max[0]; xVal += stepSize[0]){
		iY = 0;
		for(double yVal = min[1]; yVal < max[1]; yVal+= stepSize[1]){
				//double val = fmax(fmin(1.0,), 0.0);
			double r, g, b_;
			if(complex){
				double l = 0, a = 0, b = 0;
				double sum = 0;
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
				double l = 0, a = 0, b = 0;
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
		delete points[i];
	}
	std::list<int> empty;
	drawSvgDataPoints(file, data, min, max, dimVec, empty, classAmount);
	closeSvgFile(file);
}

void DataWriterForVisu::writeHisto(const std::string&fileName, const std::list<double>& list, const unsigned int nrOfBins, const double minValue, const double maxValue){
	if(list.size() == 0){
		printError("No data is given!");
		return;
	}
	double min, max;
	if(minValue == maxValue && minValue == -1){
		DataConverter::getMinMax(list, min, max);
	}else{
		min = minValue;
		max = maxValue;
	}
	if(min == max){
		printError("The min and max value are the same, the calculation of the histogram is not possible!"); return;
	}
	Eigen::VectorXd counter = Eigen::VectorXd::Zero(nrOfBins);
	for(std::list<double>::const_iterator it = list.begin(); it != list.end(); ++it){
		++counter[(*it - min) / (max - min) * (nrOfBins - 1)];
	}
	double minCounter, maxCounter;
	DataConverter::getMinMax(counter, minCounter, maxCounter);
	std::ofstream file;
	openSvgFile(fileName, 820., 1., 1., file);
	const double startOfData = 7.5;
	drawSvgCoords(file, 5., 5., 7.5, 7.5, nrOfBins + 1, maxCounter, 0, maxCounter, 820, 820, true);
	const double dataWidth = (100.0 - 2. * startOfData);
	const double width = (1. / (double) nrOfBins *  (dataWidth * 0.95));
	const double offset = (1. / (double) nrOfBins * (dataWidth * 0.05));
	for(unsigned int i = 0; i < nrOfBins; ++i){
		const double height = counter[i] / (double) maxCounter * dataWidth;
		drawSvgRect(file, i / (double) nrOfBins * dataWidth + offset + startOfData, startOfData, width, height == 0. ? 1. : height, 76, 5, 78);
	}
	closeSvgFile(file);
}

void DataWriterForVisu::writeSvg(const std::string& fileName, const IVM& ivm, const std::list<int>& selectedInducingPoints,
		const ClassData& data, const int x, const int y, const int type){
	if(data.size() == 0){
		printError("No data is given, this data is needed to find min and max!");
		return;
	}
	const int dim = data[0]->rows();
	Eigen::Vector2i dimVec;
	dimVec << x,y;
	Eigen::Vector2d min, max;
	DataConverter::getMinMaxIn2D(data, min, max, dimVec);
	const Eigen::Vector2d diff = max - min;
	if(diff[0] <= 1e-7 || diff[1] <= 1e-7){
		printError("The min and max of the desired axis is equal for " << fileName << "!"); return;
	}
	max[0] += diff[0] * 0.2;
	max[1] += diff[1] * 0.2;
	min[0] -= diff[0] * 0.2;
	min[1] -= diff[1] * 0.2;
	const double amountOfPointsOnOneAxis = std::max(CommandSettings::get_visuRes(), CommandSettings::get_visuResSimple());
	Eigen::Vector2d stepSize = (1. / amountOfPointsOnOneAxis) * (max - min);
	int amount = 0;
	const double elementInX = (int)((max[0] - min[0]) / stepSize[0]);
	const double elementInY = (int)((max[1] - min[1]) / stepSize[1]);
	int iX = 0, iY;
	std::ofstream file;
	openSvgFile(fileName, 820., (double) (max[0] - min[0]), (double) (max[1] - min[1]), file);
	const bool complex = CommandSettings::get_visuRes() > 0;
	std::vector< std::vector<double> > colors(2, std::vector<double>(3));
	ColorConverter::RGB2LAB(1,0,0, colors[0][0], colors[0][1], colors[0][2]);
	ColorConverter::RGB2LAB(0,0,1, colors[1][0], colors[1][1], colors[1][2]);
	double minMu = DBL_MAX, minSigma = DBL_MAX, maxMu = -DBL_MAX, maxSigma = -DBL_MAX;
	if(type == 1 || type == 2){
		for(double xVal = min[0]; xVal < max[0]; xVal += stepSize[0]){
			for(double yVal = min[1]; yVal < max[1]; yVal+= stepSize[1]){
				DataPoint ele(dim);
				for(int i = 0; i < dim; ++i){
					if(i == x){
						ele[i] = xVal;
					}else if(i == y){
						ele[i] = yVal;
					}else{
						ele[i] = 0;
					}
				}
				double prob = 0;
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
	for(double xVal = min[0]; xVal < max[0]; xVal += stepSize[0]){
	//for(double xVal = max[0]; xVal >= min[0]; xVal -= stepSize[0]){
		iY = 0;
		for(double yVal = min[1]; yVal < max[1]; yVal+= stepSize[1]){
			DataPoint ele(dim);
			for(int i = 0; i < dim; ++i){
				if(i == x){
					ele[i] = xVal;
				}else if(i == y){
					ele[i] = yVal;
				}else{
					ele[i] = 0;
				}
			}
			double prob;
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
			double r, g, b_;
			double l = prob * colors[0][0] + (1-prob) * colors[1][0];
			double a = prob * colors[0][1] + (1-prob) * colors[1][1];
			double b = prob * colors[0][2] + (1-prob) * colors[1][2];
			ColorConverter::LAB2RGB(l, a, b, r, g, b_);

			//double val = fmax(fmin(1.0,), 0.0);
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

void DataWriterForVisu::writeSvg(const std::string& fileName, const ClassData& data, const int x, const int y){
	Eigen::Vector2i dimVec;
	dimVec << x,y;
	Eigen::Vector2d min, max;
	DataConverter::getMinMaxIn2D(data, min, max, dimVec);
	const Eigen::Vector2d diff = max - min;
	max[0] += diff[0] * 0.2;
	max[1] += diff[1] * 0.2;
	min[0] -= diff[0] * 0.2;
	min[1] -= diff[1] * 0.2;
	std::list<int> empty;
	std::ofstream file;
	std::map<unsigned int, unsigned int> classCounter;
	for(ClassData::const_iterator it = data.begin(); it != data.end(); ++it){
		std::map<unsigned int, unsigned int>::iterator itClass = classCounter.find((*it)->getLabel());
		if(itClass == classCounter.end()){
			classCounter.insert(std::pair<unsigned int, unsigned int>((*it)->getLabel(), 0));
		}
	}
	openSvgFile(fileName, 820., (double) (max[0] - min[0]), (double) (max[1] - min[1]), file);
	drawSvgDataPoints(file, data, min, max, dimVec, empty, classCounter.size());
	closeSvgFile(file);
}

void DataWriterForVisu::writeImg(const std::string& fileName, const PredictorMultiClass* predictor,
		const ClassData& data, const int x, const int y){
	if(data.size() == 0){
		printError("No data is given, this data is needed to find min and max!");
		return;
	}
	const int dim = data[0]->rows();
	Eigen::Vector2i dimVec;
	dimVec << x,y;
	Eigen::Vector2d min, max;
	DataConverter::getMinMaxIn2D(data, min, max, dimVec);
	const Eigen::Vector2d diff = max - min;
	max[0] += diff[0] * 0.2;
	max[1] += diff[1] * 0.2;
	min[0] -= diff[0] * 0.2;
	min[1] -= diff[1] * 0.2;
	const double amountOfPointsOnOneAxis = std::max(CommandSettings::get_visuRes(), CommandSettings::get_visuResSimple());
	Eigen::Vector2d stepSize = (1. / amountOfPointsOnOneAxis) * (max - min);
	int amount = 0;
	std::vector<double> labels;
	const double elementInX = ceil((double)(max[0] - min[0]) / stepSize[0]) + 1;
	const double elementInY = ceil((double)(max[1] - min[1]) / stepSize[1]) + 1;
	const int classAmount = predictor->amountOfClasses();
	Data points;
	for(double xVal = min[0]; xVal < max[0]; xVal += stepSize[0]){
		//for(double xVal = max[0]; xVal >= min[0]; xVal -= stepSize[0]){
		for(double yVal = min[1]; yVal < max[1]; yVal+= stepSize[1]){
			DataPoint* ele = new DataPoint(dim);
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
	std::vector< std::vector<double> > probs;
	if(complex){
		predictor->predictData(points, result, probs);
	}else{
		predictor->predictData(points, result);
	}
	int counter = 0;
	int iX = 0, iY;
	std::vector< std::vector<double> > colors(classAmount, std::vector<double>(3));
	for(unsigned int i = 0; i < classAmount; ++i){ // save all colors
		ColorConverter::HSV2LAB((i + 1) / (double) (classAmount) * 360., 1.0, 1.0, colors[i][0], colors[i][1], colors[i][2]);
	}
	int fac = 32;
	cv::Mat img(elementInY * fac, elementInX * fac, CV_8UC3, cv::Scalar(0, 0, 0));
	for(double xVal = min[0]; xVal < max[0]; xVal += stepSize[0]){
		if(iX >= elementInX){
			printError("This should not happen for x: " << iX);
			continue;
		}
		iY = 0;
		for(double yVal = min[1]; yVal < max[1]; yVal += stepSize[1]){
			if(iY >= elementInY){
				printError("This should not happen for y: " << iY);
				continue;
			}
			//double val = fmax(fmin(1.0,), 0.0);
			double r, g, b_;
			if(complex){
				double l = 0, a = 0, b = 0;
				double sum = 0;
				for(unsigned int i = 0; i < classAmount; ++i){
					sum += probs[counter][i];
				}
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
				double l = 0, a = 0, b = 0;
				l = colors[result[counter]][0];
				a = colors[result[counter]][1];
				b = colors[result[counter]][2];
				ColorConverter::LAB2RGB(l, a, b, r, g, b_);
			}
			for(unsigned int t = 0; t < fac; ++t){
				for(unsigned int l = 0; l < fac; ++l){
					cv::Vec3b& color = img.at<cv::Vec3b>((elementInY - iY - 1) * fac + t,iX * fac + l);
					color[0] = r * 255;
					color[1] = g * 255;
					color[2] = b_ * 255;
				}
			}
			++counter;
			++iY;
		}
		++iX;
	}
	cv::Scalar black(0,0,0);
	for(ClassDataConstIterator it = data.cbegin(); it != data.cend(); ++it, ++counter){
		const int dx = ((**it)[x] - min[0]) / (max[0] - min[0]) * elementInX * fac;
		const int dy = (1. - ((**it)[y] - min[1]) / (max[1] - min[1])) * elementInY * fac;
		const int label = (*it)->getLabel();
		double r, g, b;
		ColorConverter::LAB2RGB(colors[label][0], colors[label][1], colors[label][2], r, g, b);
		cv::Scalar actColor(r * 255,g * 255,b * 255);
		cv::circle(img, cv::Point(dx, dy), fac / 2 * amountOfPointsOnOneAxis / 50, actColor, CV_FILLED, CV_AA);
		cv::circle(img, cv::Point(dx, dy), fac / 2 * amountOfPointsOnOneAxis / 50, black, fac / 6 * amountOfPointsOnOneAxis / 50, CV_AA);
	}
	cv::imwrite(fileName, img);
	for(unsigned int i = 0; i < points.size(); ++i){
		delete points[i];
	}
}

void DataWriterForVisu::drawSvgDataPoints(std::ofstream& file, const ClassData& points,
		const Eigen::Vector2d& min, const Eigen::Vector2d& max, const Eigen::Vector2i& dim, const std::list<int>& selectedInducingPoints, const int amountOfClasses, const IVM* ivm){
	unsigned int counter = 0;
	if(selectedInducingPoints.size() > 0){
		std::list<ClassDataConstIterator> inducedPoints;
		for(ClassDataConstIterator it = points.cbegin(); it != points.cend(); ++it, ++counter){
			bool isInduced = false;
			for(std::list<int>::const_iterator itI = selectedInducingPoints.begin(); itI != selectedInducingPoints.end(); ++itI){
				if((*itI) == counter){
					isInduced = true; break;
				}
			}
			if(isInduced){
				inducedPoints.push_back(it);
				continue;
			}
			const double dx = ((**it)[dim[0]] - min[0]) / (max[0] - min[0]) * 100.;
			const double dy = ((**it)[dim[1]] - min[1]) / (max[1] - min[1]) * 100.;
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
		for(std::list<ClassDataConstIterator>::const_iterator it = inducedPoints.cbegin(); it != inducedPoints.cend(); ++it){
			const double dx = ((***it)[dim[0]] - min[0]) / (max[0] - min[0]) * 100.;
			const double dy = ((***it)[dim[1]] - min[1]) / (max[1] - min[1]) * 100.;
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
		for(ClassDataConstIterator it = points.cbegin(); it != points.cend(); ++it, ++counter){
			const double dx = ((**it)[dim[0]] - min[0]) / (max[0] - min[0]) * 100.;
			const double dy = ((**it)[dim[1]] - min[1]) / (max[1] - min[1]) * 100.;
			if(amountOfClasses == 2){
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
			}else{
				double r, g, b;
				//std::cout << "(*it)->getLabel(): " << (*it)->getLabel() + 1 << " " << amountOfClasses << ",";
				ColorConverter::HSV2RGB(((*it)->getLabel() + 1) / (double) (amountOfClasses) * 360., 1.0, 1.0, r, g, b);
				file << "<circle cx=\"" << dx << "%\" cy=\"" << dy << "%\" r=\"8\" fill=\"rgb(" << r * 100. << "%," << g * 100. << "%," << b * 100. << "%)\" stroke=\"black\" stroke-width=\"2\"/> \n";
			}
		}
	}
}

void DataWriterForVisu::writeSvg(const std::string& fileName, const Eigen::MatrixXd& mat){
	std::ofstream file;
	openSvgFile(fileName, 1920, mat.cols(), mat.rows(), file);
	double min, max;
	const bool ignoreDBLMAXNEG = true;
	DataConverter::getMinMax(mat, min, max, ignoreDBLMAXNEG);
	for(int iX = 0; iX < mat.rows(); ++iX){
		for(int iY = 0; iY < mat.cols(); ++iY){
			if(mat(iX,iY) > -DBL_MAX){
				const double prob = (mat(iX,iY)  - min) / (max - min);
				double r,g,b;
				ColorConverter::HSV2RGB(prob * 360.,1.0, 1.0,r,g,b);
				drawSvgRect(file, iX / (double)mat.rows() * 100., iY / (double)mat.cols()  * 100., 100.0 / mat.rows(), 100.0 / mat.cols(), (r * 100.0), (g * 100.0), (b * 100.0));
			}else{
				drawSvgRect(file, iX / (double)mat.rows() * 100., iY / (double)mat.cols()  * 100., 100.0 / mat.rows(), 100.0 / mat.cols(), 10, 10, 10);
			}
		}
	}
	closeSvgFile(file);
}

void DataWriterForVisu::writeSvg(const std::string& fileName, const std::list<double>& list, const std::list<std::string>& colors){
	if(list.size() == 0 || list.size() != colors.size()){
		return;
	}
	Eigen::VectorXd vec(list.size());
	unsigned int t = 0;
	for(std::list<double>::const_iterator it = list.begin(); it != list.end(); ++it, ++t){
		vec[t] = *it;
	}
	double min, max;
	DataConverter::getMinMax(vec, min, max);
	std::ofstream file;
	openSvgFile(fileName, 820., 1.0, 1.0, file);
	drawSvgCoords(file, 7.5, 7.5, 10, 10, vec.size(), max - min, min, max, 820., 820.);
	drawSvgDots(file, vec, 10., 10., min, max, 820., 820., colors);
	closeSvgFile(file);
}

void DataWriterForVisu::writeSvg(const std::string& fileName, const Eigen::VectorXd& vec, const bool drawLine){
	if(vec.rows() == 0){
		return;
	}
	double min, max;
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

void DataWriterForVisu::writeSvg(const std::string& fileName, const std::list<Eigen::VectorXd>& vecs, const bool drawLine){
	if(vecs.size() == 0){
		return;
	}
	const int size = vecs.begin()->rows();
	for(std::list<Eigen::VectorXd>::const_iterator it = vecs.begin(); it != vecs.end(); ++it){
		if(size != it->rows()){
			return;
		}
	}
	double min = DBL_MAX, max = -DBL_MAX;
	for(std::list<Eigen::VectorXd>::const_iterator it = vecs.begin(); it != vecs.end(); ++it){
		double minAct, maxAct;
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
	std::vector<std::string>::const_iterator itColor = colors.begin();
	if(drawLine){
		for(std::list<Eigen::VectorXd>::const_iterator it = vecs.begin(); it != vecs.end(); ++it){
			drawSvgLine(file, *it, 10., 10., min, max, 820., 820., *itColor);
			++itColor;
			if(itColor == colors.end())
				itColor = colors.begin();
		}
	}else{
		for(std::list<Eigen::VectorXd>::const_iterator it = vecs.begin(); it != vecs.end(); ++it){
			drawSvgDots(file, *it, 10., 10., min, max, 820., 820.,  *itColor);
			++itColor;
			if(itColor == colors.end())
				itColor = colors.begin();
		}
	}
	closeSvgFile(file);
}


void DataWriterForVisu::writeSvg(const std::string& fileName, const std::list<double>& list, const bool drawLine){
	if(list.size() == 0){
		return;
	}
	Eigen::VectorXd vec(list.size());
	unsigned int i = 0;
	for(std::list<double>::const_iterator it = list.begin(); it != list.end(); ++it, ++i){
		vec[i] = *it;
	}
	writeSvg(fileName, vec, drawLine);
}

void DataWriterForVisu::drawSvgCoords(std::ofstream& file,
		const double startX, const double startY, const double startXForData, const double startYForData, const double xSize,
		const double ySize, const double min, const double max, const double width, const double heigth, const bool useAllXSegments){
	file << "<path d=\"M " << startX / 100. * width << " "<< startY / 100. * heigth
		 << " l " << (100. - 2. * startX) / 100. * width  << " 0"
		 << " M " << startX / 100. * width << " "<< startY / 100. * heigth
		 << " l 0 " << (100. - 2. * startY) / 100. * heigth
		 << "\" fill=\"transparent\" stroke=\"black\"/> \n";
	const double widthOfMarks = 8;
	int amountOfSegm = useAllXSegments ? xSize - 1 : std::min(10, (int) xSize - 1);
	double segmentWidth = ((100 - startXForData - startXForData) / 100. * width) / amountOfSegm;
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
			 << (int)((xSize - 1) / amountOfSegm * i) + 1 << "</text>\n";
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
			 << number2String(((max - min) * i / (double) amountOfSegm + min), 5) << "</text>\n";
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

void DataWriterForVisu::drawSvgLine(std::ofstream& file, const Eigen::VectorXd vec,
		const double startX, const double startY, const double min,
		const double max, const double width, const double heigth, const std::string& color){
	if(vec.rows() > 0){
		const double diff = max == min ? 1 : max - min;
		file << "<path d=\"M " << (0 / (double) (vec.rows() - 1) * (100. - 2. * startX) + startX) / 100. * width << " "
				 	 	 	   << ((vec[0] - min) / diff * (100. - 2. * startY) + startY) / 100. * heigth << " \n";
		for(unsigned int i = 1; i < vec.rows(); ++i){
			file << " L "<< (i / (double) (vec.rows() - 1) * (100. - 2. * startX) + startX) / 100. * width << " "
						 << ((vec[i] - min) / diff * (100. - 2. * startY) + startY) / 100. * heigth << " \n";
		}
		file << "\" fill=\"transparent\" stroke=\"" << color << "\"/> \n";
	}
}

void DataWriterForVisu::drawSvgDots(std::ofstream& file, const Eigen::VectorXd vec,
		const double startX, const double startY, const double min,
		const double max, const double width, const double heigth, const std::string& color){
	const double diff = max == min ? 1 : max - min;
	for(unsigned int i = 0; i < vec.rows(); ++i){
		file << "<circle cx=\"" << (i / (double) vec.rows() * (100. - 2. * startX) + startX) / 100. * width
				<< "\" cy=\"" << ((vec[i] - min) / diff * (100. - 2. * startY) + startY) / 100. * heigth
				<< "\" r=\"3\" fill=\"transparent\" stroke=\"" << color << "\" /> \n";
	}
}

void DataWriterForVisu::drawSvgDots(std::ofstream& file, const Eigen::VectorXd vec,
		const double startX, const double startY, const double min,
		const double max, const double width, const double heigth, const std::list<std::string>& colors){
	const double diff = max == min ? 1 : max - min;
	std::list<std::string>::const_iterator it = colors.begin();
	for(unsigned int i = 0; i < vec.rows(); ++i){
		file << "<circle cx=\"" << (i / (double) vec.rows() * (100. - 2. * startX) + startX) / 100. * width
				<< "\" cy=\"" << ((vec[i] - min) / diff * (100. - 2. * startY) + startY) / 100. * heigth
				<< "\" r=\"3\" fill=\"transparent\" stroke=\"" << *it << "\" /> \n";
		++it;
	}
}

bool DataWriterForVisu::openSvgFile(const std::string& fileName, const double width, const double problemWidth, const double problemHeight, std::ofstream& file){
	file.open(fileName);
	const double height = (width / problemHeight *  problemWidth);
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

void DataWriterForVisu::drawSvgRect(std::ofstream& file, const double xPos, const double yPos,
			const double width, const double height,
			const int r, const int g, const int b){
	file << "<rect x=\""<< xPos <<"%\" y=\""<< yPos << "%\" width=\"" << width << "%\" height=\""
						<< height << "%\" fill=\"rgb(" << r << "%," << g << "%," << b << "%)\" /> \n";
}

