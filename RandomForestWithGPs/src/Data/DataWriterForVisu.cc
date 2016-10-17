/*
 * DataWriterForVisu.cc
 *
 *  Created on: 04.06.2016
 *      Author: Max
 */

#include "DataWriterForVisu.h"
#include "../Utility/ColorConverter.h"
#include "DataConverter.h"

DataWriterForVisu::DataWriterForVisu()
{
	// TODO Auto-generated constructor stub

}

DataWriterForVisu::~DataWriterForVisu()
{
	// TODO Auto-generated destructor stub
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

void DataWriterForVisu::generateGrid(const std::string& fileName, const RandomForest& forest,
		const double amountOfPointsOnOneAxis, const ClassData& data,
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
	Labels labels;
	forest.predictData(points, labels);
	for(int i = 0; i < amount; ++i){
		file << points[i][0] << " " << points[i][1] << " " << labels[i] << "\n";
		delete(points[i]);
	}
	file.close();
}

void DataWriterForVisu::generateGrid(const std::string& fileName, const RandomForestGaussianProcess& rfgp,
		const double amountOfPointsOnOneAxis, const ClassData& data,
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
	std::vector<double> labels;
	std::vector<double> prob(rfgp.amountOfClasses());
	for(double xVal = max[0]; xVal >= min[0]; xVal -= stepSize[0]){
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
			int val = rfgp.predict(*ele, prob);
			labels.push_back(prob[val]);
			++amount;
		}
	}
	for(int i = 0; i < amount; ++i){
		file << points[i][0] << " " << points[i][1] << " " << labels[i] << "\n";
		delete(points[i]);
	}
	file.close();
}

void DataWriterForVisu::generateGrid(const std::string& fileName, const GaussianProcess& gp,
		const double amountOfPointsOnOneAxis, const ClassData& data,
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
	const Eigen::Vector2d diff = max - min;
	max[0] += diff[0] * 0.2;
	max[1] += diff[1] * 0.2;
	min[0] -= diff[0] * 0.2;
	min[1] -= diff[1] * 0.2;
	Eigen::Vector2d stepSize = (1. / amountOfPointsOnOneAxis) * (max - min);
	std::ofstream file;
	file.open(fileName);
	Data points;
	points.reserve(amountOfPointsOnOneAxis * (amountOfPointsOnOneAxis + 1));
	int amount = 0;
	std::vector<double> labels;
	for(double xVal = max[0]; xVal >= min[0]; xVal -= stepSize[0]){
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
			labels.push_back(gp.predict(*ele));
			++amount;
		}
	}
	for(int i = 0; i < amount; ++i){
		file << points[i][0] << " " << points[i][1] << " " << labels[i] << "\n";
		delete(points[i]);
	}
	file.close();
}

void DataWriterForVisu::writeSvg(const std::string& fileName, const GaussianProcess& gp,
		const double amountOfPointsOnOneAxis, const ClassData& data, const int x, const int y){
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
	Eigen::Vector2d stepSize = (1. / amountOfPointsOnOneAxis) * (max - min);
	int amount = 0;
	std::vector<double> labels;
	const double elementInX = (int)((max[0] - min[0]) / stepSize[0]);
	const double elementInY = (int)((max[1] - min[1]) / stepSize[1]);
	int iX = 0, iY;
	std::ofstream file;
	openSvgFile(fileName, 820., (double) (max[0] - min[0]), (double) (max[1] - min[1]), file);
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
			double val = fmax(fmin(1.0,gp.predict(ele,50000)), 0.0);
			drawSvgRect(file, iX / elementInX * 100., iY / elementInY  * 100.,
					100.0 / elementInX, 100.0 / elementInY, val * 100, 0, (1-val) * 100);
			++amount;
			++iY;
		}
		++iX;
	}

	std::list<int> empty;
	drawSvgDataPoints(file, data, min, max, dimVec, empty);
	closeSvgFile(file);
}


void DataWriterForVisu::writeSvg(const std::string& fileName, const GaussianProcessMultiBinary& gp,
		const double amountOfPointsOnOneAxis, const ClassData& data, const int x, const int y){
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
	Eigen::Vector2d stepSize = (1. / amountOfPointsOnOneAxis) * (max - min);
	int amount = 0;
	std::vector<double> labels;
	const double elementInX = (int)((max[0] - min[0]) / stepSize[0]);
	const double elementInY = (int)((max[1] - min[1]) / stepSize[1]);
	int iX = 0, iY;
	std::ofstream file;
	openSvgFile(fileName, 820., (double) (max[0] - min[0]), (double) (max[1] - min[1]), file);
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
			std::vector<double> prob;
			gp.predict(ele, prob);
			//double val = fmax(fmin(1.0,), 0.0);
			drawSvgRect(file, iX / elementInX * 100., iY / elementInY  * 100.,
					100.0 / elementInX, 100.0 / elementInY, prob[0] * 100, 0, prob[1] * 100);
			++amount;
			++iY;
		}
		++iX;
	}
	std::list<int> empty;
	drawSvgDataPoints(file, data, min, max, dimVec, empty);
	closeSvgFile(file);
}


void DataWriterForVisu::writeHisto(const std::string&fileName, const std::list<double> list, const unsigned int nrOfBins){
	if(list.size() == 0){
		printError("No data is given!");
		return;
	}
	double min, max;
	DataConverter::getMinMax(list, min, max);
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
		const Eigen::VectorXd& labels, const double amountOfPointsOnOneAxis, const ClassData& data, const int x, const int y){
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
	Eigen::Vector2d stepSize = (1. / amountOfPointsOnOneAxis) * (max - min);
	int amount = 0;
	const double elementInX = (int)((max[0] - min[0]) / stepSize[0]);
	const double elementInY = (int)((max[1] - min[1]) / stepSize[1]);
	int iX = 0, iY;
	std::ofstream file;
	openSvgFile(fileName, 820., (double) (max[0] - min[0]), (double) (max[1] - min[1]), file);
	//for(double xVal = max[0]; xVal >= min[0]; xVal -= stepSize[0]){
	for(double xVal = min[0]; xVal < max[0]; xVal += stepSize[0]){
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
			double prob = ivm.predict(ele);
			//double val = fmax(fmin(1.0,), 0.0);
			drawSvgRect(file, iX / elementInX * 100., iY / elementInY  * 100.,
					100.0 / elementInX, 100.0 / elementInY, prob * 100, 0, (1-prob) * 100);
			++amount;
			++iY;
		}
		++iX;
	}
	drawSvgDataPoints(file, data, min, max, dimVec, selectedInducingPoints);
	closeSvgFile(file);
}

void DataWriterForVisu::drawSvgDataPoints(std::ofstream& file, const ClassData& points,
		const Eigen::Vector2d& min, const Eigen::Vector2d& max, const Eigen::Vector2i& dim, const std::list<int>& selectedInducingPoints){
	unsigned int counter = 0;
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
		if((*it)->getLabel() == 0){
			color = "red";
		}
		file << "<circle cx=\"" << dx << "%\" cy=\"" << dy << "%\" r=\"8\" fill=\"" << color << "\" stroke=\"black\" stroke-width=\"2\"/> \n";
	}
	// draw after all points, to make them more visible
	for(std::list<ClassDataConstIterator>::const_iterator it = inducedPoints.cbegin(); it != inducedPoints.cend(); ++it){
		const double dx = ((***it)[dim[0]] - min[0]) / (max[0] - min[0]) * 100.;
		const double dy = ((***it)[dim[1]] - min[1]) / (max[1] - min[1]) * 100.;
		std::string color = "blue";
		if((**it)->getLabel() == 0){
			color = "red";
		}
		file << "<circle cx=\"" << dx << "%\" cy=\"" << dy << "%\" r=\"8\" fill=\"" << color << "\" stroke=\"black\" stroke-width=\"2\"/> \n";
		file << "<circle cx=\"" << dx << "%\" cy=\"" << dy << "%\" r=\"3\" fill=\"white\" /> \n";
	}
}

void DataWriterForVisu::writeSvg(const std::string& fileName, const Eigen::MatrixXd mat){
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

void DataWriterForVisu::writeSvg(const std::string& fileName, const std::list<double> list, const std::list<std::string>& colors){
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

void DataWriterForVisu::writeSvg(const std::string& fileName, const Eigen::VectorXd vec, const bool drawLine){
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

void DataWriterForVisu::writeSvg(const std::string& fileName, const std::list<double> list, const bool drawLine){
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

