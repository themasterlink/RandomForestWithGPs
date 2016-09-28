/*
 * DataWriterForVisu.cc
 *
 *  Created on: 04.06.2016
 *      Author: Max
 */

#include "DataWriterForVisu.h"

DataWriterForVisu::DataWriterForVisu()
{
	// TODO Auto-generated constructor stub

}

DataWriterForVisu::~DataWriterForVisu()
{
	// TODO Auto-generated destructor stub
}

void DataWriterForVisu::writeData(const std::string& fileName, const Data& data, const Labels& labels,
		const int x, const int y){
	if(data.size() > 0){
		if(!(data[0].rows() > x && data[0].rows() > y && x != y && y >= 0 && x >= 0)){
			printError("These axis x: " << x << ", y: " << y << " aren't printable!");
			return;
		}
		std::ofstream file;
		file.open(fileName);
		if(file.is_open()){
			int i = 0;
			for(Data::const_iterator it = data.cbegin(); it != data.cend(); ++it){
				file << (*it)[x] << " " << (*it)[y] << " " << labels[i++] << "\n";
			}
		}
		file.close();
	}else{
		printError("No data -> no writting!");
	}
}

void DataWriterForVisu::generateGrid(const std::string& fileName, const RandomForest& forest,
		const double amountOfPointsOnOneAxis, const Data& data,
		const int x, const int y){
	if(data.size() == 0){
		printError("No data is given, this data is needed to find min and max!");
		return;
	}
	const int dim = data[0].rows();
	Eigen::Vector2d dimVec;
	dimVec << x,y;
	Eigen::Vector2d min, max;
	for(int i = 0; i < 2; ++i){
		min[i] = 1000000;
		max[i] = -1000000;
	}
	for(Data::const_iterator it = data.cbegin(); it != data.cend(); ++it){
		for(int i = 0; i < 2; ++i){
			int j = dimVec[i];
			if(min[i] > (*it)[j]){
				min[i] = (*it)[j];
			}
			if(max[i] < (*it)[j]){
				max[i] = (*it)[j];
			}
		}
	}
	Eigen::Vector2d stepSize = (1. / amountOfPointsOnOneAxis) * (max - min);
	std::ofstream file;
	file.open(fileName);
	Data points;
	points.reserve(amountOfPointsOnOneAxis * (amountOfPointsOnOneAxis + 1));
	int amount = 0;
	for(double xVal = max[0]; xVal >= min[0]; xVal -= stepSize[0]){
		for(double yVal = min[1]; yVal < max[1]; yVal+= stepSize[1]){
			DataElement ele(dim);
			for(int i = 0; i < dim; ++i){
				if(i == x){
					ele[i] = xVal;
				}else if(i == y){
					ele[i] = yVal;
				}else{
					ele[i] = 0;
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
	}
	file.close();
}

void DataWriterForVisu::generateGrid(const std::string& fileName, const RandomForestGaussianProcess& rfgp,
		const double amountOfPointsOnOneAxis, const Data& data,
		const int x, const int y){
	if(data.size() == 0){
		printError("No data is given, this data is needed to find min and max!");
		return;
	}
	const int dim = data[0].rows();
	Eigen::Vector2d dimVec;
	dimVec << x,y;
	Eigen::Vector2d min, max;
	for(int i = 0; i < 2; ++i){
		min[i] = 1000000;
		max[i] = -1000000;
	}
	for(Data::const_iterator it = data.cbegin(); it != data.cend(); ++it){
		for(int i = 0; i < 2; ++i){
			int j = dimVec[i];
			if(min[i] > (*it)[j]){
				min[i] = (*it)[j];
			}
			if(max[i] < (*it)[j]){
				max[i] = (*it)[j];
			}
		}
	}
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
			DataElement ele(dim);
			for(int i = 0; i < dim; ++i){
				if(i == x){
					ele[i] = xVal;
				}else if(i == y){
					ele[i] = yVal;
				}else{
					ele[i] = 0;
				}
			}
			points.push_back(ele);
			int val = rfgp.predict(ele, prob);
			labels.push_back(prob[val]);
			++amount;
		}
	}
	for(int i = 0; i < amount; ++i){
		file << points[i][0] << " " << points[i][1] << " " << labels[i] << "\n";
	}
	file.close();
}



void DataWriterForVisu::generateGrid(const std::string& fileName, const GaussianProcess& gp,
		const double amountOfPointsOnOneAxis, const Data& data,
		const int x, const int y){
	if(data.size() == 0){
		printError("No data is given, this data is needed to find min and max!");
		return;
	}
	const int dim = data[0].rows();
	Eigen::Vector2d dimVec;
	dimVec << x,y;
	Eigen::Vector2d min, max;
	for(int i = 0; i < 2; ++i){
		min[i] = 1000000;
		max[i] = -1000000;
	}
	for(Data::const_iterator it = data.cbegin(); it != data.cend(); ++it){
		for(int i = 0; i < 2; ++i){
			int j = dimVec[i];
			if(min[i] > (*it)[j]){
				min[i] = (*it)[j];
			}
			if(max[i] < (*it)[j]){
				max[i] = (*it)[j];
			}
		}
	}
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
			DataElement ele(dim);
			for(int i = 0; i < dim; ++i){
				if(i == x){
					ele[i] = xVal;
				}else if(i == y){
					ele[i] = yVal;
				}else{
					ele[i] = 0;
				}
			}
			points.push_back(ele);
			labels.push_back(gp.predict(ele));
			++amount;
		}
	}
	for(int i = 0; i < amount; ++i){
		file << points[i][0] << " " << points[i][1] << " " << labels[i] << "\n";
	}
	file.close();
}

void DataWriterForVisu::writeSvg(const std::string& fileName, const GaussianProcess& gp,
		const double amountOfPointsOnOneAxis, const Data& data, const int x, const int y){
	if(data.size() == 0){
		printError("No data is given, this data is needed to find min and max!");
		return;
	}
	const int dim = data[0].rows();
	Eigen::Vector2d dimVec;
	dimVec << x,y;
	Eigen::Vector2d min, max;
	for(int i = 0; i < 2; ++i){
		min[i] = 1000000;
		max[i] = -1000000;
	}
	for(Data::const_iterator it = data.cbegin(); it != data.cend(); ++it){
		for(int i = 0; i < 2; ++i){
			int j = dimVec[i];
			if(min[i] > (*it)[j]){
				min[i] = (*it)[j];
			}
			if(max[i] < (*it)[j]){
				max[i] = (*it)[j];
			}
		}
	}
	const Eigen::Vector2d diff = max - min;
	max[0] += diff[0] * 0.2;
	max[1] += diff[1] * 0.2;
	min[0] -= diff[0] * 0.2;
	min[1] -= diff[1] * 0.2;
	Eigen::Vector2d stepSize = (1. / amountOfPointsOnOneAxis) * (max - min);
	std::ofstream file;
	file.open(fileName);
	int amount = 0;
	std::vector<double> labels;
	const double elementInX = (int)((max[0] - min[0]) / stepSize[0]);
	const double elementInY = (int)((max[1] - min[1]) / stepSize[1]);
	int iX = 0, iY;
	file << "<svg version=\"1.1\" " <<
			"\nbaseProfile=\"full\"" <<
			"\nwidth=\"" << 1920 << "\" height=\""<< (int) (1920. / (max[0] - min[0]) *  (max[1] - min[1]))   << "\"\n" <<
			"xmlns=\"http://www.w3.org/2000/svg\">" << "\n";
	for(double xVal = max[0]; xVal >= min[0]; xVal -= stepSize[0]){
		iY = 0;
		for(double yVal = min[1]; yVal < max[1]; yVal+= stepSize[1]){
			DataElement ele(dim);
			for(int i = 0; i < dim; ++i){
				if(i == x){
					ele[i] = xVal;
				}else if(i == y){
					ele[i] = yVal;
				}else{
					ele[i] = 0;
				}
			}
			double val = fmax(fmin(1.0,gp.predict(ele)), 0.0);
			file << "<rect x=\""<< iX / elementInX * 100. <<"%\" y=\""<< iY / elementInY  * 100.<< "%\" width=\"" << 100.0 / elementInX << "%\" height=\""
					<< 100.0 / elementInY << "%\" fill=\"rgb("
					<< (int)(val * 100.0) << "%,0%," << (int)((1.0-val) * 100.0)<< "%)\" /> \n";
			++amount;
			++iY;
		}
		++iX;
	}
	for(Data::const_iterator it = data.cbegin(); it != data.cend(); ++it){
		const double dx = ((*it)[x] - min[0]) / (max[0] - min[0]) * 100.;
		const double dy = ((*it)[y] - min[1]) / (max[1] - min[1]) * 100.;
		file << "<circle cx=\"" << dx << "%\" cy=\"" << dy << "%\" r=\"20\" fill=\"green\" /> \n";
	}
	file << "</svg>\n";
	file.close();
}


void DataWriterForVisu::writeSvg(const std::string& fileName, const GaussianProcessMultiBinary& gp,
		const double amountOfPointsOnOneAxis, const Data& data, const int x, const int y){
	if(data.size() == 0){
		printError("No data is given, this data is needed to find min and max!");
		return;
	}
	const int dim = data[0].rows();
	Eigen::Vector2d dimVec;
	dimVec << x,y;
	Eigen::Vector2d min, max;
	for(int i = 0; i < 2; ++i){
		min[i] = 1000000;
		max[i] = -1000000;
	}
	for(Data::const_iterator it = data.cbegin(); it != data.cend(); ++it){
		for(int i = 0; i < 2; ++i){
			int j = dimVec[i];
			if(min[i] > (*it)[j]){
				min[i] = (*it)[j];
			}
			if(max[i] < (*it)[j]){
				max[i] = (*it)[j];
			}
		}
	}
	const Eigen::Vector2d diff = max - min;
	max[0] += diff[0] * 0.2;
	max[1] += diff[1] * 0.2;
	min[0] -= diff[0] * 0.2;
	min[1] -= diff[1] * 0.2;
	Eigen::Vector2d stepSize = (1. / amountOfPointsOnOneAxis) * (max - min);
	std::ofstream file;
	file.open(fileName);
	int amount = 0;
	std::vector<double> labels;
	const double elementInX = (int)((max[0] - min[0]) / stepSize[0]);
	const double elementInY = (int)((max[1] - min[1]) / stepSize[1]);
	int iX = 0, iY;
	file << "<svg version=\"1.1\" " <<
			"\nbaseProfile=\"full\"" <<
			"\nwidth=\"" << 1920 << "\" height=\""<< (int) (1920. / (max[0] - min[0]) *  (max[1] - min[1]))   << "\"\n" <<
			"xmlns=\"http://www.w3.org/2000/svg\">" << "\n";
	for(double xVal = max[0]; xVal >= min[0]; xVal -= stepSize[0]){
		iY = 0;
		for(double yVal = min[1]; yVal < max[1]; yVal+= stepSize[1]){
			DataElement ele(dim);
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
			file << "<rect x=\""<< iX / elementInX * 100. <<"%\" y=\""<< iY / elementInY  * 100.<< "%\" width=\"" << 100.0 / elementInX << "%\" height=\""
					<< 100.0 / elementInY << "%\" fill=\"rgb("
					<< (int)(prob[0] * 100.0) << "%,0%," << (int)((1.0-prob[1]) * 100.0)<< "%)\" /> \n";
			++amount;
			++iY;
		}
		++iX;
	}
	for(Data::const_iterator it = data.cbegin(); it != data.cend(); ++it){
		const double dx = ((*it)[x] - min[0]) / (max[0] - min[0]) * 100.;
		const double dy = ((*it)[y] - min[1]) / (max[1] - min[1]) * 100.;
		file << "<circle cx=\"" << dx << "%\" cy=\"" << dy << "%\" r=\"20\" fill=\"green\" /> \n";
	}
	file << "</svg>\n";
	file.close();
}

void DataWriterForVisu::writeSvg(const std::string& fileName, const Eigen::MatrixXd mat){
	std::ofstream file;
	file.open(fileName);
	file << "<svg version=\"1.1\" " <<
				"\nbaseProfile=\"full\"" <<
				"\nwidth=\"" << 1920 << "\" height=\""<< (int) (1920. / (mat.cols()) *  (mat.rows()))   << "\"\n" <<
				"xmlns=\"http://www.w3.org/2000/svg\">" << "\n";
	double max = -DBL_MAX;
	double min = DBL_MAX;
	for(int iX = 0; iX < mat.cols(); ++iX){
		for(int iY = 0; iY < mat.rows(); ++iY){
			if(mat(iX,iY) < min && mat(iX,iY) > -DBL_MAX){
				min = mat(iX,iY);
			}
			if(mat(iX,iY) > max){
				max = mat(iX,iY);
			}
		}
	}

	for(int iX = 0; iX < mat.cols(); ++iX){
		for(int iY = 0; iY < mat.rows(); ++iY){
			if(mat(iX,iY) > -DBL_MAX){
				const double prob = (mat(iX,iY)  - min) / (max - min);
				file << "<rect x=\""<< iX / (double)mat.cols() * 100. <<"%\" y=\""<< iY / (double)mat.rows()  * 100.<< "%\" width=\"" << 100.0 / mat.cols() << "%\" height=\""
						<< 100.0 / mat.rows() << "%\" fill=\"rgb("
						<< (int)(prob * 100.0) << "%,0%," << (int)((1.0-prob) * 100.0)<< "%)\" /> \n";
			}else{
				file << "<rect x=\""<< iX / (double)mat.cols() * 100. <<"%\" y=\""<< iY / (double)mat.rows()  * 100.<< "%\" width=\"" << 100.0 / mat.cols() << "%\" height=\""
						<< 100.0 / mat.rows() << "%\" fill=\"rgb(0%,100%,0%)\" /> \n";
			}
		}
	}
	file << "</svg>\n";
	file.close();
}

