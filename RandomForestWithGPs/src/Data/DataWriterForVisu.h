/*
 * DataWriterForVisu.h
 *
 *  Created on: 04.06.2016
 *      Author: Max
 */

#ifndef DATA_DATAWRITERFORVISU_H_
#define DATA_DATAWRITERFORVISU_H_

#include "../Base/Predictor.h"
#include "../Data/ClassData.h"
#include <string>
#include "../GaussianProcess/IVM.h"

class DataWriterForVisu{
public:

	static void writeData(const std::string& fileName, const ClassData& data, const int x = 0, const int y = 1);

	static void generateGrid(const std::string& fileName,  const PredictorBinaryClass* predictor,
			const double amountOfPointsOnOneAxis, const ClassData& dataForMinMax, const int x = 0, const int y = 1);

	static void generateGrid(const std::string& fileName, const PredictorMultiClass* predictor,
			const double amountOfPointsOnOneAxis, const ClassData& dataForMinMax, const int x = 0, const int y = 1);

	static void writeSvg(const std::string& fileName, const PredictorBinaryClass* predictor,
			const ClassData& dataForMinMax, const int x = 0, const int y = 1);

	static void writeSvg(const std::string& fileName, const PredictorMultiClass* predictor,
			const ClassData& dataForMinMax, const int x = 0, const int y = 1);

	static void writeImg(const std::string& fileName, const PredictorMultiClass* predictor,
			const ClassData& dataForMinMax, const int x = 0, const int y = 1);

	static void writeSvg(const std::string& fileName, const IVM& ivm, const std::list<unsigned int>& selectedInducingPoints,
			const ClassData& data, const int x = 0, const int y = 1, const int type = 0);

	static void writeSvg(const std::string& fileName, const ClassData& data, const int x = 0, const int y = 1);

	static void writeSvg(const std::string& fileName, const Eigen::MatrixXd& mat);

	static void writeSvg(const std::string& fileName, const Eigen::VectorXd& vec, const bool drawLine = false);

	static void writeSvg(const std::string& fileName, const std::list<double>& vec, const bool drawLine = false);

	static void writeSvg(const std::string& fileName, const std::list<Eigen::VectorXd>& vec, const bool drawLine = false);

	static void writeSvg(const std::string& fileName, const std::list<double>& vec, const std::list<std::string>& colors);

	static void writePointsIn2D(const std::string& fileName, const std::list<Eigen::Vector2d>& points, const std::list<double>& values);

	static void writeHisto(const std::string&fileName, const std::list<double>& list, const unsigned int nrOfBins = 40, const double minValue = -1, const double maxValue = -1);

private:

	static void drawSvgDataPoints(std::ofstream& file, const ClassData& points,
			const Eigen::Vector2d& min, const Eigen::Vector2d& max,
			const Eigen::Vector2i& dim, const std::list<unsigned int>& selectedInducingPoints,
			const int amountOfClasses = 2, const IVM* ivm = nullptr);

	static void drawSvgLine(std::ofstream& file, const Eigen::VectorXd vec,
			const double startX, const double startY, const double min,
			const double max, const double width, const double heigth,
			const std::string& color = std::string("black"));

	static void drawSvgDots(std::ofstream& file, const Eigen::VectorXd vec,
			const double startX, const double startY, const double min,
			const double max, const double width, const double heigth, const std::string& color);

	static void drawSvgDots(std::ofstream& file, const Eigen::VectorXd vec,
			const double startX, const double startY, const double min,
			const double max, const double width, const double heigth, const std::list<std::string>& color);

	static bool openSvgFile(const std::string& fileName, const double width,
			const double problemWidth, const double problemHeight, std::ofstream& file);

	static void closeSvgFile(std::ofstream& file);

	static void drawSvgRect(std::ofstream& file, const double xPos, const double yPos,
			const double width, const double height,
			const int r, const int g, const int b);

	static void drawSvgCoords(std::ofstream& file,
			const double startX, const double startY, const double startXForData, const double startYForData, const double xSize,
			const double ySize, const double min, const double max, const double width, const double heigth, const bool useAllXSegments = false,
			const double minX = 0, const double maxX = 0);

	static void drawSvgCoords2D(std::ofstream& file,
			const double startX, const double startY, const double startXForData, const double startYForData, const Eigen::Vector2d& min,
			const Eigen::Vector2d& max, const unsigned int amountOfSegm, const double width, const double heigth);


	DataWriterForVisu();
	virtual ~DataWriterForVisu();

};

#endif /* DATA_DATAWRITERFORVISU_H_ */
