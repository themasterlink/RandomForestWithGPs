/*
 * DataWriterForVisu.h
 *
 *  Created on: 04.06.2016
 *      Author: Max
 */

#ifndef DATA_DATAWRITERFORVISU_H_
#define DATA_DATAWRITERFORVISU_H_

#include "../Base/Predictor.h"
#include "LabeledVectorX.h"
#include <string>
#include "../GaussianProcess/IVM.h"

class DataWriterForVisu {
public:

	static void writeData(const std::string& fileName, const LabeledData& data, const int x = 0, const int y = 1);

	static void generateGrid(const std::string& fileName,  const PredictorBinaryClass* predictor,
			const Real amountOfPointsOnOneAxis, const LabeledData& dataForMinMax, const int x = 0, const int y = 1);

	static void generateGrid(const std::string& fileName, const PredictorMultiClass* predictor,
			const Real amountOfPointsOnOneAxis, const LabeledData& dataForMinMax, const int x = 0, const int y = 1);

	static void writeSvg(const std::string& fileName, const PredictorBinaryClass* predictor,
			const LabeledData& dataForMinMax, const int x = 0, const int y = 1);

	static void writeSvg(const std::string& fileName, const PredictorMultiClass* predictor,
			const LabeledData& dataForMinMax, const int x = 0, const int y = 1);

	static void writeImg(const std::string& fileName, const PredictorBinaryClass* predictor,
			const LabeledData& dataForMinMax, const int x = 0, const int y = 1);

	static void writeImg(const std::string& fileName, const IVM* predictor,
			const LabeledData& dataForMinMax, const int x = 0, const int y = 1);

	static void writeImg(const std::string& fileName, const PredictorMultiClass* predictor,
			const LabeledData& dataForMinMax, const int x = 0, const int y = 1);

	static void writeSvg(const std::string& fileName, const IVM& ivm, const std::list<unsigned int>& selectedInducingPoints,
			const LabeledData& data, const int x = 0, const int y = 1, const int type = 0);

	static void writeSvg(const std::string& fileName, const LabeledData& data, const int x = 0, const int y = 1);

	static void writeSvg(const std::string& fileName, const Matrix& mat);

	static void writeSvg(const std::string& fileName, const VectorX& vec, const bool drawLine = false);

	static void writeSvg(const std::string& fileName, const std::list<Real>& vec, const bool drawLine = false);

	static void writeSvg(const std::string& fileName, const std::list<VectorX>& vec, const bool drawLine = false);

	static void writeSvg(const std::string& fileName, const std::list<Real>& vec, const std::list<std::string>& colors);

	static void writePointsIn2D(const std::string& fileName, const std::list<Vector2>& points, const std::list<Real>& values);

	static void writeHisto(const std::string&fileName, const std::list<Real>& list, const unsigned int nrOfBins = 40, const Real minValue = -1, const Real maxValue = -1);

private:

	static void drawSvgDataPoints(std::ofstream& file, const LabeledData& points,
			const Vector2& min, const Vector2& max,
			const Vector2i& dim, const std::list<unsigned int>& selectedInducingPoints,
			const int amountOfClasses = 2, const IVM* ivm = nullptr);

	static void drawSvgLine(std::ofstream& file, const VectorX vec,
			const Real startX, const Real startY, const Real min,
			const Real max, const Real width, const Real heigth,
			const std::string& color = std::string("black"));

	static void drawSvgDots(std::ofstream& file, const VectorX vec,
			const Real startX, const Real startY, const Real min,
			const Real max, const Real width, const Real heigth, const std::string& color);

	static void drawSvgDots(std::ofstream& file, const VectorX vec,
			const Real startX, const Real startY, const Real min,
			const Real max, const Real width, const Real heigth, const std::list<std::string>& color);

	static bool openSvgFile(const std::string& fileName, const Real width,
			const Real problemWidth, const Real problemHeight, std::ofstream& file);

	static void closeSvgFile(std::ofstream& file);

	static void drawSvgRect(std::ofstream& file, const Real xPos, const Real yPos,
			const Real width, const Real height,
			const int r, const int g, const int b);

	static void drawSvgCoords(std::ofstream& file,
			const Real startX, const Real startY, const Real startXForData, const Real startYForData, const Real xSize,
			const Real ySize, const Real min, const Real max, const Real width, const Real heigth, const bool useAllXSegments = false,
			const Real minX = 0, const Real maxX = 0);

	static void drawSvgCoords2D(std::ofstream& file,
			const Real startX, const Real startY, const Real startXForData, const Real startYForData, const Vector2& min,
			const Vector2& max, const unsigned int amountOfSegm, const Real width, const Real heigth);


	DataWriterForVisu();
	~DataWriterForVisu();

};

#endif /* DATA_DATAWRITERFORVISU_H_ */
