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

class DataWriterForVisu{
public:

	static void writeData(const std::string& fileName, const LabeledData& data, const int x = 0, const int y = 1);

	static void generateGrid(const std::string& fileName,  const PredictorBinaryClass* predictor,
			const real amountOfPointsOnOneAxis, const LabeledData& dataForMinMax, const int x = 0, const int y = 1);

	static void generateGrid(const std::string& fileName, const PredictorMultiClass* predictor,
			const real amountOfPointsOnOneAxis, const LabeledData& dataForMinMax, const int x = 0, const int y = 1);

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

	static void writeSvg(const std::string& fileName, const std::list<real>& vec, const bool drawLine = false);

	static void writeSvg(const std::string& fileName, const std::list<VectorX>& vec, const bool drawLine = false);

	static void writeSvg(const std::string& fileName, const std::list<real>& vec, const std::list<std::string>& colors);

	static void writePointsIn2D(const std::string& fileName, const std::list<Vector2>& points, const std::list<real>& values);

	static void writeHisto(const std::string&fileName, const std::list<real>& list, const unsigned int nrOfBins = 40, const real minValue = -1, const real maxValue = -1);

private:

	static void drawSvgDataPoints(std::ofstream& file, const LabeledData& points,
			const Vector2& min, const Vector2& max,
			const Vector2i& dim, const std::list<unsigned int>& selectedInducingPoints,
			const int amountOfClasses = 2, const IVM* ivm = nullptr);

	static void drawSvgLine(std::ofstream& file, const VectorX vec,
			const real startX, const real startY, const real min,
			const real max, const real width, const real heigth,
			const std::string& color = std::string("black"));

	static void drawSvgDots(std::ofstream& file, const VectorX vec,
			const real startX, const real startY, const real min,
			const real max, const real width, const real heigth, const std::string& color);

	static void drawSvgDots(std::ofstream& file, const VectorX vec,
			const real startX, const real startY, const real min,
			const real max, const real width, const real heigth, const std::list<std::string>& color);

	static bool openSvgFile(const std::string& fileName, const real width,
			const real problemWidth, const real problemHeight, std::ofstream& file);

	static void closeSvgFile(std::ofstream& file);

	static void drawSvgRect(std::ofstream& file, const real xPos, const real yPos,
			const real width, const real height,
			const int r, const int g, const int b);

	static void drawSvgCoords(std::ofstream& file,
			const real startX, const real startY, const real startXForData, const real startYForData, const real xSize,
			const real ySize, const real min, const real max, const real width, const real heigth, const bool useAllXSegments = false,
			const real minX = 0, const real maxX = 0);

	static void drawSvgCoords2D(std::ofstream& file,
			const real startX, const real startY, const real startXForData, const real startYForData, const Vector2& min,
			const Vector2& max, const unsigned int amountOfSegm, const real width, const real heigth);


	DataWriterForVisu();
	virtual ~DataWriterForVisu();

};

#endif /* DATA_DATAWRITERFORVISU_H_ */
