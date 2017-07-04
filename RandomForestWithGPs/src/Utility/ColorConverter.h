/*
 * ColorConverter.h
 *
 *  Created on: 28.09.2016
 *      Author: Max
 */

#ifndef UTILITY_COLORCONVERTER_H_
#define UTILITY_COLORCONVERTER_H_

#include "Util.h"

#ifdef USE_OPEN_CV
#include <opencv2/opencv.hpp>
#endif // USE_OPEN_CV

namespace ColorConverter {

	static const Real firstColor[3] = {1, 1, 0};
	static const Real secondColor[3] = {0, 0, 1};
	static const Real thirdColor[3] = {1, 0, 0};
	static const Real fourthColor[3] = {0, 1, 1};

	void openCVColorChange(const int code, const Real i, const Real j,
						   const Real k, Real& u, Real& v, Real& w){
#ifdef USE_OPEN_CV
		cv::Mat_<cv::Vec3f> start(cv::Vec3f(i, j, k));
		cv::Mat_<cv::Vec3f> end;
		cvtColor(start, end, code);
		u = (*end[0])[0];
		v = (*end[0])[1];
		w = (*end[0])[2];
#else
		printError("This function does not work without opencv!");
#endif // USE_OPEN_CV
	}

	void HSV2RGB(const Real h, const Real s, const Real v, Real& r,
				 Real& g, Real& b){
#ifdef USE_OPEN_CV
		openCVColorChange(cv::COLOR_HSV2RGB, h, s, v, r, g, b);
#else
		printError("This function does not work without opencv!");
#endif // USE_OPEN_CV
	}

	void RGB2XYZ(Real r, Real g, Real b, Real& x, Real& y, Real& z){
#ifdef USE_OPEN_CV
		openCVColorChange(cv::COLOR_RGB2XYZ, r, g, b, x, y, z);
#else
		printError("This function does not work without opencv!");
#endif // USE_OPEN_CV
	}

	void XYZ2RGB(const Real x, const Real y, const Real z, Real& r,
				 Real& g, Real& b){
#ifdef USE_OPEN_CV
		openCVColorChange(cv::COLOR_XYZ2RGB, x, y, z, r, g, b);
#else
		printError("This function does not work without opencv!");
#endif // USE_OPEN_CV
	}

	void RGB2LAB(const Real r, const Real g, const Real b, Real& l,
				 Real& a, Real& b_){
#ifdef USE_OPEN_CV
		openCVColorChange(cv::COLOR_RGB2Lab, r, g, b, l, a, b_);
#else
		printError("This function does not work without opencv!");
#endif // USE_OPEN_CV
	}

	void LAB2RGB(const Real l, const Real a, const Real b, Real& r,
				 Real& g, Real& b_){
#ifdef USE_OPEN_CV
		openCVColorChange(cv::COLOR_Lab2RGB, l, a, b, r, g, b_);
#else
		printError("This function does not work without opencv!");
#endif // USE_OPEN_CV
	}

	void HSV2LAB(const Real h, const Real s, const Real v, Real& l,
				 Real& a, Real& b){
		Real r, g;
		HSV2RGB(h, s, v, r, g, b);
		RGB2LAB(r, g, b, l, a, b);
	}

	void getProbColorForBinaryRGB(const Real prob, Real& r, Real& g, Real& b){
		r = prob * firstColor[0] + (1 - prob) * secondColor[0];
		g = prob * firstColor[1] + (1 - prob) * secondColor[1];
		b = prob * firstColor[2] + (1 - prob) * secondColor[2];
	}

	void getProbColorForSecondBinaryRGB(const Real prob, Real& r, Real& g, Real& b){
		r = prob * thirdColor[0] + (1 - prob) * fourthColor[0];
		g = prob * thirdColor[1] + (1 - prob) * fourthColor[1];
		b = prob * thirdColor[2] + (1 - prob) * fourthColor[2];
	}


}

#endif /* UTILITY_COLORCONVERTER_H_ */
