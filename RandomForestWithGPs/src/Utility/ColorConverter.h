/*
 * ColorConverter.h
 *
 *  Created on: 28.09.2016
 *      Author: Max
 */

#ifndef UTILITY_COLORCONVERTER_H_
#define UTILITY_COLORCONVERTER_H_

#include "Util.h"
#include <opencv2/opencv.hpp>

namespace ColorConverter{

static const double firstColor[3] = {1,1,0};
static const double secondColor[3] = {0,0,1};
static const double thirdColor[3] = {1,0,0};
static const double fourthColor[3] = {0,1,1};

void openCVColorChange(const int code, const double i, const double j,
		const double k, double& u, double& v, double& w) {
	cv::Mat_<cv::Vec3f> start(cv::Vec3f(i, j, k));
	cv::Mat_<cv::Vec3f> end;
	cvtColor(start, end, code);
	u = (*end[0])[0];
	v = (*end[0])[1];
	w = (*end[0])[2];
}

void HSV2RGB(const double h, const double s, const double v, double& r,
		double &g, double& b) {
	openCVColorChange(cv::COLOR_HSV2RGB, h, s, v, r, g, b);
}

void RGB2XYZ(double r, double g, double b, double& x, double& y, double& z) {
	openCVColorChange(cv::COLOR_RGB2XYZ, r, g, b, x, y, z);
}

void XYZ2RGB(const double x, const double y, const double z, double& r,
		double& g, double& b) {
	openCVColorChange(cv::COLOR_XYZ2RGB, x, y, z, r, g, b);
}

void RGB2LAB(const double r, const double g, const double b, double& l,
		double& a, double& b_) {
	openCVColorChange(cv::COLOR_RGB2Lab, r, g, b, l, a, b_);
}

void LAB2RGB(const double l, const double a, const double b, double& r,
		double& g, double& b_) {
	openCVColorChange(cv::COLOR_Lab2RGB, l, a, b, r, g, b_);
}

void HSV2LAB(const double h, const double s, const double v, double& l,
		double &a, double& b) {
	double r, g;
	HSV2RGB(h,s,v,r,g,b);
	RGB2LAB(r,g,b,l,a,b);
}

void getProbColorForBinaryRGB(const double prob, double& r, double& g, double& b){
	r = prob * firstColor[0] + (1-prob) * secondColor[0];
	g = prob * firstColor[1] + (1-prob) * secondColor[1];
	b = prob * firstColor[2] + (1-prob) * secondColor[2];
}

void getProbColorForSecondBinaryRGB(const double prob, double& r, double& g, double& b){
	r = prob * thirdColor[0] + (1-prob) * fourthColor[0];
	g = prob * thirdColor[1] + (1-prob) * fourthColor[1];
	b = prob * thirdColor[2] + (1-prob) * fourthColor[2];
}


}

#endif /* UTILITY_COLORCONVERTER_H_ */
