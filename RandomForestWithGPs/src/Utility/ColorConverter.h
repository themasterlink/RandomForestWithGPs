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

static const real firstColor[3] = {1,1,0};
static const real secondColor[3] = {0,0,1};
static const real thirdColor[3] = {1,0,0};
static const real fourthColor[3] = {0,1,1};

void openCVColorChange(const int code, const real i, const real j,
		const real k, real& u, real& v, real& w) {
	cv::Mat_<cv::Vec3f> start(cv::Vec3f(i, j, k));
	cv::Mat_<cv::Vec3f> end;
	cvtColor(start, end, code);
	u = (*end[0])[0];
	v = (*end[0])[1];
	w = (*end[0])[2];
}

void HSV2RGB(const real h, const real s, const real v, real& r,
		real &g, real& b) {
	openCVColorChange(cv::COLOR_HSV2RGB, h, s, v, r, g, b);
}

void RGB2XYZ(real r, real g, real b, real& x, real& y, real& z) {
	openCVColorChange(cv::COLOR_RGB2XYZ, r, g, b, x, y, z);
}

void XYZ2RGB(const real x, const real y, const real z, real& r,
		real& g, real& b) {
	openCVColorChange(cv::COLOR_XYZ2RGB, x, y, z, r, g, b);
}

void RGB2LAB(const real r, const real g, const real b, real& l,
		real& a, real& b_) {
	openCVColorChange(cv::COLOR_RGB2Lab, r, g, b, l, a, b_);
}

void LAB2RGB(const real l, const real a, const real b, real& r,
		real& g, real& b_) {
	openCVColorChange(cv::COLOR_Lab2RGB, l, a, b, r, g, b_);
}

void HSV2LAB(const real h, const real s, const real v, real& l,
		real &a, real& b) {
	real r, g;
	HSV2RGB(h,s,v,r,g,b);
	RGB2LAB(r,g,b,l,a,b);
}

void getProbColorForBinaryRGB(const real prob, real& r, real& g, real& b){
	r = prob * firstColor[0] + (1-prob) * secondColor[0];
	g = prob * firstColor[1] + (1-prob) * secondColor[1];
	b = prob * firstColor[2] + (1-prob) * secondColor[2];
}

void getProbColorForSecondBinaryRGB(const real prob, real& r, real& g, real& b){
	r = prob * thirdColor[0] + (1-prob) * fourthColor[0];
	g = prob * thirdColor[1] + (1-prob) * fourthColor[1];
	b = prob * thirdColor[2] + (1-prob) * fourthColor[2];
}


}

#endif /* UTILITY_COLORCONVERTER_H_ */
