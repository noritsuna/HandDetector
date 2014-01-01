/*
   HandDetector for Android NDK
   Copyright (c) 2006-2013 SIProp Project http://www.siprop.org/

   This software is provided 'as-is', without any express or implied warranty.
   In no event will the authors be held liable for any damages arising from the use of this software.
   Permission is granted to anyone to use this software for any purpose,
   including commercial applications, and to alter it and redistribute it freely,
   subject to the following restrictions:

   1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
   2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
   3. This notice may not be removed or altered from any source distribution.
*/
#ifndef HANDDETECTOR_H_
#define HANDDETECTOR_H_

#include <android/log.h>

#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <errno.h>
#include <fcntl.h>

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/cvaux.h>
#include <opencv/ml.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

const int CELL_X = 5;
const int CELL_Y = 5;
const int CELL_BIN = 9;
const int BLOCK_X = 3;
const int BLOCK_Y = 3;

const int RESIZE_X = 40;
const int RESIZE_Y = 40;

const int CELL_WIDTH = RESIZE_X / CELL_X;
const int CELL_HEIGHT = RESIZE_Y / CELL_Y;
const int BLOCK_WIDTH = CELL_WIDTH - BLOCK_X + 1;
const int BLOCK_HEIGHT = CELL_HEIGHT - BLOCK_Y + 1;

const int BLOCK_DIM	= BLOCK_X * BLOCK_Y * CELL_BIN;
const int TOTAL_DIM	= BLOCK_DIM * BLOCK_WIDTH * BLOCK_HEIGHT;

void calcHistgram(IplImage* srcImage, CvHistogram** hist, double* v_min, double* v_max);
void detectSkinColorArea(IplImage* srcImage_hsv,
						 IplImage** skinColorAreaImage,
						 CvHistogram* hist,
						 CvSeq** convers,
						 double* vmin,
						 double* vmax);
void getHoG(IplImage* srcImage, double* feat);
double getDistance(double* feat1, double* feat2);

#endif /* HANDDETECTOR_H_ */
