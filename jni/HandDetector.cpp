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
#include "HandDetector.hpp"

#define  LOG_TAG    "Detect Hand"
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG,__VA_ARGS__)
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGW(...)  __android_log_print(ANDROID_LOG_WARN,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)


void calcHistgram(IplImage* srcImage, CvHistogram** hist, double* v_min, double* v_max) {

	CvSize size = cvGetSize(srcImage);
	IplImage* hsv = cvCreateImage(size, IPL_DEPTH_8U, 3);
	cvCvtColor( srcImage, hsv, CV_BGR2HSV );

	IplImage* h_plane  = cvCreateImage(size, IPL_DEPTH_8U, 1);
	IplImage* s_plane  = cvCreateImage(size, IPL_DEPTH_8U, 1);
	IplImage* v_plane  = cvCreateImage(size, IPL_DEPTH_8U, 1);
	IplImage* planes[] = { h_plane, s_plane };
	cvCvtPixToPlane( hsv, h_plane, s_plane, v_plane, 0 );

	const int H_BINS = 18;
	const int S_BINS = 25;
	const int H_MIN  = 0;
	const int H_MAX  = 180;
	const int S_MIN  = 0;
	const int S_MAX  = 255;
	int    hist_size[] = { H_BINS, S_BINS };
	float  h_ranges[]  = { H_MIN, H_MAX };
	float  s_ranges[]  = { S_MIN, S_MAX };
	float* ranges[]    = { h_ranges, s_ranges };
	*hist = cvCreateHist(2,
						hist_size,
						CV_HIST_ARRAY,
						ranges,
						1);

	cvCalcHist(planes, *hist, 0, 0);
	cvReleaseImage(&h_plane);
	cvReleaseImage(&s_plane);

	cvMinMaxLoc(v_plane, v_min, v_max);
	*v_min = 1;
	cvReleaseImage(&v_plane);
}

void detectSkinColorArea(IplImage* srcImage_hsv,
						 IplImage** skinColorAreaImage,
						 CvHistogram* hist,
						 CvSeq** convers,
						 double* v_min,
						 double* v_max) {

	CvMemStorage* storage = cvCreateMemStorage(0);

	CvSize size = cvGetSize(srcImage_hsv);
	IplImage* dstImage = cvCreateImage(size, IPL_DEPTH_8U, 1);
	cvZero(dstImage);

	IplImage* backProjectImage = cvCreateImage(size, IPL_DEPTH_8U, 1);
	IplImage* maskImage = cvCreateImage(size, IPL_DEPTH_8U, 1);

	{
		IplImage* h_plane = cvCreateImage(size, IPL_DEPTH_8U,1);
		IplImage* s_plane = cvCreateImage(size, IPL_DEPTH_8U,1);
		IplImage* v_plane = cvCreateImage(size, IPL_DEPTH_8U,1);
		IplImage* planes[] = {h_plane, s_plane};

		cvCvtPixToPlane(srcImage_hsv, h_plane, s_plane, v_plane, NULL);
		cvCalcBackProject(planes, backProjectImage, hist);

		cvThreshold(v_plane, maskImage, *v_min, *v_max, CV_THRESH_BINARY);
		cvAnd(backProjectImage, maskImage, backProjectImage);

		cvReleaseImage(&h_plane);
		cvReleaseImage(&s_plane);
		cvReleaseImage(&v_plane);
	}

	CvSeq* contours = NULL;
	{

		cvThreshold(backProjectImage, dstImage, 10,255, CV_THRESH_BINARY);
//		cvThreshold(imgBackproj, dst_image, 40,255, CV_THRESH_BINARY);

		cvErode(dstImage, dstImage, NULL, 1);
		cvDilate(dstImage, dstImage, NULL, 1);


		cvFindContours(dstImage, storage, &contours);
		CvSeq* hand_ptr = NULL;
		double maxArea = -1;
		for (CvSeq* c= contours; c != NULL; c = c->h_next){
			double area = abs(cvContourArea(c, CV_WHOLE_SEQ));
			if (maxArea < area) {
				maxArea = area;
				hand_ptr = c;
			}
		}

		cvZero(dstImage);
		if (hand_ptr == NULL) {
			*skinColorAreaImage = cvCreateImage(cvSize(1, 1), IPL_DEPTH_8U, 1);
		} else {
			hand_ptr->h_next = NULL;
			*convers = hand_ptr;

			cvDrawContours(dstImage, hand_ptr, cvScalarAll(255), cvScalarAll(0),100);
			CvRect rect= cvBoundingRect(hand_ptr,0);
			cvSetImageROI(dstImage, rect);
			*skinColorAreaImage = cvCreateImage(cvSize(rect.width, rect.height), IPL_DEPTH_8U, 1);
			cvCopy(dstImage, *skinColorAreaImage);

			cvResetImageROI(dstImage);
		}
	}
	cvReleaseImage(&backProjectImage);
	cvReleaseImage(&maskImage);
	cvReleaseImage(&dstImage);
	cvReleaseMemStorage(&storage);
}

void getHoG(IplImage* src, double* feat) {

	IplImage* img = cvCreateImage(cvSize(RESIZE_X,RESIZE_Y), IPL_DEPTH_8U, 1);
	cvResize(src, img);

	const int width = RESIZE_X;
	const int height = RESIZE_Y;

	double hist[CELL_WIDTH][CELL_HEIGHT][CELL_BIN];
	memset(hist, 0, CELL_WIDTH*CELL_HEIGHT*CELL_BIN*sizeof(double));

	for(int y=0; y<height; y++){
		for(int x=0; x<width; x++){
			if(x==0 || y==0 || x==width-1 || y==height-1){
				continue;
			}
			double dx = img->imageData[y*img->widthStep+(x+1)] - img->imageData[y*img->widthStep+(x-1)];
			double dy = img->imageData[(y+1)*img->widthStep+x] - img->imageData[(y-1)*img->widthStep+x];
			double m = sqrt(dx*dx+dy*dy);
			double deg = (atan2(dy, dx)+CV_PI) * 180.0 / CV_PI;
			int bin = CELL_BIN * deg/360.0;
			if(bin < 0) bin=0;
			if(bin >= CELL_BIN) bin = CELL_BIN-1;
			hist[(int)(x/CELL_X)][(int)(y/CELL_Y)][bin] += m;
		}
	}

	for(int y=0; y<BLOCK_HEIGHT; y++){
		for(int x=0; x<BLOCK_WIDTH; x++){
			double vec[BLOCK_DIM];
			memset(vec, 0, BLOCK_DIM*sizeof(double));
			for(int j=0; j<BLOCK_Y; j++){
				for(int i=0; i<BLOCK_X; i++){
					for(int d=0; d<CELL_BIN; d++){
						int index = j*(BLOCK_X*CELL_BIN) + i*CELL_BIN + d;
						vec[index] = hist[x+i][y+j][d];
					}
				}
			}

			double norm = 0.0;
			for(int i=0; i<BLOCK_DIM; i++){
				norm += vec[i]*vec[i];
			}
			for(int i=0; i<BLOCK_DIM; i++){
				vec[i] /= sqrt(norm + 1.0);
			}

			for(int i=0; i<BLOCK_DIM; i++){
				int index = y*BLOCK_WIDTH*BLOCK_DIM + x*BLOCK_DIM + i;
				feat[index] = vec[i];
			}
		}
	}
	cvReleaseImage(&img);
	return;
}

double getDistance(double* feat1, double* feat2) {
	double dist = 0.0;
	for(int i = 0; i < TOTAL_DIM; i++){
		dist += fabs(feat1[i] - feat2[i])*fabs(feat1[i] - feat2[i]);
	}
	return sqrt(dist);
}
