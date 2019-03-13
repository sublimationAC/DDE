#pragma once
#include "face_x.h"

void rotateImage(cv::Mat img_rotate, float degree, cv::Point2d center);
void align_image(cv::Mat img, std::vector<cv::Point2d> &landmarks, std::vector<cv::Point2d> &mean_land_);
void run_align_image(const FaceX & facex_5, cv::Mat img, cv::Rect rect_last);