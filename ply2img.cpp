//#include "opencv2/imgproc.hpp"
//#include "opencv2/highgui.hpp"
//#include <iostream>
//#include <ctype.h>
//using namespace cv;
//using namespace std;
//const int width = 1280;
//const int height = 720;
//double x[887231], y[887231], z[887231];
//int r[887231], g[887231], b[887231];
//int main()
//{
//	FILE *fp;
//	fopen_s(&fp,"./point_cloud_PLY_18417_720_06-10-2018-11-49-10.ply", "r");
//	char temp[300];
//	for (int i = 0; i < 10; i++) fgets(temp,255,fp);
//
//	Mat img(height, width, CV_32FC3);
//	double mi_x = 99999, ma_x = -999999,mi_y = 999999,ma_y = -999999;
//	for (int i = 0; i < 887231; i++) {
//		fscanf_s(fp,"%lf%lf%lf%d%d%d", &x[i], &y[i], &z[i], &r[i], &g[i], &b[i]);
//		mi_x = min(mi_x, x[i]), mi_y = min(mi_y, y[i]);
//		ma_x = max(ma_x, x[i]), ma_y = max(ma_y, y[i]);
//	}
//	fclose(fp);
//	for (int i = 0; i < 887231; i++) {
//		Point pos(
//			width - 1 - (int)(round((y[i] - mi_y) / (ma_y - mi_y)*(width - 1))),
//			height - 1 - (int)(round((x[i] - mi_x) / (ma_x - mi_x)*(height - 1))));
//		img.at<Vec3f>(pos) = Vec3f(b[i], g[i], r[i]);
//	}
//	//GaussianBlur(img, img, Size(3, 3), 0, 0);
//	//blur(img, img, Size(10, 10));
//	imwrite("test.jpg", img);
//	
//	return 0;
//}
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{

	// Read input image
	cv::Mat im = cv::imread("headPose.jpg");
	//cv::imshow("head",im);
	//cv::waitKey(0);
	// 2D image points. If you change the image, you need to change vector
	std::vector<cv::Point2d> image_points;
	image_points.push_back(cv::Point2d(359, 391));    // Nose tip
	image_points.push_back(cv::Point2d(399, 561));    // Chin
	image_points.push_back(cv::Point2d(337, 297));     // Left eye left corner
	image_points.push_back(cv::Point2d(513, 301));    // Right eye right corner
	image_points.push_back(cv::Point2d(345, 465));    // Left Mouth corner
	image_points.push_back(cv::Point2d(453, 469));    // Right mouth corner

	// 3D model points.
	std::vector<cv::Point3d> model_points;
	model_points.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));               // Nose tip
	model_points.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));          // Chin
	model_points.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));       // Left eye left corner
	model_points.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));        // Right eye right corner
	model_points.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));      // Left Mouth corner
	model_points.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));       // Right mouth corner

	// Camera internals
	double focal_length = im.cols; // Approximate focal length.
	Point2d center = cv::Point2d(im.cols / 2, im.rows / 2);
	cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
	cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion

	cout << "Camera Matrix " << endl << camera_matrix << endl;
	// Output rotation and translation
	cv::Mat rotation_vector; // Rotation in axis-angle form
	cv::Mat translation_vector;

	// Solve for pose
	cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);


	// Project a 3D point (0, 0, 1000.0) onto the image plane.
	// We use this to draw a line sticking out of the nose

	vector<Point3d> nose_end_point3D;
	vector<Point2d> nose_end_point2D;
	nose_end_point3D.push_back(Point3d(0, 0, 1000.0));

	projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, nose_end_point2D);


	for (int i = 0; i < image_points.size(); i++)
	{
		circle(im, image_points[i], 3, Scalar(0, 0, 255), -1);
	}

	cv::line(im, image_points[0], nose_end_point2D[0], cv::Scalar(255, 0, 0), 2);

	cout << "Rotation Vector " << endl << rotation_vector << endl 
		<< "rot type:" << rotation_vector.type() << endl 
		<< rotation_vector.rows << ' ' << rotation_vector.cols << endl;
	cv::Mat rot;
	cv::Rodrigues(rotation_vector, rot);

	cout << " rot:\n" << rot << endl;
	cout << "Translation Vector" << endl << translation_vector << endl;

	cout << nose_end_point2D << endl;

	// Display image.
	cv::imshow("Output", im);
	cv::waitKey(0);

}