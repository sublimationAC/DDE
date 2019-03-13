//ÄæÊ±ÕëÐý×ªÍ¼Ïñdegree½Ç¶È£¨Ô­³ß´ç£©
#include "image_process.hpp"

void rotateImage(cv::Mat img_rotate, float degree, cv::Point2d center)
{
	//¼ÆËã¶þÎ¬Ðý×ªµÄ·ÂÉä±ä»»¾ØÕó
	cv::Mat rot_matrix = cv::getRotationMatrix2D(center, degree * 180 / CV_PI, 1.0);
	std::cout << rot_matrix << "\n";
	cv::warpAffine(img_rotate, img_rotate, rot_matrix, cv::Size(), 1, 0, cv::Scalar(0));
}

void align_image(cv::Mat img, std::vector<cv::Point2d> &landmarks, std::vector<cv::Point2d> &mean_land_) {
	Transform t = Procrustes(mean_land_, landmarks);
	std::cout << "scale rotation:\n" << t.scale_rotation << "\n\n";
	std::cout << "translation:\n" << t.translation << "\n\n";
	float anglex = atan(t.scale_rotation(0, 1) / t.scale_rotation(0, 0));
	float angley = atan(-t.scale_rotation(1, 0) / t.scale_rotation(1, 1));
	printf("%.10f %.10f %.10f %.10f\n", anglex, anglex * 180 / CV_PI, angley, angley * 180 / CV_PI);
	rotateImage(img, anglex, landmarks[2]);
	//std::vector<cv::Point2d> temp(mean_land_);
	//t.Apply(&temp);
	//for (cv::Point2d landmark : temp)
	//	cv::circle(img, landmark, 0.1, cv::Scalar(255, 0, 0), 2);
	
}
void run_align_image(const FaceX & facex_5, cv::Mat img, cv::Rect rect_last) {
	cv::Mat gray_image;
	cv::cvtColor(img, gray_image, cv::COLOR_BGR2GRAY);
	std::vector<cv::Point2d> landmarks = facex_5.Alignment(gray_image, cv::Rect(0, 0, gray_image.cols, gray_image.rows));

	std::vector<cv::Point2d> mean_land_;
	mean_land_.push_back(cv::Point2d(94 ,116));
	mean_land_.push_back(cv::Point2d(137, 116));
	mean_land_.push_back(cv::Point2d(115.5, 142));
	mean_land_.push_back(cv::Point2d(96, 164));
	mean_land_.push_back(cv::Point2d(135, 164));

	//std::vector<cv::Point2d> landmarks = facex_5.Alignment(gray_image, rect_last);
	align_image(img, landmarks, mean_land_);
	cv::imshow("align_image", img);
	cv::waitKey(0);
	//system("pause");
}
//void test_align_image() {
//	puts("A");
//	cv::Mat image = cv::imread(kTestImage);// +pic_name);
//	vector<cv::Point2d> landmarks(5), ref_landmarks(5);
//	FILE *fp;
//	fopen_s(&fp, kTestImage_land_path.c_str(), "r");
//	puts("B");
//	printf("%d\n", landmarks.size());
//	//float x, y;
//	//fscanf_s(fp, "%f %f\n", &x, &y);
//	//printf("%.5f %.5f\n", x,y);
//	for (cv::Point2d &landmark : landmarks)
//	{
//		fscanf_s(fp, "%lf %lf\n", &landmark.x, &landmark.y);
//		printf("%.10f %.10f \n", landmark.x, landmark.y);
//		cv::circle(image, landmark, 0.1, cv::Scalar(0, 255, 0), 2);
//	}
//	fclose(fp);
//	puts("C");
//	fopen_s(&fp, ref_land_path.c_str(), "r");
//	for (cv::Point2d &landmark : ref_landmarks)
//	{
//		fscanf_s(fp, "%lf %lf\n", &landmark.x, &landmark.y);
//		printf("%.10f %.10f \n", landmark.x, landmark.y);
//		cv::circle(image, landmark, 0.1, cv::Scalar(0, 0, 255), 2);
//	}
//	puts("D");
//	fclose(fp);
//	//system("pause");
//	align_image(image, landmarks, ref_landmarks);
//}

//void deal_image(const FaceX & face_x)
//{
//	//cout << "picture name:";
//	//string pic_name;
//	//cin >> pic_name;
//
//	cv::Mat image = cv::imread(kTestImage);// +pic_name);
//	//rotate image
//	rotateImage(image, 1.0, cv::Point2d(image.cols / 2, image.rows / 2));
//
//
//	cv::Mat gray_image;
//	cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
//
//	//cv::imshow("rotation result", image);
//	//cv::imshow("Image gray", gray_image);
//	//cv::waitKey();
//
//	cv::CascadeClassifier cc(kAlt2);
//	if (cc.empty())
//	{
//		cout << "Cannot open model file " << kAlt2 << " for OpenCV face detector!" << endl;
//		return;
//	}
//	vector<cv::Rect> faces;
//	double start_time = cv::getTickCount();
//	cc.detectMultiScale(gray_image, faces);
//	cout << "Detection time: " << (cv::getTickCount() - start_time) / cv::getTickFrequency()
//		<< "s" << "\nthe number of detected faces:" << faces.size() << endl;
//	//-------------------------------debug---------------------
//	faces.push_back(cv::Rect(70, 90, 100, 120));//70,90~170*210
//	//-------------------------------debug---------------------
//	cv::Mat image_save, image_save_gray;
//	for (cv::Rect face : faces)
//	{
//		//rect_scale(face, 1.2);
//		std::cout << face << "\n";
//		//image(face).copyTo(image_save);
//		image.copyTo(image_save);
//		cv::resize(image_save, image_save, cv::Size(230, 230));
//
//		cv::cvtColor(image_save, image_save_gray, cv::COLOR_BGR2GRAY);
//
//		//cv::rectangle(image, face, cv::Scalar(0, 0, 255), 2);
//		//start_time = cv::getTickCount();
//		//vector<cv::Point2d> landmarks = face_x.Alignment(gray_image, face);
//		//cout << "Alignment time: "
//		//	<< (cv::getTickCount() - start_time) / cv::getTickFrequency()
//		//	<< "s" << endl;
//
//		//for (cv::Point2d landmark : landmarks)
//		//{
//		//	cv::circle(image, landmark, 0.1, cv::Scalar(0, 255, 0), 2);
//		//}
//
//
//		//image_save = image_temp.clone();
//		cv::imshow("Image save alignment result2", image_save);
//		cv::waitKey();
//
//
//		start_time = cv::getTickCount();
//		std::vector<cv::Point2d> landmarks = face_x.Alignment(image_save_gray, cv::Rect(0, 0, image_save.cols, image_save.rows));
//
//		cout << "Alignment time 2: "
//			<< (cv::getTickCount() - start_time) / cv::getTickFrequency()
//			<< "s" << endl;
//		FILE *fp;
//		fopen_s(&fp, kTestImage_land_path.c_str(), "w");
//		for (cv::Point2d landmark : landmarks)
//		{
//			fprintf(fp, "%.10f %.10f\n", landmark.x, landmark.y);
//			cv::circle(image_save, landmark, 0.1, cv::Scalar(0, 255, 0), 2);
//		}
//		fclose(fp);
//
//		cv::imwrite("test_roll.jpg", image_save);
//
//	}
//
//	cv::imshow("Alignment result", image);
//	cv::imshow("Image save alignment result", image_save);
//	cv::waitKey();
//
//
//}
//
