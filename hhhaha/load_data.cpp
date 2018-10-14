#include "utils_train.h"
#include <dirent.h>
#include <io.h>
#define flap_2dland
#define debug


int num = 0;
void load_img_land(std::string path, std::string sfx, std::vector<DataPoint> &img) {
	DIR *dir;
	struct dirent *dp;
	if ((dir = opendir(path.c_str())) == NULL) {
		perror("Cannot open .");
		exit(1);
	}
	while ((dp = readdir(dir)) != NULL) {
		
		if (dp->d_name[0] == '.') continue;
		if (dp->d_type == DT_DIR) {			
			std::cout << dp->d_name << ' ' << strlen(dp->d_name) << "\n";
			//printf("Loading identity %d...\n", num);
			load_img_land(path + "/" + dp->d_name, sfx, img);
		}
		else {
			int len = strlen(dp->d_name);
			if (dp->d_name[len - 1] == 'd' && dp->d_name[len - 2] == 'n') {
				////	
				DataPoint temp;
				std::string p = path + "/" + dp->d_name;


				if (_access((p.substr(0, p.find(".land")) + sfx).c_str(), 0) == -1) continue;
				load_land(path + "/" + dp->d_name, temp);
				
				load_img(p.substr(0, p.find(".land")) + sfx, temp);
#ifdef  flap_2dland
				for (int i = 0; i < temp.landmarks.size(); i++)
					temp.landmarks[i].y = temp.image.rows - temp.landmarks[i].y;
#endif //  flap_2dland
				cal_rect(temp);
				//system("pause");
				img.push_back(temp);


				num++;
				test_data_2dland(temp);
			}
		}
	}
	closedir(dir);

}
void load_land(std::string p, DataPoint &temp) {
	std::cout << p << '\n';
	FILE *fp;
	fopen_s(&fp, p.c_str(), "r");
	int n;
	fscanf_s(fp, "%d", &n);
	//temp.landmarks.clear();
	temp.landmarks.resize(n);
	for (int i = 0; i < n; i++) {
		double x, y;
		fscanf_s(fp, "%lf%lf", &(temp.landmarks[i].x), &(temp.landmarks[i].y));
		/*temp.landmarks.push_back(cv::Point2d(x, y));
		printf("%d %.10f %.10f \n", temp.landmarks.size(),temp.landmarks[i].x, temp.landmarks[i].y);*/
	}
	fclose(fp);
	//system("pause");
}

void load_img(std::string p, DataPoint &temp) {	
	temp.image = cv::imread(p);// , CV_LOAD_IMAGE_GRAYSCALE);
}
const std::string kAlt2 = "haarcascade_frontalface_alt2.xml";
void cal_rect(DataPoint &temp) {
	//puts("testing image");
	cv::Mat gray_image;
	cv::cvtColor(temp.image, gray_image, CV_BGR2GRAY);
	cv::CascadeClassifier cc(kAlt2);
	if (cc.empty())
	{
		std::cout << "Cannot open model file " << kAlt2 << " for OpenCV face detector!\n";
		return;
	}
	std::vector<cv::Rect> faces;
	double start_time = cv::getTickCount();

	cc.detectMultiScale(gray_image, faces);
	//std::cout << "Detection time: " << (cv::getTickCount() - start_time) / cv::getTickFrequency()
	//	<< "s" << "\n";

	int cnt = 0, ma = 0;
	for (cv::Rect face : faces) {
		face.x = max(0, face.x - 10);// face.y = max(0, face.y - 10);
		face.width = min(temp.image.rows - face.x, face.width + 25);
		face.height = min(temp.image.cols - face.y, face.height + 25);
		int in_num = 0;
		for (cv::Point2d landmark : temp.landmarks)
			if (landmark.inside(face)) in_num++;
		if (in_num > ma) ma = in_num, temp.face_rect = face;
		cnt++;
	}
	//printf("faces number: %d \n", cnt);
	double left = 10000, right = -10000, top = 10000, bottom = -10000;
	for (cv::Point2d landmark : temp.landmarks) {
		left = min(left, landmark.x);
		right = max(right, landmark.x);
		top = min(top, landmark.y);
		bottom = max(bottom, landmark.y);
	}
	if (ma == 0) temp.face_rect = cv::Rect(left - 10, top - 10, right - left + 21, bottom - top + 21);
}



void test_data_2dland(DataPoint &temp) {
	puts("testing image");
	system("pause");
	cv::Mat gray_image;
	cv::cvtColor(temp.image, gray_image, CV_BGR2GRAY);
	cv::CascadeClassifier cc(kAlt2);
	if (cc.empty())
	{
		std::cout << "Cannot open model file " << kAlt2 << " for OpenCV face detector!\n";
		return;
	}
	std::vector<cv::Rect> faces;
	double start_time = cv::getTickCount();

	cc.detectMultiScale(gray_image, faces);
	std::cout << "Detection time: " << (cv::getTickCount() - start_time) / cv::getTickFrequency()
		<< "s" << "\n";

	int cnt = 0,ma=0;
	for (cv::Rect face : faces) {		
		face.x = max(0, face.x - 10);// face.y = max(0, face.y - 10);
		face.width = min(temp.image.rows - face.x, face.width + 25);
		face.height = min(temp.image.cols - face.y, face.height + 25);		
		cv::rectangle(temp.image, face, cv::Scalar(0, 0, 255), 2);
		int in_num = 0;
		for (cv::Point2d landmark : temp.landmarks)
			if (landmark.inside(face)) in_num++;
		if (in_num > ma) ma = in_num, temp.face_rect = face;
		cnt++;
	}
	
	
	printf("faces number: %d \n", cnt);
	double left = 10000, right = -10000, top = 10000, bottom = -10000;
	for (cv::Point2d landmark : temp.landmarks){
		cv::circle(temp.image, landmark,1, cv::Scalar(0, 255, 0), 2);
		std::cout << "-+-" << landmark.x << ' ' << landmark.y << "\n";
		left = min(left, landmark.x);
		right = max(right, landmark.x);
		top = min(top, landmark.y);
		bottom = max(bottom, landmark.y);
	}
	printf("%.5f %.5f %.5f %.5f\n", left, right, top, bottom);
	
	if (ma == 0) temp.face_rect = cv::Rect(left - 10, top - 10, right - left + 21, bottom - top + 21);

	cv::rectangle(temp.image, cv::Rect(left - 10, top - 10, right - left + 21, bottom - top + 21), cv::Scalar(255, 0, 255), 2);
	cv::rectangle(temp.image, temp.face_rect, cv::Scalar(100, 200, 255), 2);
	cv::imshow("result", temp.image);
	cv::waitKey();
	system("pause");
}