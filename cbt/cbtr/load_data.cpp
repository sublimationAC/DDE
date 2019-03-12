#include "load_data.hpp"
#include <dirent.h>
#include <io.h>
#define flap_2dland
#define debug


int num = 0;
void load_img_land_coef(std::string path, std::string sfx, std::vector<DataPoint> &img) {
	struct dirent **namelist;
	int n;
	n = scandir(path.c_str(), &namelist, 0, alphasort);
	if (n < 0)
	{
		std::cout << "scandir return " << n << "\n";
		perror("Cannot open .");
		exit(1);
	}
	else
	{
		int index = 0;
		struct dirent *dp;
		while (index < n)
		{
			if (num > 20) break;
			printf("load_img_land_coef idx:%d n:%d\n", index, n);
			dp = namelist[index];

			if (dp->d_name[0] == '.') {
				free(namelist[index]);
				index++;
				continue;
			}
			if (dp->d_type == DT_DIR) {
				std::cout << dp->d_name << ' ' << strlen(dp->d_name) << "\n";
				//printf("Loading identity %d...\n", num);
				load_img_land_coef(path + "/" + dp->d_name, sfx, img);
			}
			else {
				int len = strlen(dp->d_name);
				if (dp->d_name[len - 1] == 'd' && dp->d_name[len - 2] == 'n') {
					////	
					DataPoint temp;
					std::string p = path + "/" + dp->d_name;


					if (_access((p.substr(0, p.find(".land")) + sfx).c_str(), 0) == -1) {
						free(namelist[index]);
						index++;
						continue;
					}
					load_land(path + "/" + dp->d_name, temp);

					load_img(p.substr(0, p.find(".land")) + sfx, temp);
#ifdef  flap_2dland
					for (int i = 0; i < temp.landmarks.size(); i++)
						temp.landmarks[i].y = temp.image.rows - temp.landmarks[i].y;
#endif //  flap_2dland
					//cal_rect(temp);
					//system("pause");
#ifdef  normalization
					if (_access((p.substr(0, p.find(".land")) + ".lv").c_str(), 0) == -1) {
						puts("No lv !!!! error !");
						exit(1);

					}
					load_fitting_coef_one(p.substr(0, p.find(".land")) + ".lv", temp);
#endif //  normalization

#ifdef perspective
					if (_access((p.substr(0, p.find(".land")) + ".psp_f").c_str(), 0) == -1) {
						puts("No psp_f !!!! error !");
						exit(1);
					}
					load_fitting_coef_one(p.substr(0, p.find(".land")) + ".psp_f", temp);
#endif // perspective


					img.push_back(temp);

					num++;
		//			test_data_2dland(temp);
				}
			}
			free(namelist[index]);
			index++;
		}
		free(namelist);
	}

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
	temp.image = cv::imread(p, CV_LOAD_IMAGE_GRAYSCALE);
}
const std::string kAlt2 = "haarcascade_frontalface_alt2.xml";
#include<algorithm>
void cal_rect(DataPoint &temp) {
	puts("testing image");
	//puts("AA");
	cv::Mat gray_image;
	//puts("AB");
	//cv::cvtColor(temp.image, gray_image, CV_BGR2GRAY);
	gray_image = temp.image;
	//puts("AC");
	cv::CascadeClassifier cc(kAlt2);
	if (cc.empty())
	{
		std::cout << "Cannot open model file " << kAlt2 << " for OpenCV face detector!\n";
		return;
	}
	//puts("AA");
	std::vector<cv::Rect> faces;
	double start_time = cv::getTickCount();

	cc.detectMultiScale(gray_image, faces);
	//std::cout << "Detection time: " << (cv::getTickCount() - start_time) / cv::getTickFrequency()
	//	<< "s" << "\n";
	
	int cnt = 0, ma = 0;
	for (cv::Rect face : faces) {
		//rect_scale(face, 1.5);
		//face.x = max(face.x, 0); face.y = max(0, face.y);
		//face.width = min(face.width, gray_image.cols - face.x);
		//face.height = min(face.height, gray_image.rows - face.y);
		//face.x = max(0, face.x - 10);// face.y = max(0, face.y - 10);
		//face.width = min(temp.image.rows - face.x, face.width + 25);
		//face.height = min(temp.image.cols - face.y, face.height + 25);
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

	//if (ma == 0) temp.face_rect = cv::Rect(left - 10, top - 10, right - left + 21, bottom - top + 21);
	//cv::Rect temp_rect(left, top, right - left, bottom - top);
	//rect_scale(temp_rect, 1.5);
	//temp_rect.x = max(temp_rect.x, 0); temp_rect.y = max(0, temp_rect.y);
	//temp_rect.width = min(temp_rect.width, gray_image.cols - temp_rect.x);
	//temp_rect.height = min(temp_rect.height, gray_image.rows - temp_rect.y);
	if (ma == 0) temp.face_rect = cv::Rect(left, top, right - left, bottom - top);
#ifndef small_rect_def
	rect_scale(temp.face_rect, 1.5);
	temp.face_rect.x = max(temp.face_rect.x, 0); temp.face_rect.y = max(0, temp.face_rect.y);
	temp.face_rect.width = min(temp.face_rect.width, gray_image.cols - temp.face_rect.x);
	temp.face_rect.height = min(temp.face_rect.height, gray_image.rows - temp.face_rect.y);
#endif // !small_rect_def
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
	
	if (ma == 0) temp.face_rect = cv::Rect(left - (float)10.0, top - (float)10, right - left + (float)21, bottom - top + (float)21);

	cv::rectangle(temp.image, cv::Rect(left - (float)10, top - (float)10, right - left + (float)21, bottom - top + (float)21), cv::Scalar(255, 0, 255), 2);
	cv::rectangle(temp.image, temp.face_rect, cv::Scalar(100, 200, 255), 2);
	cv::imshow("result", temp.image);
	cv::waitKey();
	system("pause");
}

void load_fitting_coef_one(std::string name, DataPoint &temp) {
	std::cout << "loading coefficients...file:" << name << "\n";
	FILE *fp;
	fopen_s(&fp, name.c_str(), "rb");
	temp.user.resize(G_iden_num);
	for (int j = 0; j < G_iden_num; j++)
		fread(&temp.user(j), sizeof(float), 1, fp);

	temp.land_2d.resize(G_land_num, 2);
	for (int i_v = 0; i_v < G_land_num; i_v++) {
		fread(&temp.land_2d(i_v, 0), sizeof(float), 1, fp);
		fread(&temp.land_2d(i_v, 1), sizeof(float), 1, fp);
	}

	
	fread(&temp.center(0), sizeof(float), 1, fp);
	fread(&temp.center(1), sizeof(float), 1, fp);

	temp.shape.exp.resize(G_nShape);
	for (int i_shape = 0; i_shape < G_nShape; i_shape++)
		fread(&temp.shape.exp(i_shape), sizeof(float), 1, fp);

	//for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
	//	fread(&temp.shape.rot(i, j), sizeof(float), 1, fp);
	//temp.shape.angle = get_uler_angle(temp.shape.rot);

	Eigen::Matrix3f rot;
	for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
		fread(&rot(i, j), sizeof(float), 1, fp);
	temp.shape.angle = get_uler_angle_zyx(rot);

	for (int i = 0; i < 3; i++) fread(&temp.shape.tslt(i), sizeof(float), 1, fp);
//#ifdef normalization
//	temp.shape.tslt(2) = 0;
//#endif // normalization

	temp.land_cor.resize(G_land_num);
	for (int i_v = 0; i_v < G_land_num; i_v++) fread(&temp.land_cor(i_v), sizeof(int), 1, fp);


#ifdef normalization
	temp.s.resize(2, 3);
	for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++)
		fread(&temp.s(i, j), sizeof(float), 1, fp);
#endif // normalization

#ifdef perspective
	fread(&temp.fcs, sizeof(float), 1, fp);
#endif // perspective


	temp.shape.dis.resize(G_land_num, 2);
	for (int i_v = 0; i_v < G_land_num; i_v++) {
		fread(&temp.shape.dis(i_v, 0), sizeof(float), 1, fp);
		fread(&temp.shape.dis(i_v, 1), sizeof(float), 1, fp);
	}

	fclose(fp);
#ifdef normalization
	temp.land_2d.rowwise() += temp.center;
#endif // normalization	
	puts("load successful!");
}

void load_bldshps(Eigen::MatrixXf &bldshps, std::string &name) {

	puts("loading blendshapes...");
	std::cout << name << std::endl;
	FILE *fp;
	fopen_s(&fp, name.c_str(), "rb");
	for (int i = 0; i < G_iden_num; i++) {
		for (int j = 0; j < G_nShape*G_nVerts * 3; j++)
			fread(&bldshps(i, j), sizeof(float), 1, fp);
	}
	fclose(fp);
}
