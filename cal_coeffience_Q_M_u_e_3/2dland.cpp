#include "2dland.h"
#define flap_2dland

#define cal_land_num
void load_img_land(std::string path, std::string sfx, iden *ide, int &id_idx, std::vector< std::vector<cv::Mat_<uchar> > > &imgs) {
#ifdef win64
	int hFile = 0;
	struct _finddata_t fileinfo;
	std::string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1) {
		//puts("-----------------");
		//std::cout << "id: " << hFile << '\n';
		do {
			if (fileinfo.name[0] == '.') continue;
			std::cout << fileinfo.name << '\n';
			if ((fileinfo.attrib & _A_SUBDIR)) {
				std::vector<cv::Mat_<uchar> > img_temp;
				printf("Loading identity %d...\n", id_idx);
				id_idx += load_img_land_same_id(p.assign(path).append("/").append(fileinfo.name).c_str(), sfx, ide, id_idx, img_temp);
				imgs.push_back(img_temp);
			}
			if (id_idx > 5) break;////////////////////////////////////////////////////////////////////////////////////////////////debug
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
#endif // win64

#ifdef linux
	DIR *dir;
	struct dirent *dp;
	if ((dir = opendir(path.c_str())) == NULL) {
		perror("Cannot open .");
		exit(1);
	}
	while ((dp = readdir(dir)) != NULL) {
		std::cout << dp->d_name << ' ' << strlen(dp->d_name) << "\n";
		if (dp->d_name[0] == '.') continue;
		if (dp->d_type == DT_DIR) {
			std::vector<cv::Mat_<uchar> > img_temp;
			printf("Loading identity %d...\n", id_idx);
			id_idx += load_img_land_same_id(path + "/" + dp->d_name, sfx, ide, id_idx, img_temp);
			imgs.push_back(img_temp);
		}
	}
	closedir(dir);
#endif // linux	
}

int load_img_land_same_id(std::string path, std::string sfx, iden *ide, int id_idx, std::vector<cv::Mat_<uchar> > &img_temp) {
#ifdef win64
	int hFile = 0, flag = 0;
	struct _finddata_t fileinfo;
	std::string p;
	std::cout << path << '\n';
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1) {
		do {
			if (fileinfo.name[0] == '.') continue;
			//std::cout << fileinfo.name << '\n';
			if ((fileinfo.attrib & _A_SUBDIR)) {
				flag |= load_img_land_same_id(p.assign(path).append("/").append(fileinfo.name).c_str(), sfx, ide, id_idx, img_temp);
			}
			else {
				int len = strlen(fileinfo.name);
				if (fileinfo.name[len - 1] == 'd' && fileinfo.name[len - 2] == 'n') {
					////	
					flag = 1;
					load_land(p.assign(path).append("/").append(fileinfo.name), ide, id_idx);

					p = path + "/" + fileinfo.name;
					cv::Mat_<uchar> temp;
					load_img(p.substr(0, p.find(".land")) + sfx, temp);
					img_temp.push_back(temp);

#ifdef  flap_2dland
					for (int i = G_land_num * (ide[id_idx].num); i < G_land_num*(ide[id_idx].num + 1); i++)
						ide[id_idx].land_2d(i, 1)=temp.rows- ide[id_idx].land_2d(i, 1);					
#endif //  flap_2dland



					ide[id_idx].num++;
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
	return flag;
#endif // win64

#ifdef linux
	int flag = 0;
	DIR *dir;
	struct dirent *dp;
	if ((dir = opendir(path.c_str())) == NULL) {
		perror("Cannot open .");
		exit(1);
	}
	while ((dp = readdir(dir)) != NULL) {
		//std::cout << dp->d_name << ' ' << dp->d_namlen << "\n";
		if (dp->d_name[0] == '.') continue;
		if (dp->d_type == DT_DIR)
			flag |= load_img_land_same_id(path + "/" + dp->d_name, sfx, ide, id_idx, img_temp);
		else {
			int len =strlen( dp->d_name);
			if (dp->d_name[len - 1] == 'd' && dp->d_name[len - 2] == 'n') {
				////	
				flag = 1;
				load_land(path + "/" + dp->d_name, ide, id_idx);
				std::string p = path + "/" + dp->d_name;
				cv::Mat_<uchar> temp;
				load_img(p.substr(0, p.find(".land")) + sfx, temp);
				img_temp.push_back(temp);
#ifdef  flap_2dland
				for (int i = G_land_num * (ide[id_idx].num); i < G_land_num*(ide[id_idx].num + 1); i++)
					ide[id_idx].land_2d(i, 1) = temp.rows - ide[id_idx].land_2d(i, 1);
#endif //  flap_2dland
				ide[id_idx].num++;
			}
		}
	}
	closedir(dir);
	return flag;
#endif // linux
}

void load_land(std::string p, iden *ide, int id_idx) {
	std::cout << p << '\n';
	FILE *fp;
	fopen_s(&fp, p.c_str(), "r");
	int n;
	fscanf_s(fp, "%d", &n);
	ide[id_idx].land_2d.conservativeResize(n*(ide[id_idx].num + 1), 2);
	for (int i = n * (ide[id_idx].num); i < n*(ide[id_idx].num + 1); i++)
		fscanf_s(fp, "%f%f", &ide[id_idx].land_2d(i, 0), &ide[id_idx].land_2d(i, 1));
	fclose(fp);
}

void load_img(std::string p, cv::Mat_<uchar> &temp) {
	temp = cv::imread(p , 0);
}

void test_data_2dland(
	std::vector < std::vector <cv::Mat_<uchar> > >& imgs,
	iden *ide, int id_idx, int img_idx) {	
	for (int i = img_idx* G_land_num; i < (img_idx+1)*G_land_num; i++) {
#ifdef flap_2dland
		cv::circle(
			imgs[id_idx][img_idx],
			cv::Point2f(ide[id_idx].land_2d(i, 0), ide[id_idx].land_2d(i, 1)),
			1, cv::Scalar(244, 244, 244), -1, 8, 0);
#else
		cv::circle(
			imgs[id_idx][img_idx],
			cv::Point2f(ide[id_idx].land_2d(i, 0), imgs[id_idx][img_idx].rows - ide[id_idx].land_2d(i, 1)),
			1, cv::Scalar(244, 244, 244), -1, 8, 0);
#endif // flap_2dland

		
		/*rectangle(images[j], Point2d(bounding_box[j].start_x, bounding_box[j].start_y)
			, Point2d(bounding_box[j].start_x + bounding_box[j].width, bounding_box[j].start_y + bounding_box[j].height)
			, Scalar(255, 244, 244), 3, 4, 0);*/
	}
#ifdef cal_land_num
	//int mi = img_idx * G_land_num + 70;
		CvFont font;
		cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 0.01, 0.5, 1, 0.5, 8);
		IplImage *pImage = cvLoadImage("D:\\sydney\\first\\data_me\\test\\Tester_1\\TrainingPose/pose_22.jpg");
		std::string si;
		for (int i = img_idx * G_land_num + 27; i < img_idx * G_land_num + 35; i++) {
			//27-35 66-74
			si = std::to_string(i - img_idx * G_land_num), cvPutText(pImage, si.c_str(), cv::Point2f(ide[id_idx].land_2d(i, 0),ide[id_idx].land_2d(i, 1))
				, &font, cv::Scalar(255, 255, 255));
			cv::circle(
				imgs[id_idx][img_idx],
				cv::Point2f(ide[id_idx].land_2d(i, 0), ide[id_idx].land_2d(i, 1)),
				3, cv::Scalar(244, 244, 244), -1, 8, 0);
		}
		cvShowImage("Original", pImage);
		/*rectangle(images[j], Point2d(bounding_box[j].start_x, bounding_box[j].start_y)
		, Point2d(bounding_box[j].start_x + bounding_box[j].width, bounding_box[j].start_y + bounding_box[j].height)
		, Scalar(255, 244, 244), 3, 4, 0);*/
	
#endif // cal_land_num

	
	cv::imshow("test_image", imgs[id_idx][img_idx]);
	cv::waitKey(0);
	/*for (int j = 0; j < landmark_num; j++)
	printf("%.10f %.10f\n",ground_truth_shapes[1][j][0], ground_truth_shapes[1][j][1]);*/
}

void load_inner_land_corr(Eigen::VectorXi &cor) {
	puts("loading inner landmarks correspondence...");
	FILE *fp;
	fopen_s(&fp, "./inner_jaw/inner_vertex_corr.txt", "r");
	for (int i = 0; i < G_inner_land_num; i++) fscanf_s(fp, "%d", &cor(i));
	//std::cout << cor <<"------------------------------\n"<< '\n';
	fclose(fp);
}
void load_jaw_land_corr(Eigen::VectorXi &jaw_cor) {
	puts("loading jaw landmarks correspondence...");
	FILE *fp;
	fopen_s(&fp, "./inner_jaw/jaw_vertex.txt", "r");
	for (int i = 0; i < G_jaw_land_num; i++) fscanf_s(fp, "%d", &jaw_cor(i));
	//std::cout << cor <<"------------------------------\n"<< '\n';
	fclose(fp);
}

void load_slt(
	std:: vector <int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
	std::string path_slt,std::string path_rect) {
	puts("loading silhouette line&vertices...");
	FILE *fp;
	fopen_s(&fp, path_slt.c_str(), "r");
	for (int i = 0; i < G_line_num; i++) {
		int num;
		fscanf_s(fp, "%d", &num);
		slt_line[i].resize(num);
		for (int j = 0; j < num; j++)
			fscanf_s(fp, "%d", &slt_line[i][j]);
	}
	fclose(fp);
	fopen_s(&fp, path_rect.c_str(), "r");
	for (int i = 0; i < 496; i++) {
		int idx, num;
		fscanf_s(fp, "%d%d", &idx, &num);
		slt_point_rect[idx].resize(num);
		for (int j = 0; j < num; j++) fscanf_s(fp, "%d%d", &slt_point_rect[idx][j].first, &slt_point_rect[idx][j].second);
	}
	fclose(fp);
}

void test_3d22dland(cv::Mat_<uchar> img, std::string path,iden *ide,int id_idx,int exp_idx) {
	FILE *fp;
	fopen_s(&fp, path.c_str(), "r");
	int num = 0;
	fscanf_s(fp, "%d", &num);
	for (int i = 0; i < num; i++) {
		cv::Mat_<uchar> temp;
		img.copyTo(temp);
		for (int j = 0; j < G_land_num; j++) {
			float x, y;
			fscanf_s(fp, "%f%f", &x, &y);
			cv::circle(
				temp, cv::Point2f(x, y),
				1, cv::Scalar(244, 244, 244), -1, 8, 0);
		}
		cv::imshow("test_image", temp);
		cv::waitKey(0);
	}
	fclose(fp);
}