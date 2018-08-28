#include "2dland.h"

void load_img_land(std::string path, std::string sfx, iden *ide, int &id_idx, std::vector< std::vector<cv::Mat_<uchar> > > &imgs) {
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
				id_idx+=load_img_land_same_id(p.assign(path).append("/").append(fileinfo.name).c_str(), sfx, ide, id_idx,img_temp);
				imgs.push_back(img_temp);
			}
			//if (id_idx > 5) break;////////////////////////////////////////////////////////////////////////////////////////////////debug
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

int load_img_land_same_id(std::string path, std::string sfx, iden *ide, int id_idx, std::vector<cv::Mat_<uchar> > &img_temp) {
	int hFile = 0,flag=0;
	struct _finddata_t fileinfo;
	std::string p;
	std::cout << path << '\n';
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1) {
		do {
			if (fileinfo.name[0] == '.') continue;
			//std::cout << fileinfo.name << '\n';
			if ((fileinfo.attrib & _A_SUBDIR)) {
				flag|=load_img_land_same_id(p.assign(path).append("/").append(fileinfo.name).c_str(), sfx, ide, id_idx,img_temp);
			}
			else {
				int len = strlen(fileinfo.name);
				if (fileinfo.name[len - 1] == 'd' && fileinfo.name[len - 2] == 'n') {
					////	
					flag = 1;
					load_land(p.assign(path).append("/").append(fileinfo.name),ide,id_idx);

					p = path + "/" + fileinfo.name;
					cv::Mat_<uchar> temp;
					load_img(p.substr(0, p.find(".land"))+sfx, temp);
					img_temp.push_back(temp);

					ide[id_idx].num++;
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
	return flag;
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
	temp = cv::imread(p, 0);
}

void test_data_2dland(
	std::vector < std::vector <cv::Mat_<uchar> > >& imgs,
	iden *ide, int id_idx, int img_idx) {
	int mi = img_idx * G_land_num + 70;
	for (int i = img_idx* G_land_num; i < (img_idx+1)*G_land_num; i++) {
		cv::circle(
			imgs[id_idx][img_idx], 
			cv::Point2f(ide[id_idx].land_2d(i, 0), imgs[id_idx][img_idx].rows-ide[id_idx].land_2d(i, 1)),
			1, cv::Scalar(244, 244, 244), -1, 8, 0);
		/*rectangle(images[j], Point2d(bounding_box[j].start_x, bounding_box[j].start_y)
			, Point2d(bounding_box[j].start_x + bounding_box[j].width, bounding_box[j].start_y + bounding_box[j].height)
			, Scalar(255, 244, 244), 3, 4, 0);*/
	}
	//for (int i = mi; i < (img_idx + 1)*land_num; i++) {
		CvFont font;
		cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 0.01, 0.5, 1, 0.5, 8);
		IplImage *pImage = cvLoadImage("D:\\sydney\\first\\data\\Tester_ (28)\\TrainingPose/pose_17.png");
		std::string si;
		for (int i = img_idx * G_land_num; i < img_idx * G_land_num + 15; i++) {
			si = std::to_string(i - img_idx * G_land_num), cvPutText(pImage, si.c_str(), cv::Point2f(ide[id_idx].land_2d(i, 0), imgs[id_idx][img_idx].rows - ide[id_idx].land_2d(i, 1))
				, &font, cv::Scalar(255, 255, 255));
			cv::circle(
				imgs[id_idx][img_idx],
				cv::Point2f(ide[id_idx].land_2d(i, 0), imgs[id_idx][img_idx].rows - ide[id_idx].land_2d(i, 1)),
				3, cv::Scalar(244, 244, 244), -1, 8, 0);
		}
		cvShowImage("Original", pImage);
		/*rectangle(images[j], Point2d(bounding_box[j].start_x, bounding_box[j].start_y)
		, Point2d(bounding_box[j].start_x + bounding_box[j].width, bounding_box[j].start_y + bounding_box[j].height)
		, Scalar(255, 244, 244), 3, 4, 0);*/
	//}
	cv::imshow("test_image", imgs[id_idx][img_idx]);
	cv::waitKey(0);
	/*for (int j = 0; j < landmark_num; j++)
	printf("%.10f %.10f\n",ground_truth_shapes[1][j][0], ground_truth_shapes[1][j][1]);*/
}

void load_inner_land_corr(Eigen::VectorXi &cor) {
	puts("loading inner landmarks correspondence...");
	FILE *fp;
	fopen_s(&fp, "inner_vertex_corr.txt", "r");
	for (int i = 0; i < 62; i++) fscanf_s(fp, "%d", &cor(i));
	//std::cout << cor <<"------------------------------\n"<< '\n';
	fclose(fp);
}

void load_slt(std:: vector <int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect) {
	puts("loading silhouette line&vertices...");
	FILE *fp;
	fopen_s(&fp, "D:\\openframework\\of_v0.10.0_vs2017_release\\apps\\3d22d\\3d22d/sillht.txt", "r");
	for (int i = 0; i < G_line_num; i++) {
		int num;
		fscanf_s(fp, "%d", &num);
		slt_line[i].resize(num);
		for (int j = 0; j < num; j++)
			fscanf_s(fp, "%d", &slt_line[i][j]);
	}
	fclose(fp);
	fopen_s(&fp, "D:\\openframework\\of_v0.10.0_vs2017_release\\apps\\3d22d\\3d22d/slt_point_rect.txt", "r");
	for (int i = 0; i < 605; i++) {
		int idx, num;
		fscanf_s(fp, "%d%d", &idx, &num);
		slt_point_rect[idx].resize(num);
		for (int j = 0; j < num; j++) fscanf_s(fp, "%d%d", &slt_point_rect[idx][j].first, &slt_point_rect[idx][j].second);
	}
	fclose(fp);
}