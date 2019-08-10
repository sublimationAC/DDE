#include "2dland.h"
//#define flap_2dland

//#define cal_land_num
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
	struct dirent **namelist;
	int n;
	n = scandir(path.c_str(), &namelist, 0, alphasort);
	if (n < 0)
	{
		std :: cout << "scandir return " << n << "\n";
		perror("Cannot open .");
		exit(1);
	}
	else
	{
		int index = 0;
		struct dirent *dp;
		while (index < n)
		{
			dp = namelist[index];
			std::cout << dp->d_name << ' ' << strlen(dp->d_name) << "\n";
			if (dp->d_name[0] == '.') {
				free(namelist[index]);
				index++;
				continue;
			}
			if (dp->d_type == DT_DIR) {
				std::vector<cv::Mat_<uchar> > img_temp;
				printf("Loading identity %d...\n", id_idx);
				id_idx += load_img_land_same_id(path + "/" + dp->d_name, sfx, ide, id_idx, img_temp);
				imgs.push_back(img_temp);
			}
			free(namelist[index]);
			index++;
		}
		free(namelist);
	}

	/*DIR *dir;
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
	closedir(dir);*/
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
				if (fileinfo.name[len - 3] == 'd' && fileinfo.name[len - 4] == 'n' &&
					fileinfo.name[len - 1] == '3' && fileinfo.name[len - 2] == '7') {
					////	
					flag = 1;
					load_land(p.assign(path).append("/").append(fileinfo.name), ide, id_idx);

					p = path + "/" + fileinfo.name;
					cv::Mat_<uchar> temp;
					load_img(p.substr(0, p.find(".land73")) + sfx, temp);
					img_temp.push_back(temp);

#ifdef  flap_2dland
					for (int i = G_land_num * (ide[id_idx].num); i < G_land_num*(ide[id_idx].num + 1); i++)
						ide[id_idx].land_2d(i, 1)=temp.rows- ide[id_idx].land_2d(i, 1);					
#endif //  flap_2dland

					ide[id_idx].img_size.conservativeResize(ide[id_idx].num + 1, 2);
					ide[id_idx].img_size(ide[id_idx].num, 0) = temp.cols;
					ide[id_idx].img_size(ide[id_idx].num, 1) = temp.rows;

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
			dp = namelist[index];
			if (dp->d_name[0] == '.') {
				free(namelist[index]);
				index++;
				continue;
			}
			if (dp->d_type == DT_DIR)
				flag |= load_img_land_same_id(path + "/" + dp->d_name, sfx, ide, id_idx, img_temp);
			else {
				int len = strlen(dp->d_name);
				if (dp->d_name[len - 3] == 'd' && dp->d_name[len - 4] == 'n' &&
					dp->d_name[len - 1] == '3' && dp->d_name[len - 2] == '7') {
					////	
					flag = 1;
					load_land(path + "/" + dp->d_name, ide, id_idx);
					std::string p = path + "/" + dp->d_name;
					cv::Mat_<uchar> temp;
					load_img(p.substr(0, p.find(".land73")) + sfx, temp);
					img_temp.push_back(temp);
#ifdef  flap_2dland
					for (int i = G_land_num * (ide[id_idx].num); i < G_land_num*(ide[id_idx].num + 1); i++)
						ide[id_idx].land_2d(i, 1) = temp.rows - ide[id_idx].land_2d(i, 1);
#endif //  flap_2dland
					ide[id_idx].img_size.conservativeResize(ide[id_idx].num + 1, 2);
					ide[id_idx].img_size(ide[id_idx].num, 0) = temp.cols;
					ide[id_idx].img_size(ide[id_idx].num, 1) = temp.rows;
					ide[id_idx].num++;
				}
			}
			free(namelist[index]);
			index++;

		}
		free(namelist);
	}

//	DIR *dir;
//	struct dirent *dp;
//	if ((dir = opendir(path.c_str())) == NULL) {
//		perror("Cannot open .");
//		exit(1);
//	}
//	while ((dp = readdir(dir)) != NULL) {
//		//std::cout << dp->d_name << ' ' << dp->d_namlen << "\n";
//		if (dp->d_name[0] == '.') continue;
//		if (dp->d_type == DT_DIR)
//			flag |= load_img_land_same_id(path + "/" + dp->d_name, sfx, ide, id_idx, img_temp);
//		else {
//			int len =strlen( dp->d_name);
//			if (dp->d_name[len - 1] == 'd' && dp->d_name[len - 2] == 'n') {
//				////	
//				flag = 1;
//				load_land(path + "/" + dp->d_name, ide, id_idx);
//				std::string p = path + "/" + dp->d_name;
//				cv::Mat_<uchar> temp;
//				load_img(p.substr(0, p.find(".land")) + sfx, temp);
//				img_temp.push_back(temp);
//#ifdef  flap_2dland
//				for (int i = G_land_num * (ide[id_idx].num); i < G_land_num*(ide[id_idx].num + 1); i++)
//					ide[id_idx].land_2d(i, 1) = temp.rows - ide[id_idx].land_2d(i, 1);
//#endif //  flap_2dland
//				ide[id_idx].num++;
//			}
//		}
//	}
//	closedir(dir);


	return flag;
#endif // linux
}

void load_land(std::string p, iden *ide, int id_idx) {
	std::cout << p << '\n';
	FILE *fp;
	fp=fopen( p.c_str(), "r");
	int n;
	fscanf(fp, "%d", &n);
	assert(n == G_land_num);
	ide[id_idx].land_2d.conservativeResize(n*(ide[id_idx].num + 1), 2);
	for (int i = n * (ide[id_idx].num); i < n*(ide[id_idx].num + 1); i++)
		fscanf(fp, "%f%f", &ide[id_idx].land_2d(i, 0), &ide[id_idx].land_2d(i, 1));
	fclose(fp);
}

void load_img(std::string p, cv::Mat_<uchar> &temp) {
	temp = cv::imread(p , 0);
	//cv::imshow("a", temp);
	//cv::waitKey(0);
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
	printf("%.6f %.6f\n",ground_truth_shapes[1][j][0], ground_truth_shapes[1][j][1]);*/
}

void test_2dslt(
	std::vector < std::vector <cv::Mat_<uchar> > >& imgs,
	iden *ide, int id_idx, int img_idx) {
#ifdef cal_land_num
	//int mi = img_idx * G_land_num + 70;
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 0.05, 0.7, 1, 0.5, 8);
	IplImage *pImage = cvLoadImage("D:\\sydney\\first\\data_me\\FaceWarehouse\\Tester_6\\TrainingPose/pose_0.jpg");
	std::string si;
	Eigen::RowVector2f center;
	center.setZero();
	for (int i = 0; i < G_land_num; i++)
		center += ide[id_idx].land_2d.row(img_idx * G_land_num + i);
	center.array() /= G_land_num;
	for (int i = img_idx * G_land_num; i < img_idx * G_land_num + 74; i++) {
		//27-35 66-74
		si = std::to_string(i - img_idx * G_land_num), cvPutText(pImage, si.c_str()
			, cv::Point2f(ide[id_idx].land_2d(i, 0)*3- center(0)*2, 
				(imgs[id_idx][img_idx].rows - ide[id_idx].land_2d(i, 1))*3-center(1) * 2)
			, &font, cv::Scalar(255, 255, 255));
		cv::circle(
			imgs[id_idx][img_idx],
			cv::Point2f(ide[id_idx].land_2d(i, 0), imgs[id_idx][img_idx].rows - ide[id_idx].land_2d(i, 1)),
			3, cv::Scalar(244, 244, 244), -1, 8, 0);
	}
	cvShowImage("Original", pImage);
	cv::waitKey(0);
	/*rectangle(images[j], Point2d(bounding_box[j].start_x, bounding_box[j].start_y)
	, Point2d(bounding_box[j].start_x + bounding_box[j].width, bounding_box[j].start_y + bounding_box[j].height)
	, Scalar(255, 244, 244), 3, 4, 0);*/

#endif // cal_land_num
	cv::Mat_<uchar> image = cv::imread("D:\\sydney\\first\\code\\2017\\DDE\\FaceX\\photo_test\\test_samples\\22.png");
	cv::imshow("test_image", image);// imgs[id_idx][img_idx]);
	cv::waitKey(0);
	FILE *fp;
	fp=fopen( "./test_slt_vtx.txt", "r");
	int n;
	fscanf(fp,"%d",&n);
	for (int i = 0; i < n; i++) {
		float x, y;
		fscanf(fp, "%f%f", &x, &y);
		cv::circle(
			image,//imgs[id_idx][img_idx],
			cv::Point2f(x, y),
			1, cv::Scalar(244, 244, 244), -1, 8, 0);
		//if (i<=12) 
		//	cv::circle(
		//	imgs[id_idx][img_idx],
		//	cv::Point2f(x, y),
		//	3, cv::Scalar(244, 244, 244), -1, 8, 0);
	}
	cv::imshow("test_image", image);// imgs[id_idx][img_idx]);
	cv::waitKey(0);
	fclose(fp);
	/*for (int j = 0; j < landmark_num; j++)
	printf("%.6f %.6f\n",ground_truth_shapes[1][j][0], ground_truth_shapes[1][j][1]);*/
}

void load_inner_land_corr(Eigen::VectorXi &cor) {
	puts("loading inner landmarks correspondence...");
	FILE *fp;
	fp=fopen( "../const_file/inner_jaw/inner_vertex_corr_58_416ans.txt", "r");
	for (int i = 0; i < G_inner_land_num; i++) fscanf(fp, "%d", &cor(i));
	//std::cout << cor <<"------------------------------\n"<< '\n';
	fclose(fp);
}
void load_jaw_land_corr(Eigen::VectorXi &jaw_cor) {
	/*
	puts("loading jaw landmarks correspondence...");
	FILE *fp;
	fp=fopen( "./inner_jaw/jaw_vertex.txt", "r");
	for (int i = 0; i < G_jaw_land_num; i++) fscanf(fp, "%d", &jaw_cor(i));
	//std::cout << cor <<"------------------------------\n"<< '\n';
	fclose(fp);
	*/
}

void load_slt(
	std:: vector <int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
	std::string path_slt,std::string path_rect) {
	puts("loading silhouette line&vertices...");
	FILE *fp;
	fp=fopen( path_slt.c_str(), "r");
	int line_num;
	fscanf(fp, "%d", &line_num);
	if (line_num != G_line_num) {
		puts("Line num error!!!");
		exit(-1);
	}
	for (int i = 0; i < line_num; i++) {
		int x,num;

		fscanf(fp, "%d%d",&x, &num);
		slt_line[i].resize(num);
		for (int j = 0; j < num; j++)
			fscanf(fp, "%d", &slt_line[i][j]);
	}
	fclose(fp);
	fp=fopen( path_rect.c_str(), "r");
	int vtx_num;
	fscanf(fp, "%d", &vtx_num);
	for (int i = 0; i < vtx_num; i++) {
		int idx, num;
		fscanf(fp, "%d%d", &idx, &num);
		slt_point_rect[idx].resize(num);
		for (int j = 0; j < num; j++) fscanf(fp, "%d%d", &slt_point_rect[idx][j].first, &slt_point_rect[idx][j].second);
	}
	fclose(fp);
}

void test_3d22dland(cv::Mat_<uchar> img, std::string path,iden *ide,int id_idx,int exp_idx) {
	FILE *fp;
	fp=fopen( path.c_str(), "r");
	int num = 0;
	fscanf(fp, "%d", &num);
	for (int i = 0; i < num; i++) {
		cv::Mat_<uchar> temp;
		img.copyTo(temp);
		for (int j = 0; j < G_land_num; j++) {
			float x, y;
			fscanf(fp, "%f%f", &x, &y);
			cv::circle(
				temp, cv::Point2f(x, y),
				1, cv::Scalar(244, 244, 244), -1, 8, 0);
		}
		cv::imshow("test_image", temp);
		cv::waitKey(0);
	}
	fclose(fp);
}
//void test_slt_2dland(cv::Mat_<uchar> img, std::string path, iden *ide, int id_idx, int exp_idx) {
//
//	FILE *fp;
//	fp=fopen( path.c_str(), "r");
//	int num = 0;
//	fscanf(fp, "%d", &num);
//	for (int i = 0; i < num; i++) {
//		cv::Mat temp;
//		//temp = cv::imread("D:/sydney/first/data_me/test_only_one_2d/t_1/lv_out_first_frame.jpg");
//		temp = cv::imread("D:/sydney/first/data_me/test_only_one_2d/t_1/pose_1.jpg");
//
//		for (int j = 0; j < G_line_num; j++) {
//			float x, y;
//			fscanf(fp, "%f%f", &x, &y);
//			y = temp.rows - y;
//			cv::circle(
//				temp, cv::Point2f(x, y),
//				1, cv::Scalar(244, 244, 244), -1, 8, 0);
//		}
//		for (int j = 0; j < G_land_num; j++) {
//			float x, y;
//			x = ide[id_idx].land_2d(exp_idx*G_land_num + j, 0);
//			y = ide[id_idx].land_2d(exp_idx*G_land_num + j, 1);
//			y = temp.rows - y;
//			cv::circle(
//				temp, cv::Point2f(x, y),
//				2, cv::Scalar(0, 244, 0), -1, 8, 0);
//		}
//		cv::imshow("test_image", temp);
//		cv::waitKey(0);
//	}
//	fclose(fp);
//}
void test_slt_me_2dland(cv::Mat_<uchar> img, std::string path, iden *ide, int id_idx, int exp_idx) {
	
	FILE *fp;
	fp=fopen( path.c_str(), "r");
	int num = 1;
	fscanf(fp, "%d", &num);
	for (int i = 0; i < num; i++) {
		cv::Mat temp;
		//temp = cv::imread("D:/sydney/first/data_me/test_only_one_2d_me/t_1/lv_out_first_frame.jpg");		
		//temp = cv::imread("D:/sydney/first/data_me/test_only_one_2d/t_1/pose_0.jpg");
		//temp = cv::imread("D:/sydney/first/data_me/test_only_one_2d_left/t_1/pose_0.jpg");
		temp = cv::imread("D:/sydney/first/data_me/test_only_one_2d_right/t_1/ID24_007.bmp");
		//temp = cv::imread("D:/sydney/first/data_me/test_only_one_2d_errbig/t_1/George_W_Bush_0459.jpg");
		int ptnum;
		fscanf(fp, "%d", &ptnum);
		printf("%d\n", ptnum);
		for (int j = 0; j < ptnum; j++) {
			float x, y;
			fscanf(fp, "%f%f", &x, &y);
			y = temp.rows - y;
			if (j > ptnum / 2) continue;
			cv::circle(
				temp, cv::Point2f(x, y),
				1, cv::Scalar(244, 244, 244), -1, 8, 0);
		}
		for (int j = 0; j < G_land_num; j++) {
			float x, y;
			x = ide[id_idx].land_2d(exp_idx*G_land_num + j, 0);
			y = ide[id_idx].land_2d(exp_idx*G_land_num + j, 1);
			y = temp.rows - y;
			cv::circle(
				temp, cv::Point2f(x, y),
				2, cv::Scalar(0, 244, 0), -1, 8, 0);
		}
		cv::imshow("test_image", temp);
		cv::waitKey(0);
	}
	fclose(fp);
}

void test_inner_2dland(cv::Mat_<uchar> img, std::string path, iden *ide, int id_idx, int exp_idx) {

	FILE *fp;
	fp=fopen( path.c_str(), "r");
	int num = 0;
	fscanf(fp, "%d", &num);
	for (int i = 0; i < num; i++) {
		cv::Mat temp;
		//temp = cv::imread("D:/sydney/first/data_me/test_only_one_2d_me/t_1/lv_out_first_frame.jpg");
		//temp = cv::imread("D:/sydney/first/data_me/test_only_one_2d/t_1/pose_0.jpg");
		//temp = cv::imread("D:/sydney/first/data_me/test_only_one_2d_left/t_1/pose_0.jpg");
		temp = cv::imread("D:/sydney/first/data_me/test_only_one_2d_right/t_1/ID24_007.bmp");
		//temp = cv::imread("D:/sydney/first/data_me/test_only_one_2d_errbig/t_1/George_W_Bush_0459.jpg");
		printf("%d  %d %d\n",i, temp.rows, temp.cols);



		for (int j = 0; j < 7; j++) {
			float x, y;
			fscanf(fp, "%f%f", &x, &y);
			y = temp.rows - y;
			cv::circle(
				temp, cv::Point2f(x, y),
				2, cv::Scalar(0, 0, 244), -1, 8, 0);
		}
		for (int j = 7; j < 15; j++) {
			float x, y;
			fscanf(fp, "%f%f", &x, &y);
			y = temp.rows - y;
			cv::circle(
				temp, cv::Point2f(x, y),
				2, cv::Scalar(0, 244, 244), -1, 8, 0);
		}
		for (int j = 15; j < G_land_num; j++) {
			float x, y;
			fscanf(fp, "%f%f", &x, &y);
			y = temp.rows - y;
			cv::circle(
				temp, cv::Point2f(x, y),
				1, cv::Scalar(244, 244, 244), -1, 8, 0);
		}
		for (int j = 0; j < 15; j++) {
			float x, y;
			x = ide[id_idx].land_2d(exp_idx*G_land_num + j, 0);
			y = ide[id_idx].land_2d(exp_idx*G_land_num + j, 1);
			y = temp.rows - y;
			cv::circle(
				temp, cv::Point2f(x, y),
				2, cv::Scalar(0, 244, 0), -1, 8, 0);
		}
		for (int j = 15; j < G_land_num; j++) {
			float x, y;
			x = ide[id_idx].land_2d(exp_idx*G_land_num + j, 0);
			y = ide[id_idx].land_2d(exp_idx*G_land_num + j, 1);
			y = temp.rows - y;
			cv::circle(
				temp, cv::Point2f(x, y),
				0.5, cv::Scalar(0, 244, 0), -1, 8, 0);
		}
		cv::imshow("test_image", temp);
		cv::waitKey(0);
	}
	fclose(fp);
}

void save_result(iden *ide, int tot_id, std ::string name) {
	std ::cout << "saving coefficients...file:" << name << "\n";
	FILE *fp;
	fp=fopen( name.c_str(), "wb");

	fwrite(&tot_id, sizeof(int), 1, fp);
	for (int i_id = 0; i_id < tot_id; i_id++) {
		fwrite(&ide[i_id].num, sizeof(int), 1, fp);
		for (int j = 0; j < G_iden_num; j++)
			fwrite(&ide[i_id].user(j), sizeof(float), 1, fp);
		for (int exp_idx = 0; exp_idx < ide[i_id].num; exp_idx++) {
			for (int i_v = 0; i_v < G_land_num; i_v++) {
				fwrite(&ide[i_id].land_2d(exp_idx*G_land_num + i_v, 0), sizeof(float), 1, fp);
				fwrite(&ide[i_id].land_2d(exp_idx*G_land_num + i_v, 1), sizeof(float), 1, fp);
			}

			fwrite(&ide[i_id].center(exp_idx, 0), sizeof(float), 1, fp);
			fwrite(&ide[i_id].center(exp_idx, 1), sizeof(float), 1, fp);

			for(int i_shape=0;i_shape<G_nShape;i_shape++)
				fwrite(&ide[i_id].exp(exp_idx,i_shape), sizeof(float), 1, fp);

			for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
				fwrite(&ide[i_id].rot(exp_idx * 3 + i, j), sizeof(float), 1, fp);

			for (int i=0;i<3;i++) fwrite(&ide[i_id].tslt(exp_idx,i), sizeof(float), 1, fp);

			for (int i_v = 0; i_v < G_land_num; i_v++) fwrite(&ide[i_id].land_cor(exp_idx, i_v), sizeof(int), 1, fp);

#ifdef posit			
			fwrite(&ide[i_id].fcs, sizeof(float), 1, fp);
#endif // posit
#ifdef normalization
			for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++)
				fwrite(&ide[i_id].s(exp_idx * 2 + i, j), sizeof(float), 1, fp);

#endif // normalization

			
			for (int i_v = 0; i_v < G_land_num; i_v++) {
				fwrite(&ide[i_id].dis(exp_idx*G_land_num + i_v, 0), sizeof(float), 1, fp);
				fwrite(&ide[i_id].dis(exp_idx*G_land_num + i_v, 1), sizeof(float), 1, fp);
			}
		}
	}
	fclose(fp);

	puts("save successful!");
}

//float cal_3d_vtx(
//	iden *ide, Eigen::MatrixXf &bldshps,
//	int id_idx, int exp_idx, int vtx_idx, int axis) {
//
//	//puts("calculating one vertex coordinate...");
//	float ans = 0;
//
//	for (int i_id = 0; i_id < G_iden_num; i_id++)
//		for (int i_shape = 0; i_shape < G_nShape; i_shape++)
//			if (i_shape == 0)
//				ans += ide[id_idx].exp(exp_idx, i_shape)*ide[id_idx].user(i_id)
//				*bldshps(i_id, vtx_idx * 3 + axis);
//			else
//				ans += ide[id_idx].exp(exp_idx, i_shape)*ide[id_idx].user(i_id)
//				*(bldshps(i_id, 3 * G_nVerts*i_shape + vtx_idx * 3 + axis) - bldshps(i_id, vtx_idx * 3 + axis));
//	return ans;
//}
float cal_3d_vtx(
	iden *ide, Eigen::MatrixXf &bldshps,
	int id_idx, int exp_idx, int vtx_idx, int axis) {

	//puts("calculating one vertex coordinate...");
	float ans = 0;

	for (int i_id = 0; i_id < G_iden_num; i_id++)
		for (int i_shape = 0; i_shape < G_nShape; i_shape++)
			if (i_shape == 0)
				ans += ide[id_idx].exp(exp_idx, i_shape)*ide[id_idx].user(i_id)
				*bldshps(i_id, vtx_idx * 3 + axis);
			else
				ans += ide[id_idx].exp(exp_idx, i_shape)*ide[id_idx].user(i_id)
				*(bldshps(i_id, 3 * G_nVerts*i_shape + vtx_idx * 3 + axis) - bldshps(i_id, vtx_idx * 3 + axis));
	return ans;
}
void cal_dis(iden *ide, Eigen::MatrixXf &bldshps, int id_tot) {
	puts("calculating displacement...");
	for (int i_id = 0; i_id < id_tot; i_id++) {
		Eigen::MatrixX2f land(G_land_num*ide[i_id].num,2);
		for (int exp_idx = 0; exp_idx < ide[i_id].num; exp_idx++) {
			Eigen::Matrix3f R = ide[i_id].rot.block(exp_idx * 3, 0, 3, 3);
#ifdef normalization
			Eigen::MatrixX3f s = ide[i_id].s.block(exp_idx * 2, 0, 2, 3);
			Eigen::RowVector2f T = ide[i_id].tslt.block(exp_idx, 0, 1, 2);
#endif

			
			for (int i_v = 0; i_v < G_land_num; i_v++) {
				Eigen::Vector3f v;
				for (int axis = 0; axis < 3; axis++)
					v(axis) = cal_3d_vtx(ide, bldshps, i_id, exp_idx, ide[i_id].land_cor(exp_idx, i_v), axis);
#ifdef posit
				v = R * v + ide[i_id].tslt.row(exp_idx).transpose();

				land(exp_idx*G_land_num + i_v, 0) = v(0)*(ide[i_id].fcs) / v(2) + ide[i_id].center(exp_idx, 0);
				land(exp_idx*G_land_num + i_v, 1) = v(1)*(ide[i_id].fcs) / v(2) + ide[i_id].center(exp_idx, 1);
#endif // posit				
#ifdef normalization
				land.row(exp_idx*G_land_num + i_v) = (s*R*v).transpose() + T;				
#endif // normalization
			}
		}
		ide[i_id].dis.array() = ide[i_id].land_2d.array() - land.array();
	}
}

float print_error(float fcs, iden *ide, Eigen::MatrixXf &bldshps, int i_id, int exp_idx) {
	Eigen::MatrixX2f land;
	cal_2dland_fidexp(fcs, ide, bldshps, land, i_id, exp_idx);
	Eigen::MatrixX2f dis = ide[i_id].land_2d.block(G_land_num*exp_idx, 0, G_land_num, 2) - land;
	const int left_eye_num = 8;
	int idx_lft_eye[left_eye_num] = { 27,28,29,30,66,67,68,65 };
	const int right_eye_num = 8;
	int idx_rt_eye[right_eye_num] = { 31,32,33,34,70,71,72,69 };

	const int lft_bn_num = 6;
	int idx_lft_bn[lft_bn_num] = { 21,22,23,24,25,26 };
	const int rt_bn_num = 6;
	int idx_rt_bn[rt_bn_num] = { 15,16,17,18,19,20 };

	const int ms_be = 46, ms_ed = 64;
	const int ns_be = 35, ns_ed = 46;

	int id = i_id, i_exp = exp_idx;

	double err_slt = 0, err_eye_left = 0, err_eye_right = 0, err_lft_bn = 0, err_rt_bn = 0, err_ms = 0, err_ns = 0;
	for (int i = 0; i < 15; i++) err_slt += dis.row( i).squaredNorm();

	for (int i = 0; i < left_eye_num; i++) err_eye_left += dis.row( idx_lft_eye[i]).squaredNorm();
	for (int i = 0; i < right_eye_num; i++) err_eye_right += dis.row( idx_rt_eye[i]).squaredNorm();

	for (int i = 0; i < lft_bn_num; i++) err_lft_bn += dis.row( idx_lft_bn[i]).squaredNorm();
	for (int i = 0; i < rt_bn_num; i++) err_rt_bn += dis.row( idx_rt_bn[i]).squaredNorm();

	for (int i = ms_be; i < ms_ed; i++) err_ms += dis.row( i).squaredNorm();
	for (int i = ns_be; i < ns_ed; i++) err_ns += dis.row( i).squaredNorm();

	printf("id:%d exp: %d\n", i_id, exp_idx);
	printf("slt err:%.5f\nlft eye:%.5f  rt eye:%.5f\n", err_slt / 15, err_eye_left / left_eye_num, err_eye_right / right_eye_num);
	printf("lft bn:%.5f  rt bn:%.5f\n", err_lft_bn / lft_bn_num, err_rt_bn / rt_bn_num);
	printf("ms:%.5f  ns:%.5f\n", err_ms / (ms_ed - ms_be + 1), err_ns / (ns_ed - ns_be + 1));


	for (int i = 0; i < G_land_num; i++)
		printf("%d %.5f %.5f\n", i, dis( i, 0), dis( i, 1));

	printf("tot average error:%.5f\n", sqrt(dis.squaredNorm() / G_land_num / 2));
	return sqrt(dis.squaredNorm() / G_land_num / 2);
}


void cal_2dland_fidexp(float fcs,iden *ide, Eigen::MatrixXf &bldshps, Eigen::MatrixX2f &land, int i_id, int exp_idx) {

	land.resize(G_land_num, 2);

	Eigen::Matrix3f R = ide[i_id].rot.block(exp_idx * 3, 0, 3, 3);
#ifdef normalization
	Eigen::MatrixX3f s = ide[i_id].s.block(exp_idx * 2, 0, 2, 3);
	Eigen::RowVector2f T = ide[i_id].tslt.block(exp_idx, 0, 1, 2);
#endif


	for (int i_v = 0; i_v < G_land_num; i_v++) {
		Eigen::Vector3f v;
		for (int axis = 0; axis < 3; axis++)
			v(axis) = cal_3d_vtx(ide, bldshps, i_id, exp_idx, ide[i_id].land_cor(exp_idx, i_v), axis);
#ifdef posit
		v = R * v + ide[i_id].tslt.row(exp_idx).transpose();

		land( i_v, 0) = v(0)*(fcs) / v(2) + ide[i_id].center(exp_idx, 0);
		land(i_v, 1) = v(1)*(fcs) / v(2) + ide[i_id].center(exp_idx, 1);
#endif // posit				
#ifdef normalization
		land.row(i_v) = (s*R*v).transpose() + T;
#endif // normalization
	}


}

void save_result_one(iden *ide, int i_id, int exp_idx, std::string name) {
	std::cout << "saving coefficients...file:" << name << "\n";
	FILE *fp;
	fp=fopen( name.c_str(), "wb");
	for (int j = 0; j < G_iden_num; j++)
		fwrite(&ide[i_id].user(j), sizeof(float), 1, fp);

	for (int i_v = 0; i_v < G_land_num; i_v++) {
		fwrite(&ide[i_id].land_2d(exp_idx*G_land_num + i_v, 0), sizeof(float), 1, fp);
		fwrite(&ide[i_id].land_2d(exp_idx*G_land_num + i_v, 1), sizeof(float), 1, fp);
	}

	fwrite(&ide[i_id].center(exp_idx, 0), sizeof(float), 1, fp);
	fwrite(&ide[i_id].center(exp_idx, 1), sizeof(float), 1, fp);

	for (int i_shape = 0; i_shape < G_nShape; i_shape++)
		fwrite(&ide[i_id].exp(exp_idx, i_shape), sizeof(float), 1, fp);

	for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
		fwrite(&ide[i_id].rot(exp_idx * 3 + i, j), sizeof(float), 1, fp);

#ifdef normalization
	ide[i_id].tslt(exp_idx, 2)=0;
#endif
	for (int i = 0; i < 3; i++) fwrite(&ide[i_id].tslt(exp_idx, i), sizeof(float), 1, fp);

	for (int i_v = 0; i_v < G_land_num; i_v++) fwrite(&ide[i_id].land_cor(exp_idx, i_v), sizeof(int), 1, fp);

#ifdef posit			
	fwrite(&ide[i_id].fcs, sizeof(float), 1, fp);
#endif // posit
#ifdef normalization
	for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++)
		fwrite(&ide[i_id].s(exp_idx * 2 + i, j), sizeof(float), 1, fp);

#endif // normalization

	for (int i_v = 0; i_v < G_land_num; i_v++) {
		fwrite(&ide[i_id].dis(exp_idx*G_land_num + i_v, 0), sizeof(float), 1, fp);
		fwrite(&ide[i_id].dis(exp_idx*G_land_num + i_v, 1), sizeof(float), 1, fp);
	}
	// std::cout << "dis : \n" << ide[i_id].dis << "\n";
	fclose(fp);

	puts("save successful!");
}




void save_fitting_coef_each(std::string path, iden *ide, int &id_idx) {

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
			dp = namelist[index];
			std::cout << dp->d_name << ' ' << strlen(dp->d_name) << "\n";
			if (dp->d_name[0] == '.') {
				free(namelist[index]);
				index++;
				continue;
			}
			if (dp->d_type == DT_DIR) {
				printf("save_fitting_coef %d...\n", id_idx);
				int exp_idx = 0, temp = ide[id_idx].num;
				id_idx += save_fitting_coef_same_id(path + "/" + dp->d_name, ide, id_idx,exp_idx);
				if (exp_idx!=0 && exp_idx != temp) {
					std::cout << "error !!! \n";
					perror("Not match .");
					exit(1);
				}
			}
			free(namelist[index]);
			index++;
		}
		free(namelist);
	}

}

int save_fitting_coef_same_id(std::string path, iden *ide, int id_idx,int &exp_idx) {
	puts("save result same id");
	int flag = 0;
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
			dp = namelist[index];
			if (dp->d_name[0] == '.') {
				free(namelist[index]);
				index++;
				continue;
			}
			if (dp->d_type == DT_DIR)
				flag |= save_fitting_coef_same_id(path + "/" + dp->d_name, ide, id_idx, exp_idx);
			else {
				int len = strlen(dp->d_name);
				if (dp->d_name[len - 3] == 'd' && dp->d_name[len - 4] == 'n' &&
					dp->d_name[len - 1] == '3' && dp->d_name[len - 2] == '7') {
					////	
					flag = 1;
					std::string p = path + "/" + dp->d_name;
					save_result_one(ide, id_idx, exp_idx, p.substr(0, p.find(".land")) + ".psp_f");
					exp_idx++;
				}
			}
			free(namelist[index]);
			index++;
		}
		free(namelist);
	}

	return flag;
}

void load_mean_mesh(std::string mean_mesh_path,Eigen::MatrixX3f &mean_mesh) {
	mean_mesh.resize(G_nVerts, 3);
	FILE *fp;
	fp = fopen(mean_mesh_path.c_str(), "r");
	for (int i = 0; i < G_nVerts; i++)
		for (int j = 0; j < 3; j++)
			scanf("%f", &mean_mesh(i, j));
	fclose(fp);
}