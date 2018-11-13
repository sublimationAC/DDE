#include "2dland.h"
//#define flap_2dland

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
	printf("%.6f %.6f\n",ground_truth_shapes[1][j][0], ground_truth_shapes[1][j][1]);*/
}

void test_2dslt(
	std::vector < std::vector <cv::Mat_<uchar> > >& imgs,
	iden *ide, int id_idx, int img_idx) {
#ifdef cal_land_num
	//int mi = img_idx * G_land_num + 70;
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 0.01, 0.5, 1, 0.5, 8);
	IplImage *pImage = cvLoadImage("D:\\sydney\\first\\data_me\\FaceWarehouse\\Tester_6\\TrainingPose/pose_17.jpg");
	std::string si;
	for (int i = img_idx * G_land_num + 46; i < img_idx * G_land_num + 53; i++) {
		//27-35 66-74
		si = std::to_string(i - img_idx * G_land_num), cvPutText(pImage, si.c_str()
			, cv::Point2f(ide[id_idx].land_2d(i, 0), imgs[id_idx][img_idx].rows - ide[id_idx].land_2d(i, 1))
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

#endif // cal_land_num
	FILE *fp;
	fopen_s(&fp, "./test_slt_vtx.txt", "r");
	int n;
	fscanf_s(fp,"%d",&n);
	for (int i = 0; i < n; i++) {
		float x, y;
		fscanf_s(fp, "%f%f", &x, &y);
		cv::circle(
			imgs[id_idx][img_idx],
			cv::Point2f(x, y),
			1, cv::Scalar(244, 244, 244), -1, 8, 0);
		//if (i<=12) 
		//	cv::circle(
		//	imgs[id_idx][img_idx],
		//	cv::Point2f(x, y),
		//	3, cv::Scalar(244, 244, 244), -1, 8, 0);
	}
	cv::imshow("test_image", imgs[id_idx][img_idx]);
	cv::waitKey(0);
	fclose(fp);
	/*for (int j = 0; j < landmark_num; j++)
	printf("%.6f %.6f\n",ground_truth_shapes[1][j][0], ground_truth_shapes[1][j][1]);*/
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

void save_result(iden *ide, int tot_id, std ::string name) {
	std ::cout << "saving coefficients...file:" << name << "\n";
	FILE *fp;
	fopen_s(&fp, name.c_str(), "wb");

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

			for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++)
				fwrite(&ide[i_id].s(exp_idx * 2 + i, j), sizeof(float), 1, fp);

			for (int i_v = 0; i_v < G_land_num; i_v++) {
				fwrite(&ide[i_id].dis(exp_idx*G_land_num + i_v, 0), sizeof(float), 1, fp);
				fwrite(&ide[i_id].dis(exp_idx*G_land_num + i_v, 1), sizeof(float), 1, fp);
			}
		}
	}
	fclose(fp);

	puts("save successful!");
}

float cal_3d_vtx_(
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
			Eigen::MatrixX3f s = ide[i_id].s.block(exp_idx * 2, 0, 2, 3);
			Eigen::RowVector2f T = ide[i_id].tslt.block(exp_idx, 0, 1, 2);
			for (int i_v = 0; i_v < G_land_num; i_v++) {
				Eigen::Vector3f v;
				for (int axis = 0; axis < 3; axis++)
					v(axis) = cal_3d_vtx_(ide, bldshps, i_id, exp_idx, ide[i_id].land_cor(exp_idx, i_v), axis);
				land.row(exp_idx*G_land_num + i_v) = (s*R*v).transpose() + T - ide[i_id].center.row(exp_idx);
#ifdef normalization
				land.row(exp_idx*G_land_num + i_v) -= ide[i_id].center.row(exp_idx);
#endif // normalization
			}
		}
		ide[i_id].dis.array() = ide[i_id].land_2d.array() - land.array();


	}
}

void save_result_one(iden *ide, int i_id, int exp_idx, std::string name) {
	std::cout << "saving coefficients...file:" << name << "\n";
	FILE *fp;
	fopen_s(&fp, name.c_str(), "wb");
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

	for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++)
		fwrite(&ide[i_id].s(exp_idx * 2 + i, j), sizeof(float), 1, fp);

	for (int i_v = 0; i_v < G_land_num; i_v++) {
		fwrite(&ide[i_id].dis(exp_idx*G_land_num + i_v, 0), sizeof(float), 1, fp);
		fwrite(&ide[i_id].dis(exp_idx*G_land_num + i_v, 1), sizeof(float), 1, fp);
	}

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
				int exp_idx = 0;
				id_idx += save_fitting_coef_same_id(path + "/" + dp->d_name, ide, id_idx,exp_idx);
				if (exp_idx != ide[id_idx].num) {
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
				if (dp->d_name[len - 1] == 'd' && dp->d_name[len - 2] == 'n') {
					////	
					flag = 1;
					std::string p = path + "/" + dp->d_name;
					save_result_one(ide, id_idx, exp_idx, p.substr(0, p.find(".land")) + ".lv");
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