//#include <dirent.h>
//#include <iostream>
//#include <vector>
//#include <string>
//#include <utility>
//
//#include <Eigen/Core>
//#include <Eigen/LU>
//#include <Eigen/Geometry>
//
//#include <opencv2/opencv.hpp>
//const int G_land_num = 74;
//const int G_train_pic_id_num = 3300;
//const int G_nShape = 47;
//const int G_nVerts = 11510;
//const int G_nFaces = 11540;
//const int G_test_num = 77;
//const int G_iden_num = 77;
//const int G_inner_land_num = 59;
//const int G_line_num = 50;
//const int G_jaw_land_num = 20;
//#define normalization
//#define win64
////#define linux
//
//#define cppio
//#define same_id
//
//
//struct Target_type {
//	Eigen::VectorXf exp;
//	Eigen::RowVector3f tslt;
//	Eigen::Matrix3f rot;
//	Eigen::MatrixX2f dis;
//
//};
//
//struct DataPoint
//{
//	cv::Mat image;
//	cv::Rect face_rect;
//	std::vector<cv::Point2d> landmarks;
//	//std::vector<cv::Point2d> init_shape;
//	Target_type shape, init_shape;
//	Eigen::VectorXf user;
//	Eigen::RowVector2f center;
//	Eigen::MatrixX2f land_2d;
//#ifdef posit
//	float f;
//#endif // posit
//#ifdef normalization
//	Eigen::MatrixX3f s;
//#endif
//
//	Eigen::VectorXi land_cor;
//};
//void load_bldshps(Eigen::MatrixXf &bldshps, std::string &name) {
//
//	puts("loading blendshapes...");
//	std::cout << name << std::endl;
//	FILE *fp;
//	fopen_s(&fp, name.c_str(), "rb");
//	for (int i = 0; i < G_iden_num; i++) {
//		for (int j = 0; j < G_nShape*G_nVerts * 3; j++)
//			fread(&bldshps(i, j), sizeof(float), 1, fp);
//	}
//	fclose(fp);
//}
//#ifdef cppio
//void get_mesh(Eigen::MatrixX3f& base_mesh, int mesh_vertex_num, std::string suffix[34469]) {
//#else
//void get_mesh(Eigen::MatrixX3f& base_mesh, int mesh_vertex_num, char suffix[34469][100]) {
//#endif // cppio
//
//
//
//#ifdef win64
//	std::string mesh_name = "D:/sydney/first/data/Tester_ (1)/TrainingPose/pose_0.obj";
//#endif // win64
//#ifdef linux 
//	std::string mesh_name = "pose_0.obj";
//#endif // linux 
//
//	
//#ifdef cppio
//	std::ifstream op;
//	op.open(mesh_name);
//	base_mesh.resize(G_nVerts, 3);
//	for (int j = 0; j < mesh_vertex_num; j++) {
//		char c;
//		op >> c >> base_mesh(j, 0) >> base_mesh(j, 1) >> base_mesh(j, 2);
//	}
//
//	for (int j = mesh_vertex_num; j < 34469; j++)
//		getline(op, suffix[j]);
//
//	op.close();
//#else
//	FILE *fp;
//	puts("asd");
//	fopen_s(&fp, mesh_name.c_str(), "r");
//	puts("P");
//	base_mesh.resize(G_nVerts, 3);
//	char c;
//	//fscanf_s(fp, "%c", &c);
//	//putchar(c);
//	double p;
//	//fscanf_s(fp, "%lf", &p);
//	//printf("%.6f\n",p);
//	//fscanf_s(fp, "%lf", &p);
//	//printf("%.6f\n", p);
//	//fscanf_s(fp, "%lf", &p);
//	//printf("%.6f\n", p);
//	//fscanf_s(fp, "%c", &c);
//	//printf("ww%cqq\n-",c);
//	//fscanf_s(fp, "%c", &c);
//	//putchar(c);
//	//fscanf_s(fp, "%lf", &p);
//	//printf("%.6f\n", p);
//	//fscanf_s(fp, "%lf", &p);
//	//printf("%.6f\n", p);
//	//fscanf_s(fp, "%lf", &p);
//	//printf("%.6f\n", p);
//	//fscanf_s(fp, "%c", &c);
//	//printf("ww%cqq", c);
//	for (int j = 0; j < mesh_vertex_num; j++) {
//		/*char a,b,c;
//		float x, y, z;
//		fscanf_s(fp,"%c%lf%lf%lf%c",&a,&x,&y,&z,&b);
//
//		printf("%d %c %c %.6f %.6f %.6f\n", j,a,b, x, y, z);*/
//		fscanf_s(fp, "%c", &c);
//		putchar(c);
//		fscanf_s(fp, "%lf", &p);
//		printf("%.6f\n", p);
//		fscanf_s(fp, "%lf", &p);
//		printf("%.6f\n", p);
//		fscanf_s(fp, "%lf", &p);
//		printf("%d %.6f\n", j,p);
//		fscanf_s(fp, "%c", &c);
//		printf("ww%cqq", c);
//	}
//
//	for (int j = mesh_vertex_num; j < 34469; j++)
//		fgets(suffix[j], 98, fp);
//	fclose(fp);
//#endif // cppio
//	system("pause");
//	
//
//}
//#ifdef cppio
//void print_mesh(Eigen::MatrixX3f &test_mesh_1, std::string suffix[34469], std::string name) {
//	std::ofstream out(name);
//
//	for (int i = 0; i < test_mesh_1.rows(); i++) {
//		out << "v " << test_mesh_1(i, 0) << ' ' << test_mesh_1(i, 1) << ' ' << test_mesh_1(i, 2) << '\n';
//	}
//	for (int i = test_mesh_1.rows() + 1; i < 34469; i++) {
//		out << suffix[i] << '\n';
//	}
//	out.close();
//}
//#else
//void print_mesh(Eigen::MatrixX3f &test_mesh_1, char suffix[34469][100], std::string name) {
//	FILE *fp;
//	fopen_s(&fp, name.c_str(), "w");
//
//
//	for (int i = 0; i < test_mesh_1.rows(); i++) {
//		fprintf(fp,"v %.6f %.6f %.6f\n", test_mesh_1(i, 0) , test_mesh_1(i, 1) , test_mesh_1(i, 2));
//		//out << "v " << test_mesh_1(i, 0) << ' ' << test_mesh_1(i, 1) << ' ' << test_mesh_1(i, 2) << '\n';
//	}
//	for (int i = test_mesh_1.rows() + 1; i < 34469; i++) {
//		fputs(suffix[i], fp);
//		//out << suffix[i] << '\n';
//	}
//
//	fclose(fp);
//}
//#endif // cppio
//
//
//	
//
//void load_lv(std::string name, DataPoint &temp) {
//	std::cout << "load coefficients...file:" << name << "\n";
//	FILE *fp;
//	fopen_s(&fp, name.c_str(), "rb");
//	
//	temp.user.resize(G_iden_num);
//	for (int j = 0; j < G_iden_num; j++)
//		fread(&temp.user(j), sizeof(float), 1, fp);
//	std::cout << temp.user << "\n";
//	//system("pause");
//	temp.land_2d.resize(G_land_num, 2);
//	for (int i_v = 0; i_v < G_land_num; i_v++) {
//		fread(&temp.land_2d(i_v, 0), sizeof(float), 1, fp);
//		fread(&temp.land_2d(i_v, 1), sizeof(float), 1, fp);
//	}
//
//
//	fread(&temp.center(0), sizeof(float), 1, fp);
//	fread(&temp.center(1), sizeof(float), 1, fp);
//
//	temp.shape.exp.resize(G_nShape);
//	for (int i_shape = 0; i_shape < G_nShape; i_shape++)
//		fread(&temp.shape.exp(i_shape), sizeof(float), 1, fp);
//
//	for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
//		fread(&temp.shape.rot(i, j), sizeof(float), 1, fp);
//
//	for (int i = 0; i < 3; i++) fread(&temp.shape.tslt(i), sizeof(float), 1, fp);
//
//	temp.land_cor.resize(G_land_num);
//	for (int i_v = 0; i_v < G_land_num; i_v++) fread(&temp.land_cor(i_v), sizeof(int), 1, fp);
//
//	temp.s.resize(2, 3);
//	for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++)
//		fread(&temp.s(i, j), sizeof(float), 1, fp);
//
//	temp.shape.dis.resize(G_land_num, 2);
//	for (int i_v = 0; i_v < G_land_num; i_v++) {
//		fread(&temp.shape.dis(i_v, 0), sizeof(float), 1, fp);
//		fread(&temp.shape.dis(i_v, 1), sizeof(float), 1, fp);
//	}
//	std::cout << temp.shape.dis << "\n";
//	//system("pause");
//	fclose(fp);
//	puts("load successful!");
//}
//
//float cal_3d_vtx(
//	Eigen::MatrixXf &bldshps,
//	Eigen::VectorXf &user, Eigen::VectorXf &exp, int vtx_idx, int axis) {
//
//	//puts("calculating one vertex coordinate...");
//	float ans = 0;
//
//	for (int i_id = 0; i_id < G_iden_num; i_id++)
//		for (int i_shape = 0; i_shape < G_nShape; i_shape++)
//			if (i_shape == 0)
//				ans += exp(i_shape)*user(i_id)
//				*bldshps(i_id, vtx_idx * 3 + axis);
//			else
//				ans += exp(i_shape)*user(i_id)
//				*(bldshps(i_id, 3 * G_nVerts*i_shape + vtx_idx * 3 + axis) - bldshps(i_id, vtx_idx * 3 + axis));
//	return ans;
//}
//
//#ifdef cppio
//void solve(DataPoint &data, Eigen::MatrixXf &bldshps, std::string suffix[34469], std::string name) {
//#else
//void solve(DataPoint &data, Eigen::MatrixXf &bldshps, char suffix[34469][100], std::string name) {
//#endif
//
//
//
//	puts("calculating and saving mesh...");
//	std ::cout << "save obj name:" << name << "\n";
//	Eigen::MatrixX3f mesh(G_nVerts, 3);
//	Eigen::VectorXf user = data.user;
//	Eigen::VectorXf exp = data.shape.exp;
//	for (int i_v = 0; i_v < G_nVerts; i_v++) {
//		Eigen::Vector3f v;
//		for (int axis = 0; axis < 3; axis++)
//			v(axis) = cal_3d_vtx(bldshps, user, exp, i_v, axis);
//		mesh.row(i_v) = ((data.shape.rot) * v).transpose();
//	}
//	print_mesh(mesh, suffix, name);
//}
//int cnt = 0;
//#ifdef cppio
//void lv2mesh(std::string path, Eigen::MatrixXf &bldshps, std::string suffix[34469]) {
//#else
//void lv2mesh(std::string path, Eigen::MatrixXf &bldshps, char suffix[34469][100]) {
//#endif // cppio
//
//
//
//	struct dirent **namelist;
//	int n;
//	n = scandir(path.c_str(), &namelist, 0, alphasort);
//	if (n < 0)
//	{
//		std::cout << "scandir return " << n << "\n";
//		perror("Cannot open .");
//		exit(1);
//	}
//	else
//	{
//		int index = 0;
//		struct dirent *dp;
//		while (index < n)
//		{
//			dp = namelist[index];
//			std::cout << dp->d_name << ' ' << strlen(dp->d_name) << "\n";
//			if (dp->d_name[0] == '.') {
//				free(namelist[index]);
//				index++;
//				continue;
//			}
//			if (dp->d_type == DT_DIR) {
//				lv2mesh(path + "/" + dp->d_name,bldshps,suffix);
//			}
//			else {
//				int len = strlen(dp->d_name);
//				if (dp->d_name[len - 1] == 'v' && dp->d_name[len - 2] == 'l') {
//					////	
//					std::string p = path + "/" + dp->d_name;
//					DataPoint data;
//					load_lv(p,data);
//					solve(data,bldshps,suffix, p.substr(0, p.find(".lv")) + "_0norm.obj");
//					std::cout << "cnt" << cnt << "\n";
//					cnt++;
//				}
//			}
//			free(namelist[index]);
//			index++;
//		}
//		free(namelist);
//	}
//
//}
//
//
//Eigen::MatrixXf bldshps(G_iden_num, G_nShape * 3 * G_nVerts);
//#ifdef win64
//std::string bldshps_path = "D:\\sydney\\first\\code\\2017\\deal_data_2\\deal_data/blendshape_ide_svd_77.lv";
//#endif // win64
//#ifdef linux
//std::string bldshps_path = "/home/weiliu/fitting_dde/cal/deal_data/blendshape_ide_svd_77.lv";
//#endif // linux
//
//#ifdef cppio
//std::string suffix[34469];
//#else
//char suffix[34469][100];
//#endif // cppio
//
//void cal_exp_r_t_all_matrix(
//	Eigen::MatrixXf &bldshps, DataPoint &data, Eigen::MatrixXf &result) {
//
//	puts("prepare exp_point matrix for bfgs/ceres...");
//	result.resize(G_nShape, 3 * G_nVerts);
//
//	for (int i_shape = 0; i_shape < G_nShape; i_shape++)
//		for (int i_v = 0; i_v < G_nVerts; i_v++) {
//			Eigen::Vector3f V;
//			V.setZero();
//			for (int j = 0; j < 3; j++)
//				for (int i_id = 0; i_id < G_iden_num; i_id++)
//					if (i_shape == 0)
//						V(j) += data.user(i_id)*bldshps(i_id, i_v * 3 + j);
//					else
//						V(j) += data.user(i_id)*
//						(bldshps(i_id, i_shape*G_nVerts * 3 + i_v * 3 + j) - bldshps(i_id, i_v * 3 + j));
//
//			for (int j = 0; j < 3; j++)
//				result(i_shape, i_v * 3 + j) = V(j);
//		}
//#ifdef deal_64
//	result.block(0, 64 * 3, G_nShape, 3).array() = (result.block(0, 59 * 3, G_nShape, 3).array() + result.block(0, 62 * 3, G_nShape, 3).array()) / 2;
//#endif // deal_64
//}
//
//void solve_same_id(DataPoint &data, Eigen::MatrixXf &exp_r_t_all_matrix, std::string suffix[34469], std::string name) {
//
//
//	puts("calculating and saving mesh...");
//	std::cout << "save obj name:" << name << "\n";
//	Eigen::MatrixX3f mesh(G_nVerts, 3);
//	Eigen::VectorXf user = data.user;
//	Eigen::VectorXf exp = data.shape.exp;
//	for (int i_v = 0; i_v < G_nVerts; i_v++) {
//		Eigen::Vector3f v;
//		v.setZero();
//		for (int axis = 0; axis < 3; axis++)
//			for (int i_shape = 0; i_shape < G_nShape; i_shape++)
//				v(axis) += data.shape.exp(i_shape)*exp_r_t_all_matrix(i_shape, i_v * 3 + axis);
//		mesh.row(i_v) = ((data.shape.rot) * v).transpose();
//	}
//	print_mesh(mesh, suffix, name);
//}
//void lv2mesh_same_id(std::string path, Eigen::MatrixXf &exp_r_t_all_matrix, std::string suffix[34469]) {
//	struct dirent **namelist;
//	int n;
//	n = scandir(path.c_str(), &namelist, 0, alphasort);
//	if (n < 0)
//	{
//		std::cout << "scandir return " << n << "\n";
//		perror("Cannot open .");
//		exit(1);
//	}
//	else
//	{
//		int index = 0;
//		struct dirent *dp;
//		while (index < n)
//		{
//			dp = namelist[index];
//			std::cout << dp->d_name << ' ' << strlen(dp->d_name) << "\n";
//			if (dp->d_name[0] == '.') {
//				free(namelist[index]);
//				index++;
//				continue;
//			}
//			if (dp->d_type == DT_DIR) {
//				lv2mesh(path + "/" + dp->d_name, bldshps, suffix);
//			}
//			else {
//				int len = strlen(dp->d_name);
//				if (dp->d_name[len - 1] == 'v' && dp->d_name[len - 2] == 'l') {
//					////	
//					std::string p = path + "/" + dp->d_name;
//					DataPoint data;
//					load_lv(p, data);
//					solve_same_id(data, exp_r_t_all_matrix, suffix, p.substr(0, p.find(".lv")) + "_0norm.obj");
//					std::cout << "cnt" << cnt << "\n";
//					cnt++;
//				}
//			}
//			free(namelist[index]);
//			index++;
//		}
//		free(namelist);
//	}
//
//}
//
//
//int main() {
//
//	Eigen::MatrixX3f temp;
//	get_mesh(temp, G_nVerts, suffix);
//
//	load_bldshps(bldshps, bldshps_path);
//#ifdef same_id
//	DataPoint data;
//	load_lv("./lv_out/lv_out_1.lv", data);
//	Eigen::MatrixXf exp_r_t_all_matrix;
//	cal_exp_r_t_all_matrix(bldshps, data, exp_r_t_all_matrix);
//	lv2mesh_same_id("./lv_out/", exp_r_t_all_matrix, suffix);
//#else
//
//	lv2mesh("./data_one", bldshps, suffix);
//#endif // same_id
//	return 0;
//}
////g++ -Wall -std=c++11 `pkg-config --cflags opencv` -o deal main.cpp `pkg-config --libs opencv`