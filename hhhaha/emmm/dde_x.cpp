#include "dde_x.h"

#include <algorithm>
#include <stdexcept>
//#define update_slt_def

using namespace std;

DDEX::DDEX(const string & filename)
{
	cv::FileStorage model_file;
	model_file.open(filename, cv::FileStorage::READ);
	if (!model_file.isOpened())
		throw runtime_error("Cannot open model file \"" + filename + "\".");

	model_file["ref_shape"] >> ref_shape_;
	cv::FileNode fn;/*= model_file["test_init_shapes"];
	for (auto it = fn.begin(); it != fn.end(); ++it)
	{
		vector<cv::Point2d> shape;
		*it >> shape;
		test_init_shapes_.push_back(shape);
	}*/
	fn = model_file["stage_regressors"];
	for (auto it = fn.begin(); it != fn.end(); ++it)
	{
		regressor_dde r;
		*it >> r;
		stage_regressors_dde_.push_back(r);
	}
}


void change_nearest(DataPoint &data, Eigen::MatrixXf &bldshps, std::vector<DataPoint> &train_data) {
	puts("change nearest");
	float mi = 100000, mi_land = 100000000;
	int idx = 0;
	for (int i = 0; i < train_data.size(); i++) {
		float distance = (data.center - train_data[i].center).norm();
		float distance_land = (data.land_2d - train_data[i].land_2d).norm();
		if (distance < mi || ((fabs(distance - mi) < EPSILON)&&(distance_land < mi_land))) {
			idx = i;
			mi = distance;
			mi_land = distance_land;
		}
	}
	printf("nearest vertex %d\n", idx);
	data.shape.rot = train_data[idx].shape.rot;
}
void get_init_shape(std::vector<Target_type> &ans, DataPoint &data, std::vector<DataPoint>&train_data) {

	puts("calculating initial shape");
	float mi[G_dde_K], mi_land[G_dde_K]; int idx[G_dde_K];
	for (int i = 0; i < G_dde_K; i++) mi[i] = 1000000, mi_land[i] = 100000000, idx[i] = 0;
	for (int i = 0; i < train_data.size(); i++) {
		float distance = (data.center - train_data[i].center).norm();
		float distance_land = (data.land_2d - train_data[i].land_2d).norm();
		for (int j=0;j<G_dde_K;j++)
			if (distance < mi[j] ||
				((fabs(distance - mi[j]) < EPSILON) && (distance_land < mi_land[j]))) {
				for (int k = j + 1; k < G_dde_K; k++) {
					idx[k] = idx[k - 1];
					mi[k] = mi[k - 1];
					mi_land[k] = mi_land[k - 1];
				}
				idx[j] = i;
				mi[j] = distance;
				mi_land[j] = distance_land;
				break;
			}
	}
	for (int i = 0; i < G_dde_K; i++)
		ans[i] = train_data[idx[i]].shape;
}

void update_slt(
	Eigen::MatrixXf &bldshps,std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
	Eigen::VectorXi &jaw_land_corr, DataPoint &data) {
	////////////////////////////////project

	puts("updating silhouette...");
	Eigen::Matrix3f R = data.shape.rot;
	Eigen::Vector3f T = data.shape.tslt.transpose();

	//puts("A");
	Eigen::VectorXi slt_cddt(G_line_num + G_jaw_land_num);
	Eigen::MatrixX3f slt_cddt_cdnt(G_line_num + G_jaw_land_num, 3);
	//puts("B");
	//FILE *fp;
	//fopen_s(&fp, "test_slt.txt", "w");
	for (int i = 0; i < G_line_num; i++) {
		//printf("i %d\n", i);
		float min_v_n = 10000;
		int min_idx = 0;
		Eigen::Vector3f cdnt;

#ifdef normalization
		for (int j = 0, sz = slt_line[i].size(); j < sz; j++) {
			//printf("j %d\n", j);
			int x = slt_line[i][j];
			//printf("x %d\n", x);
			Eigen::Vector3f point, temp;
			for (int axis = 0; axis < 3; axis++)
				point(axis) = cal_3d_vtx(bldshps,data.user,data.shape.exp, x, axis);
			point = R * point;
			temp = point;
			point.normalize();
			if (fabs(point(2)) < min_v_n) min_v_n = fabs(point(2)), min_idx = x, cdnt = temp;// printf("%.6f %.6f %.6f \n", point[0](0), point[0](1), point[0](2));
		}
		slt_cddt(i) = min_idx;
		cdnt.block(0, 0, 2, 1) = data.s*cdnt + T.block(0, 0, 2, 1);
		/*cdnt(0) = cdnt(0)*ide[id_idx].s(exp_idx, 0) + T(0);
		cdnt(1) = cdnt(1)*ide[id_idx].s(exp_idx, 1) + T(1);*/
		slt_cddt_cdnt.row(i) = cdnt.transpose();
#endif

	}
	//fclose(fp);
	//puts("C");
	for (int i_jaw = 0; i_jaw < G_jaw_land_num; i_jaw++) {
		Eigen::Vector3f point;
		for (int axis = 0; axis < 3; axis++)
			point(axis) = cal_3d_vtx(bldshps, data.user,data.shape.exp,jaw_land_corr(i_jaw), axis);
#ifdef normalization
		point.block(0, 0, 2, 1) = data.s*R * point;
		//point(0) *= ide[id_idx].s(exp_idx, 0), point(1) *= ide[id_idx].s(exp_idx, 1);
		point = point + T;
#endif // normalization		
		slt_cddt(i_jaw + G_line_num) = jaw_land_corr(i_jaw);
		slt_cddt_cdnt.row(i_jaw + G_line_num) = point.transpose();
	}
	for (int i = 0; i < 15; i++) {
		float min_dis = 10000;
		int min_idx = 0;;
		for (int j = 0; j < G_line_num + G_jaw_land_num; j++) {
#ifdef posit
			float temp =
				fabs(slt_cddt_cdnt(j, 0) - ide[id_idx].land_2d(G_land_num*exp_idx + i, 0)) +
				fabs(slt_cddt_cdnt(j, 1) - ide[id_idx].land_2d(G_land_num*exp_idx + i, 1));
#endif // posit
#ifdef normalization
			float temp =
				fabs(slt_cddt_cdnt(j, 0) - data.land_2d(i, 0)) +
				fabs(slt_cddt_cdnt(j, 1) - data.land_2d(i, 1));
#endif // normalization


			if (temp < min_dis) min_dis = temp, min_idx = j;
		}
		//printf("%d %d %d\n", i, min_idx, slt_cddt(min_idx));
		data.land_cor(i) = slt_cddt(min_idx);

	}
	//std :: cout << "slt_cddt_cdnt\n" << slt_cddt_cdnt.block(0,0, slt_cddt_cdnt.rows(),2).rowwise()+ide[id_idx].center.row(exp_idx) << "\n";
	//std::cout << "out land\n" << ide[id_idx].land_2d.block(G_land_num*exp_idx , 0,15,2).rowwise() + ide[id_idx].center.row(exp_idx) << "\n";
	//std::cout << "out land correlation\n" << out_land_cor.transpose() << "\n";
	//system("pause");
}



void DDEX::dde(
	DataPoint &data, Eigen::MatrixXf &bldshps,
	Eigen::MatrixX3i &tri_idx, std::vector<DataPoint> &train_data, Eigen::VectorXi &jaw_land_corr,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect)const {

	std::cout << tri_idx.transpose() << "\n";


	Target_type result;
	result.dis.resize(G_land_num, 2);
	result.dis.setZero();
	result.exp.resize(G_nShape);
	result.exp.setZero();
	result.tslt.setZero();
	result.rot.setZero();

	

	change_nearest(data,bldshps,train_data);

#ifdef update_slt_def
	update_slt(bldshps, slt_line, slt_point_rect, jaw_land_corr, data);
#endif // update_slt_def
	show_image_0rect(data.image, data.landmarks);
	std::cout << data.shape.dis << "\n";
	print_datapoint(data);
	update_2d_land(data, bldshps);
	std::cout << data.landmarks << "\n";
	show_image_0rect(data.image, data.landmarks);
	//find init
	std::vector<Target_type> init_shape(G_dde_K);
	get_init_shape(init_shape, data, train_data);


	for (int i = 0; i < init_shape.size(); ++i)
	{
		printf("%d init shape\n", i);
		//Transform t = Procrustes(initial_landmarks, test_init_shapes_[i]);
		//t.Apply(&init_shape);

		Target_type result_shape = init_shape[i];
		
		for (int j = 0; j < stage_regressors_dde_.size(); ++j)
		{
			printf("outer regressor %d:\n", j);
			//Transform t = Procrustes(init_shape, mean_shape_);
			Target_type offset =
				stage_regressors_dde_[j].Apply(result_shape,tri_idx,data,bldshps);
			//t.Apply(&offset, false);
			result_shape = shape_adjustment(result_shape, offset);
		}

		std::vector<cv::Point2d> land_temp;
		cal_2d_land_i(land_temp, result_shape, bldshps, data);
		show_image_0rect(data.image, land_temp);
		result.dis.array() += result_shape.dis.array();
		result.exp.array() += result_shape.exp.array();
		result.rot.array() += result_shape.rot.array();
		result.tslt.array() += result_shape.tslt.array();

	}
	result.dis.array() /= G_dde_K;
	result.exp.array() /= G_dde_K;
	result.rot.array() /= G_dde_K;
	result.tslt.array() /= G_dde_K;
	data.shape = result;

#ifdef update_slt_def
	update_slt(bldshps, slt_line, slt_point_rect, jaw_land_corr, data);
#endif // update_slt_def
	update_2d_land(data, bldshps);
}