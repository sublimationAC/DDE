#include "calculate_coeff.h"

iden ide[G_train_pic_id_num];

Eigen :: MatrixXf bldshps(G_iden_num,G_nShape*3*G_nVerts);
std :: string fwhs_path = "D:/sydney/first/data_me/FaceWarehouse";
std::string lfw_path = "D:/sydney/first/data_me/lfw_image";
std::string gtav_path = "D:/sydney/first/data_me/GTAV_image";
std::string bldshps_path = "D:\\sydney\\first\\code\\2017\\deal_data_2\\deal_data/blendshape_ide_svd.lv";
std::vector< std::vector<cv::Mat_<uchar> > > imgs;
Eigen::VectorXi inner_land_corr(G_inner_land_num);
std::vector<std::pair<int, int> > slt_point_rect[G_nVerts];
std::vector<int> slt_line[G_line_num];


int main() {
	int id_idx = 0;

	load_img_land(fwhs_path,".jpg",ide,id_idx,imgs);
	printf("%d\n", id_idx);
	//load_img_land(lfw_path, ".jpg", ide, id_idx, imgs);
	printf("%d\n", id_idx);
	//load_img_land(gtav_path, ".bmp", ide, id_idx,imgs);
	printf("%d\n", id_idx);
	test_data_2dland(imgs, ide, 3, 9);

	//14 13
	//9 17
	//init_r_t_pq(ide, id_idx);
	
	load_inner_land_corr(inner_land_corr);
	load_slt(slt_line,slt_point_rect);
	load_bldshps(bldshps,bldshps_path);
	//cal_f
	system("pause");
	return 0;
}