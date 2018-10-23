#include "calculate_coeff.h"
//#define test_coef_ide_exp
//#define test_bldshps
//#define test_3d2dland
#define test_2dslt_def
iden ide[G_train_pic_id_num];

Eigen :: MatrixXf bldshps(G_iden_num,G_nShape*3*G_nVerts);
#ifdef win64

	std::string fwhs_path = "D:/sydney/first/data_me/FaceWarehouse";
	std::string lfw_path = "D:/sydney/first/data_me/lfw_image";
	std::string gtav_path = "D:/sydney/first/data_me/GTAV_image";
	std::string test_path = "D:/sydney/first/data_me/test";
	std::string test_path_one = "D:/sydney/first/data_me/test_only_one";
	std::string bldshps_path = "D:\\sydney\\first\\code\\2017\\deal_data_2\\deal_data/blendshape_ide_svd_77.lv";
	std::string sg_vl_path = "D:\\sydney\\first\\code\\2017\\deal_data_2\\deal_data/blendshape_ide_svd_value_sqrt_77.txt";
	std::string slt_path = "D:\\openframework\\of_v0.10.0_vs2017_release\\apps\\3d22d\\3d22d/sillht.txt";
	std::string rect_path = "D:\\openframework\\of_v0.10.0_vs2017_release\\apps\\3d22d\\3d22d/slt_point_rect.txt";
	std::string save_coef_path = "./ide_fw_p1.lv";

#endif // win64
#ifdef linux
	std::string fwhs_path = "./data_me/FaceWarehouse";
	std::string lfw_path = "./data_me/lfw_image";
	std::string gtav_path = "./data_me/GTAV_image";
	std::string test_path = "./data_me/test";
	std::string test_path_one = "./data_me/test_only_one";
	std::string test_path_two = "./data_me/test_only_two";
	std::string test_path_three = "./data_me/test_only_three";	
	std::string bldshps_path = "./deal_data/blendshape_ide_svd_77.lv";
	std::string sg_vl_path = "./deal_data/blendshape_ide_svd_value_sqrt_77.txt";
	std::string slt_path = "./3d22d/sillht.txt";
	std::string rect_path = "./3d22d/slt_point_rect.txt";
	std::string save_coef_path = "../fitting_coef/ide_fw_p1.lv";
#endif // linux



std::vector< std::vector<cv::Mat_<uchar> > > imgs;
Eigen::VectorXi inner_land_corr(G_inner_land_num);
Eigen::VectorXi jaw_land_corr(G_jaw_land_num);
std::vector<std::pair<int, int> > slt_point_rect[G_nVerts];
std::vector<int> slt_line[G_line_num];
Eigen::VectorXf ide_sg_vl(G_iden_num);


int main() {
	int id_idx = 0;

	//load_img_land(fwhs_path,".jpg",ide,id_idx,imgs);
	printf("id_idx %d\n", id_idx);
	//load_img_land(lfw_path, ".jpg", ide, id_idx, imgs);
	printf("id_idx %d\n", id_idx);
	//load_img_land(gtav_path, ".bmp", ide, id_idx,imgs);
	load_img_land(test_path_one, ".jpg", ide, id_idx, imgs);
	printf("id_idx %d\n", id_idx);      
	//test_data_2dland(imgs, ide, 0, 0);
	
#ifdef test_3d2dland
	test_3d22dland(imgs[0][0], "./server/2dland.txt",ide,0,0);
#endif // test_3d2dland

#ifdef test_2dslt_def
	test_2dslt(imgs, ide, 0, 1);
#endif
	

	//14 13
	//9 17
	
	load_inner_land_corr(inner_land_corr);
	load_jaw_land_corr(jaw_land_corr);
	//std::cout << inner_land_corr << '\n';
	load_slt(slt_line,slt_point_rect,slt_path,rect_path);
	load_bldshps(bldshps,bldshps_path,ide_sg_vl, sg_vl_path);
#ifdef test_bldshps
	print_bldshps(bldshps);
#endif


#ifdef test_coef_ide_exp
	//cal_mesh_land(bldshps);
	cal_mesh_land_exp_only(bldshps);
#else

	init_exp_ide_r_t_pq(ide, id_idx);
#ifdef posit
	cal_f(ide, bldshps, inner_land_corr, jaw_land_corr, slt_line, slt_point_rect,ide_sg_vl);
#endif // posit
#ifdef normalization
	solve(ide, bldshps, inner_land_corr, jaw_land_corr, slt_line, slt_point_rect, ide_sg_vl);
#endif // normalization
	cal_dis(ide, bldshps, id_idx);
	save_result(ide,id_idx,save_coef_path);
#endif
	system("pause");
	return 0;
}


/*
1 grep -rl 'fopen_s(&fp,' ./ | xargs sed -i 's/fopen_s(&fp,/fp=fopen(/g'
2
3
4 grep -rl 'fscanf_s' ./ | xargs sed -i 's/fscanf_s/fscanf/g'


grep -rl 'fopen_s(&fpr,' ./ | xargs sed -i 's/fopen_s(&fpr,/fpr=fopen(/g'
grep -rl 'fopen_s(&fpw,' ./ | xargs sed -i 's/fopen_s(&fpw,/fpw=fopen(/g'
*/