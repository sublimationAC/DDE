//#include "calculate_coeff.h"
////#define test_coef_ide_exp
////#define test_bldshps
////#define test_3d2dland
//#define test_2dslt_def
//#define test_2dinner_def
//
//iden ide[G_train_pic_id_num];
//
//Eigen :: MatrixXf bldshps(G_iden_num,G_nShape*3*G_nVerts);
//#ifdef win64
//
//	std::string fwhs_path = "D:/sydney/first/data_me/FaceWarehouse";
//	std::string lfw_path = "D:/sydney/first/data_me/lfw_image";
//	std::string gtav_path = "D:/sydney/first/data_me/GTAV_image";
//	std::string test_path = "D:/sydney/first/data_me/test";
//	std::string test_path_one = "D:/sydney/first/data_me/test_only_one_2d";
//	std::string bldshps_path = "D:\\sydney\\first\\code\\2017\\deal_data_2\\deal_data/blendshape_ide_svd_77.lv";
//	std::string sg_vl_path = "D:\\sydney\\first\\code\\2017\\deal_data_2\\deal_data/blendshape_ide_svd_value_sqrt_77.txt";
//	std::string slt_path = "D:\\openframework\\of_v0.10.0_vs2017_release\\apps\\3d22d\\3d22d/slt_line_4_2.txt";
//	std::string rect_path = "D:\\openframework\\of_v0.10.0_vs2017_release\\apps\\3d22d\\3d22d/slt_rect_4_2.txt";
//	std::string save_coef_path = "./ide_fw_p1.lv";
//	std::string fwhs_path_p = "./data_me/fw_p1";
//
//#endif // win64
//#ifdef linux
//	std::string fwhs_path = "./data_me/FaceWarehouse";
//	std::string fwhs_path_p = "./data_me/fw_p1";
//	std::string lfw_path = "./data_me/lfw_image";
//	std::string gtav_path = "./data_me/GTAV_image";
//	std::string test_path = "./data_me/test";
//	std::string test_path_one = "./data_me/test_only_one";
//	std::string test_path_two = "./data_me/test_only_two";
//	std::string test_path_three = "./data_me/test_only_three";	
//	std::string bldshps_path = "./deal_data/blendshape_ide_svd_50_ite25_0bound.lv";
//	std::string sg_vl_path = "./deal_data/blendshape_ide_svd_value_sqrt_77.txt";
//	std::string slt_path = "./3d22d/slt_line_new.txt";
//	std::string rect_path = "./3d22d/slt_rect_new.txt";
//	std::string save_coef_path = "../fitting_coef/ide_fw_p1.lv";
//#endif // linux
//
//
//
//std::vector< std::vector<cv::Mat_<uchar> > > imgs;
//Eigen::VectorXi inner_land_corr(G_inner_land_num);
////Eigen::VectorXi jaw_land_corr(G_jaw_land_num);
//std::vector<std::pair<int, int> > slt_point_rect[G_nVerts];
//std::vector<int> slt_line[G_line_num];
//Eigen::VectorXf ide_sg_vl(G_iden_num);
//
//
//int main() {
//	int id_idx = 0;
//
//	//load_img_land(fwhs_path,".jpg",ide,id_idx,imgs);
//	printf("id_idx %d\n", id_idx);
//	//load_img_land(lfw_path, ".jpg", ide, id_idx, imgs);
//	printf("id_idx %d\n", id_idx);
//	//load_img_land(gtav_path, ".bmp", ide, id_idx,imgs);
//	load_img_land(test_path_one, ".jpg", ide, id_idx, imgs);
//
//	printf("id_idx %d\n", id_idx);      
//	//test_data_2dland(imgs, ide, 0, 2);
//	
//#ifdef test_3d2dland
//	test_3d22dland(imgs[0][0], "./server/2dland.txt",ide,0,0);
//#endif // test_3d2dland
//
//#ifdef test_2dinner_def
//	test_inner_2dland(imgs[0][0], "./server/2dland.txt", ide, 0, 0);
//#endif
//
//#ifdef test_2dslt_def
//	//test_slt_2dland(imgs[0][0], "./server/test_updt_slt_2d_point.txt", ide, 0, 0);
//	test_slt_me_2dland(imgs[0][0], "./server/test_updt_slt_me_2d_point.txt", ide, 0, 0);
//#endif
//	
//
//	//14 13
//	//9 17
//	
//	load_inner_land_corr(inner_land_corr);
//	//load_jaw_land_corr(jaw_land_corr);
//	//std::cout << inner_land_corr << '\n';
//	load_slt(slt_line,slt_point_rect,slt_path,rect_path);
//	load_bldshps(bldshps,bldshps_path,ide_sg_vl, sg_vl_path);
//#ifdef test_bldshps
//	print_bldshps(bldshps);
//#else
//
//
//#ifdef test_coef_ide_exp
//	//cal_mesh_land(bldshps);
//	cal_mesh_land_exp_only(bldshps);
//#else
//
//	init_exp_ide_r_t_pq(ide, id_idx);
//#ifdef posit
//	cal_f(ide, bldshps, inner_land_corr, slt_line, slt_point_rect,ide_sg_vl);
//#endif // posit
//#ifdef normalization
//	solve(ide, bldshps, inner_land_corr, slt_line, slt_point_rect, ide_sg_vl);
//#endif // normalization
//	cal_dis(ide, bldshps, id_idx);
//	
//	const int left_eye_num = 8;
//	int idx_lft_eye[left_eye_num] = { 27,28,29,30,66,67,68,65 };
//	const int right_eye_num = 8;
//	int idx_rt_eye[right_eye_num] = { 31,32,33,34,70,71,72,69 };
//
//	const int lft_bn_num = 6;
//	int idx_lft_bn[lft_bn_num] = { 21,22,23,24,25,26};
//	const int rt_bn_num = 6;
//	int idx_rt_bn[rt_bn_num] = { 15,16,17,18,19,20 };
//
//	const int ms_be = 46, ms_ed = 64;
//	const int ns_be = 35, ns_ed = 46;
//
//	for (int id = 0; id < id_idx; id++) {
//		for (int i_exp = 0; i_exp < ide[id].num; i_exp++) {
//			double err_slt = 0, err_eye_left = 0, err_eye_right = 0, err_lft_bn = 0, err_rt_bn = 0, err_ms = 0, err_ns = 0;
//			for (int i = 0; i < 15; i++) err_slt += ide[id].dis.row(G_land_num*i_exp+i).squaredNorm();
//
//			for (int i=0;i<left_eye_num;i++) err_eye_left+= ide[id].dis.row(G_land_num*i_exp + idx_lft_eye[i]).squaredNorm();
//			for (int i = 0; i < right_eye_num; i++) err_eye_right += ide[id].dis.row(G_land_num*i_exp + idx_rt_eye[i]).squaredNorm();
//
//			for (int i = 0; i < lft_bn_num; i++) err_lft_bn += ide[id].dis.row(G_land_num*i_exp + idx_lft_bn[i]).squaredNorm();
//			for (int i = 0; i < rt_bn_num; i++) err_rt_bn += ide[id].dis.row(G_land_num*i_exp + idx_rt_bn[i]).squaredNorm();
//
//			for (int i = ms_be; i < ms_ed; i++) err_ms += ide[id].dis.row(G_land_num*i_exp + i).squaredNorm();
//			for (int i = ns_be; i < ns_ed; i++) err_ns += ide[id].dis.row(G_land_num*i_exp + i).squaredNorm();
//
//			printf("slt err:%.5f\nlft eye:%.5f  rt eye:%.5f\n", err_slt/15, err_eye_left/ left_eye_num, err_eye_right/right_eye_num);
//			printf("lft bn:%.5f  rt bn:%.5f\n", err_lft_bn/ lft_bn_num, err_rt_bn/ rt_bn_num);
//			printf("ms:%.5f  ns:%.5f\n", err_ms/(ms_ed-ms_be+1), err_ns/(ns_ed-ns_be+1));
//
//
//			for (int i = 0; i < G_land_num; i++)
//				printf("%d %.5f %.5f\n", i, ide[id].dis(G_land_num*i_exp + i, 0), ide[id].dis(G_land_num*i_exp + i, 1));
//		}
//	}
//
//	int id = 0;
//	save_fitting_coef_each(test_path_one, ide, id);
//#endif
//#endif
//	system("pause");
//	return 0;
//}
//
//
///*
//1 grep -rl 'fopen_s(&fp,' ./ | xargs sed -i 's/fopen_s(&fp,/fp=fopen(/g'
//2
//3
//4 grep -rl 'fscanf_s' ./ | xargs sed -i 's/fscanf_s/fscanf/g'
//
//
//grep -rl 'fopen_s(&fpr,' ./ | xargs sed -i 's/fopen_s(&fpr,/fpr=fopen(/g'
//grep -rl 'fopen_s(&fpw,' ./ | xargs sed -i 's/fopen_s(&fpw,/fpw=fopen(/g'
//*/