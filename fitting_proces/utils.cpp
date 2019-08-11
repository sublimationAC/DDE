#include "utils.h"



void load_land_cor_from_lv(std::string name, Eigen::VectorXi &land_cor) {
	std::cout << "load coefficients...file:" << name << "\n";
	FILE *fp;
	fopen_s(&fp, name.c_str(), "rb");

	DataPoint temp;
	temp.user.resize(G_iden_num);
	for (int j = 0; j < G_iden_num; j++)
		fread(&temp.user(j), sizeof(float), 1, fp);
	std::cout << temp.user << "\n";
	//system("pause");
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

	for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
		fread(&temp.shape.rot(i, j), sizeof(float), 1, fp);

	for (int i = 0; i < 3; i++) fread(&temp.shape.tslt(i), sizeof(float), 1, fp);

	land_cor.resize(G_land_num);
	for (int i_v = 0; i_v < G_land_num; i_v++) fread(&land_cor(i_v), sizeof(int), 1, fp);
#ifdef normalization


	temp.s.resize(2, 3);
	for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++)
		fread(&temp.s(i, j), sizeof(float), 1, fp);
#endif // normalization

	temp.shape.dis.resize(G_land_num, 2);
	for (int i_v = 0; i_v < G_land_num; i_v++) {
		fread(&temp.shape.dis(i_v, 0), sizeof(float), 1, fp);
		fread(&temp.shape.dis(i_v, 1), sizeof(float), 1, fp);
	}
	//std::cout << temp.shape.dis << "\n";
	//system("pause");
	fclose(fp);
	puts("load successful!");
}

void load_land_cor_from_psp_f(std::string name, Eigen::VectorXi &land_cor) {
	std::cout << "load coefficients...file:" << name << "\n";
	FILE *fp;
	fopen_s(&fp, name.c_str(), "rb");

	DataPoint temp;
	temp.user.resize(G_iden_num);
	for (int j = 0; j < G_iden_num; j++)
		fread(&temp.user(j), sizeof(float), 1, fp);
	std::cout << temp.user << "\n";
	//system("pause");
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

	for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
		fread(&temp.shape.rot(i, j), sizeof(float), 1, fp);

	for (int i = 0; i < 3; i++) fread(&temp.shape.tslt(i), sizeof(float), 1, fp);

	land_cor.resize(G_land_num);
	for (int i_v = 0; i_v < G_land_num; i_v++) fread(&land_cor(i_v), sizeof(int), 1, fp);
	std::cout << "land cor:" << land_cor << "\n";

	fread(&temp.f, sizeof(float), 1, fp);

	temp.shape.dis.resize(G_land_num, 2);
	for (int i_v = 0; i_v < G_land_num; i_v++) {
		fread(&temp.shape.dis(i_v, 0), sizeof(float), 1, fp);
		fread(&temp.shape.dis(i_v, 1), sizeof(float), 1, fp);
	}
	//std::cout << temp.shape.dis << "\n";
	//system("pause");
	fclose(fp);
	puts("load successful!");
}
void get_tst_slt_pts(Eigen::MatrixXi &slt_pts) {
	puts("getting get_tst_slt_pts");
	FILE *fp;
	//fopen_s(&fp, "D:\\sydney\\first\\code\\2017\\DDE\\FaceX-Train/test_updt_slt.txt", "r");
	fopen_s(&fp, "D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/server/test_updt_slt.txt", "r");
	int num;
	fscanf_s(fp, "%d", &num);
	slt_pts.resize(num, G_line_num);
	for (int j = 0; j < num; j++) {
		float x, y, z;
		fscanf_s(fp, "%f%f%f", &x, &y, &z);
		for (int i = 0; i < G_line_num; i++)
			fscanf_s(fp, "%d", &slt_pts(j, i));
	}
}