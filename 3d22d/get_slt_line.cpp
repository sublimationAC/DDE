#include "get_slt_line.h"

void get_scratch_line(std::string path, int &num, std::vector <int> *scrtch_line) {
	puts("getting scratch_line");
	FILE *fp;
	fopen_s(&fp, path.c_str(), "r");
	fscanf_s(fp, "%d", &num);
//	assert(num == G_line_num);

	for (int idx = 0; idx < num; idx++) {
		int p, n;
		fscanf_s(fp, "%d%d", &p, &n);
		printf("%d %d\n", p, n);
		scrtch_line[idx].push_back(p);
		for (int i = 0; i < n; i++) {
			int x;
			fscanf_s(fp, "%d", &x);
			scrtch_line[idx].push_back(x);
		}
	}
	fclose(fp);
	//fopen_s(&fp, "slt_line_new_short_ans.txt", "w");
	//fprintf(fp, "%d\n", num);
	//for (int idx = 0; idx < num; idx++) {
	//	int be, en;
	//	if (idx < 70) be = 1, en = std::min((int)(scrtch_line[idx].size()), 18);
	//	else en= std::min((int)(scrtch_line[idx].size()), 15),be=en-6;
	//	fprintf(fp, "%d %d", idx,en-be);	
	//	for (int i = be; i < en; i++) {
	//		fprintf(fp, " %d", scrtch_line[idx][i]);		
	//	}
	//	fprintf(fp, "\n");
	//}
	//fclose(fp);
	//fopen_s(&fp, "slt_line_new_short_ans.txt", "w");
	//fprintf(fp, "%d\n", num);
	//int id_idx = 0;
	//for (int idx = 0; idx < 34; idx++) {
	//	fprintf(fp, "%d %d", id_idx++, scrtch_line[idx].size() - 1);
	//	for (int i = 1; i < scrtch_line[idx].size(); i++) {
	//		fprintf(fp, " %d", scrtch_line[idx][i]);
	//	}
	//	fprintf(fp, "\n");
	//}
	//for (int idx = 83; idx>=69; idx--) {
	//	fprintf(fp, "%d %d", id_idx++, scrtch_line[idx].size() - 1);
	//	for (int i = 1; i < scrtch_line[idx].size(); i++) {
	//		fprintf(fp, " %d", scrtch_line[idx][i]);
	//	}
	//	fprintf(fp, "\n");
	//}
	//for (int idx = 68; idx >= 34; idx--) {
	//	fprintf(fp, "%d %d", id_idx++, scrtch_line[idx].size()-1);
	//	for (int i = 1; i < scrtch_line[idx].size(); i++) {
	//		fprintf(fp, " %d", scrtch_line[idx][i]);
	//	}
	//	fprintf(fp, "\n");
	//}
	//fclose(fp);
	//fopen_s(&fp, "slt_line_4_2.txt", "w");
	//fprintf(fp, "%d\n", num);
	//for (int idx = 0; idx < num; idx++) {
	//	int be, en;
	//	be = 1, en = scrtch_line[idx].size();

	//	fprintf(fp, "%d %d", idx,en-be);	
	//	for (int i = be; i < en; i++) {
	//		fprintf(fp, " %d", scrtch_line[idx][i]);		
	//	}
	//	fprintf(fp, "\n");
	//}
	//fclose(fp);


}

void deal_scratch_line(int num, std::vector <int> *scrtch_line) {
	puts("deal scratch_line");
	FILE *fp;

	//fopen_s(&fp, "slt_line_new_short_ans.txt", "w");
	//fprintf(fp, "%d\n", num);
	//for (int idx = 0; idx < num; idx++) {
	//	int be, en;
	//	if (idx < 70) be = 1, en = std::min((int)(scrtch_line[idx].size()), 18);
	//	else en= std::min((int)(scrtch_line[idx].size()), 15),be=en-6;
	//	fprintf(fp, "%d %d", idx,en-be);	
	//	for (int i = be; i < en; i++) {
	//		fprintf(fp, " %d", scrtch_line[idx][i]);		
	//	}
	//	fprintf(fp, "\n");
	//}
	//fclose(fp);
	//fopen_s(&fp, "slt_line_new_short_ans.txt", "w");
	//fprintf(fp, "%d\n", num);
	//int id_idx = 0;
	//for (int idx = 0; idx < 34; idx++) {
	//	fprintf(fp, "%d %d", id_idx++, scrtch_line[idx].size() - 1);
	//	for (int i = 1; i < scrtch_line[idx].size(); i++) {
	//		fprintf(fp, " %d", scrtch_line[idx][i]);
	//	}
	//	fprintf(fp, "\n");
	//}
	//for (int idx = 83; idx>=69; idx--) {
	//	fprintf(fp, "%d %d", id_idx++, scrtch_line[idx].size() - 1);
	//	for (int i = 1; i < scrtch_line[idx].size(); i++) {
	//		fprintf(fp, " %d", scrtch_line[idx][i]);
	//	}
	//	fprintf(fp, "\n");
	//}
	//for (int idx = 68; idx >= 34; idx--) {
	//	fprintf(fp, "%d %d", id_idx++, scrtch_line[idx].size()-1);
	//	for (int i = 1; i < scrtch_line[idx].size(); i++) {
	//		fprintf(fp, " %d", scrtch_line[idx][i]);
	//	}
	//	fprintf(fp, "\n");
	//}
	//fclose(fp);
	//fopen_s(&fp, "slt_line_4_2.txt", "w");
	//fprintf(fp, "%d\n", num);
	//for (int idx = 0; idx < num; idx++) {
	//	int be, en;
	//	be = 1, en = scrtch_line[idx].size();

	//	fprintf(fp, "%d %d", idx,en-be);	
	//	for (int i = be; i < en; i++) {
	//		fprintf(fp, " %d", scrtch_line[idx][i]);		
	//	}
	//	fprintf(fp, "\n");
	//}
	//fclose(fp);
	Eigen::VectorXi rdc(G_line_num);
	rdc.setZero();
	rdc.block(8, 0, 6, 1) <<
		1, 1, 1, 3, 0, 3;
	rdc.block(14, 0, 10, 1) <<
		6, 7, 7, 7, 5, 5, 5, 5, 5, 5;
	rdc.block(24, 0, 10, 1) <<
		6, 9, 9, 9, 9, 9, 8, 5, 4, 3;

	fopen_s(&fp, "slt_line_8_10.txt", "w");
	fprintf(fp, "%d\n", num);

	for (int idx = 0; idx < num; idx++) {
		int be, en;
		be = 1, en = scrtch_line[idx].size()-rdc(idx);

		fprintf(fp, "%d %d", idx,en-be);	
		for (int i = be; i < en; i++) {
			fprintf(fp, " %d", scrtch_line[idx][i]);		
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

}

void get_tst_slt_pts(Eigen::MatrixXi &slt_pts,int &slt_line_total) {
	puts("getting get_tst_slt_pts");
	FILE *fp;
	//fopen_s(&fp, "D:\\sydney\\first\\code\\2017\\DDE\\FaceX-Train/test_updt_slt.txt", "r");
	fopen_s(&fp, "D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/server/test_updt_slt_dde.txt", "r");
	int num;
	fscanf_s(fp, "%d", &num);
	slt_line_total = num;
	slt_pts.resize(num, G_line_num);
	for (int j = 0; j < num; j++) {
		float x, y, z;
		fscanf_s(fp, "%f%f%f", &x, &y, &z);
		for (int i = 0; i < G_line_num; i++)
			fscanf_s(fp, "%d", &slt_pts(j, i));
	}
	fclose(fp);
}

