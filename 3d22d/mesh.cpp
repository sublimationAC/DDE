#include "mesh.h"
//#define test_slt_drct
#define slt_file

void init_mesh(std::string name, Mesh_my &mesh) {
	printf("Initiating mesh...\n");
	igl::readOBJ(name, mesh.vtx, mesh.rect);
	mesh.num_rect = mesh.rect.rows();
	mesh.num_vtx = mesh.vtx.rows();
	printf("%d %d %d %d\n", mesh.num_rect, mesh.num_vtx, mesh.vtx.cols(), mesh.rect.cols());
	for (int i = 0; i < 10; i++)
		printf("%.2f %.2f %.2f\n", mesh.vtx(i, 0), mesh.vtx(i, 1), mesh.vtx(i, 2));

	for (int i = 0; i < 10; i++)
		printf("%d %d %d %d\n", mesh.rect(i, 0), mesh.rect(i, 1), mesh.rect(i, 2), mesh.rect(i, 3));
	//mesh.vtx *= ofGetWidth() ;
	cal_norm(mesh);
}
void cal_norm(Mesh_my &mesh) {
	puts("calculating norm ...");
	mesh.norm_vtx.resize(mesh.num_vtx, 3);
	mesh.norm_vtx.setZero();
	//printf("%d %d\n", mesh.norm_vtx.rows(), mesh.norm_vtx.cols());
	//int cnt = 0;
	for (int i = 0; i < mesh.num_rect; i++) {
		for (int l = 0; l < 4; l++) {

			Eigen::Vector3d point[3];
			for (int k = l - 1; k < l + 2; k++)
				for (int j = 0; j < 3; j++) {
					int kk = k;
					if (k == -1) kk = 3;
					if (k == 4) kk = 0;
					point[k - l + 1](j) = mesh.vtx(mesh.rect(i, kk), j);
				}
			Eigen::Vector3d v[3];
			v[0] = point[0] - point[1];
			v[1] = point[2] - point[1];
			v[2] = v[0].cross(v[1]);
			//v[3] = point[3] - point[0];
			//if ((v[2].dot(v[3])) > EPSILON) printf("%d %.10f\n",i, v[2].dot(v[3])),cnt++;
			//v[2] = v[2].array() / (double)(v[2].norm());
			v[2].normalize();
			if (i < 2) printf("-%d %d %.10f %.10f %.10f\n",i,l,v[2](0), v[2](1), v[2](2));
			//if () 
			mesh.norm_vtx.row(mesh.rect(i, l)) += v[2].transpose();
			/*if (i < 10) printf("++%d %.10f %.10f %.10f\n", i, mesh.norm_vtx(mesh.rect(i, 1),0)
			, mesh.norm_vtx(mesh.rect(i, 1),1), mesh.norm_vtx(mesh.rect(i, 1), 2));*/
		}
	}
	for (int i = 0; i < mesh.num_vtx; i++) {
		if (mesh.norm_vtx.row(i).norm() > EPSILON)
			mesh.norm_vtx.row(i).normalize();
		//mesh.norm_vtx.row(i) = mesh.norm_vtx.row(i).array() / (double)mesh.norm_vtx.row(i).norm();
	}
	//printf("%d %d\n", cnt,mesh.num_rect);
}

#include "GL/glut.h"
void draw_mesh(Mesh_my &mesh) {
	
	glLineWidth(0.3);
	for (int i = 0; i < mesh.num_rect; ++i) {
		//printf("%d\n",i);
		//bool fl = 0;
		//for (int p = 0; p < 4; p++) {
		//	int idx = mesh.rect(i, p);
		//	if (mesh.vtx(idx, 2) < -0.1) fl = 1;
		//}
		//if (fl) continue;
		//if (check(mesh,i)) continue;

		//glBegin(GL_QUADS);

		//for (int j = 0; j < 4; ++j) {
		//	//printf("%d %d\n", mesh.norm_vtx.rows(), mesh.norm_vtx.cols());
		//	//printf("%d %d %d\n", i, j, mesh.tri(i, j));
		//	int VertIndex = mesh.rect(i, j);

		//	/*if (i < 5) {
		//		printf("+%.10f %.10f %.10f\n", mesh.norm_vtx(VertIndex, 0), mesh.norm_vtx(VertIndex, 1), mesh.norm_vtx(VertIndex, 2));
		//		printf("-%.10f %.10f %.10f\n", mesh.vtx(VertIndex, 0), mesh.vtx(VertIndex, 1), mesh.vtx(VertIndex, 2));
		//	}*/
		//	GLdouble normal[3] = { mesh.norm_vtx(VertIndex, 0), mesh.norm_vtx(VertIndex, 1), mesh.norm_vtx(VertIndex, 2) };
		//	glNormal3dv(normal);

		//	glVertex3f(mesh.vtx(VertIndex, 0), mesh.vtx(VertIndex, 1), mesh.vtx(VertIndex, 2));
		//}
		//glEnd();
		//bool fl = 0;
		//for (int t = 0; t < 4; t++) fl |= check_mouse_vtx(mesh,mesh.rect(i,t));
		//if (fl) continue;

		for (int t = 0; t < 1; t++) {
			glBegin(GL_TRIANGLES);
			for (int p = 0; p < 3; p++) {
				int VertIndex = mesh.rect(i, (t + p) % 4);
				GLdouble normal[3] = { mesh.norm_vtx(VertIndex, 0), mesh.norm_vtx(VertIndex, 1), mesh.norm_vtx(VertIndex, 2) };
				glNormal3dv(normal);
				glVertex3f(mesh.vtx(VertIndex, 0), mesh.vtx(VertIndex, 1), mesh.vtx(VertIndex, 2));
			}
			glEnd();
		}
		for (int t = 2; t < 3; t++) {
			glBegin(GL_TRIANGLES);
			for (int p = 0; p < 3; p++) {
				int VertIndex = mesh.rect(i, (t + p) % 4);
				GLdouble normal[3] = { mesh.norm_vtx(VertIndex, 0), mesh.norm_vtx(VertIndex, 1), mesh.norm_vtx(VertIndex, 2) };
				glNormal3dv(normal);
				glVertex3f(mesh.vtx(VertIndex, 0), mesh.vtx(VertIndex, 1), mesh.vtx(VertIndex, 2));
			}
			glEnd();
		}
	}
}

void draw_mesh_point(Mesh_my &mesh) {
	glPointSize(3);
	for (int i = 0; i < mesh.num_rect; ++i) {
		//printf("%d\n",i);
		bool fl = 0;
		for (int p = 0; p < 4; p++) {
			int idx = mesh.rect(i, p);
			if (mesh.vtx(idx, 2) < -0.1) fl = 1;
		}
		if (fl) continue;
		for (int t = 0; t < 4; t++) {
			int VertIndex = mesh.rect(i, t);
			glBegin(GL_POINTS);
			glVertex3f(mesh.vtx(VertIndex, 0), mesh.vtx(VertIndex, 1), mesh.vtx(VertIndex, 2));
			glEnd();
		}
	}
	//FILE *fp;
	//fopen_s(&fp, "start_pt_idx_jaw.txt", "r");
	//const int n = 29;
	//Eigen::VectorXi cor(n);
	//for (int i = 0; i < n; i++)
	//	fscanf_s(fp, "%d", &cor(i));
	//fclose(fp);
	////puts("loading inner point");
	//glPointSize(15);
	//for (int i = 0; i < n; i++) {
	//	glBegin(GL_POINTS);
	//	glVertex3f(mesh.vtx(cor(i), 0), mesh.vtx(cor(i), 1), mesh.vtx(cor(i), 2));
	//	//printf("%d ", cor(i));
	//	glEnd();
	//}
	//glBegin(GL_POINTS);
	//glVertex3f(mesh.vtx(10828, 0), mesh.vtx(10828, 1), mesh.vtx(10828, 2));
	////printf("%d ", cor(i));
	//glEnd();
}

void draw_line(Mesh_my &mesh,double agl){
	double scale = 1.0001;
	glLineWidth(1);
	for (int i = 0; i < mesh.num_rect; ++i) {
		//printf("%d\n",i);
		for (int t = 0; t < 4; t++) {
			int idx1=mesh.rect(i,t),idx2=mesh.rect(i,(t + 1) % 4);
			//if (mesh.vtx(idx1, 1) - mesh.vtx(idx2, 1) <= 0.01 && mesh.vtx(idx1, 1) - mesh.vtx(idx2, 1) >= -0.01) {
				/*if (mesh.norm_vtx(idx1, 0)*sin(agl) + mesh.norm_vtx(idx1, 2)*cos(agl) > 0.1) continue;
				if (mesh.norm_vtx(idx1, 2)*sin(agl) + mesh.norm_vtx(idx1, 2)*cos(agl) > 0.1) continue;
				if (fabs(mesh.norm_vtx(idx1,0)*sin(agl)+mesh.norm_vtx(idx1, 2)*cos(agl)) > 0.6) continue;
				if (fabs(mesh.norm_vtx(idx2, 2)*sin(agl) +mesh.norm_vtx(idx2, 2)*cos(agl)) > 0.6) continue;*/
				//puts("asd");
				glBegin(GL_LINES);
				glVertex3f(mesh.vtx(idx1, 0)*scale, mesh.vtx(idx1, 1)*scale, mesh.vtx(idx1, 2)*scale);
				glVertex3f(mesh.vtx(idx2, 0)*scale, mesh.vtx(idx2, 1)*scale, mesh.vtx(idx2, 2)*scale);
				glEnd();
			//}
		}
	}

	// test normal vector
	/*glLineWidth(0.1);
	for (int i = 0; i < mesh.num_vtx; i++) {
		if (mesh.vtx(i, 2) < 0.1) continue;
		glBegin(GL_LINES);
		glVertex3f(mesh.vtx(i, 0), mesh.vtx(i, 1), mesh.vtx(i, 2));
		glVertex3f(mesh.vtx(i, 0) + mesh.norm_vtx(i, 0), mesh.vtx(i, 1) + mesh.norm_vtx(i, 1), mesh.vtx(i, 2) + mesh.norm_vtx(i, 2));
		glEnd();
	}*/

	//test the landmark correspongdennce

	glPointSize(3);
	for (int i = 0; i < mesh.num_vtx; i++) {
		glBegin(GL_POINTS);
		glVertex3f(mesh.vtx(i, 0), mesh.vtx(i, 1), mesh.vtx(i, 2));
		glEnd();
	}

}

void check_2d_3d_inner_corr(Mesh_my &mesh) {
	puts("loading inner point");
	Eigen::VectorXi cor;
	FILE *fp;
	fopen_s(&fp, "D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/inner_jaw/inner_vertex_corr_58_416fw.txt", "r");
	cor.resize(G_jaw_land_num + G_inner_land_num);
	for (int i = 0; i < G_inner_land_num; i++)
		fscanf_s(fp, "%d", &cor(i));
	fclose(fp);
	//puts("loading jaw point");
	//fopen_s(&fp, "D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/inner_jaw/jaw_vertex.txt", "r");
	//for (int i = G_inner_land_num; i < G_inner_land_num + G_jaw_land_num; i++)
	//	fscanf_s(fp, "%d", &cor(i));
	//fclose(fp);
	puts("loading inner point");
	glPointSize(15);
	for (int i = 0; i < G_inner_land_num; i++) {
		glBegin(GL_POINTS);
		glVertex3f(mesh.vtx(cor(i), 0), mesh.vtx(cor(i), 1), mesh.vtx(cor(i), 2));
		//printf("%d ", cor(i));
		glEnd();
	}
}

// do not forget to delete 22(right side around the eye)
bool check_slt_line(Mesh_my &mesh, int i) {
	bool fl = 0;
	for (int j = 0; j < 4; j++) {
		if (mesh.vtx(mesh.rect(i, j), 2) < -0.1) fl = 1;
		if (mesh.vtx(mesh.rect(i, j), 1) > 0.42 || mesh.vtx(mesh.rect(i, j), 1) < -0.4) fl = 1;
		if (mesh.vtx(mesh.rect(i, j), 0) < 0) {
			if (mesh.vtx(mesh.rect(i, j), 1) > -0.1) {
				if (fabs(mesh.vtx(mesh.rect(i, j), 0)) < 0.44) fl = 1;
			}
			else if (fabs(mesh.vtx(mesh.rect(i, j), 0)) < 0.3) fl = 1; 
		}
		else {
			if (mesh.vtx(mesh.rect(i, j), 1) > -0.1) {
				if (mesh.vtx(mesh.rect(i, j), 0) < 0.54) fl = 1;
			}
			else if (mesh.vtx(mesh.rect(i, j), 0) < 0.4) fl = 1;
		}
	}
	return fl;
}

std::vector<int> E[G_line_num];
void check_2d_3d_out_corr(Mesh_my &mesh) {
	double scale = 1.1;
	puts("check out corr  silhoutte...");
#ifdef slt_file
	FILE *fp;
	fopen_s(&fp, "D:\\openframework\\of_v0.10.0_vs2017_release\\apps\\3d22d\\3d22d/sillht.txt", "r");
	for (int i = 0; i < G_line_num; i++) {
		int num;
		fscanf_s(fp, "%d", &num);		
		E[i].clear();
		for (int j = 0; j < num; j++) {
			int x;
			fscanf_s(fp, "%d", &x);
			E[i].push_back(x);
		}
	}
	fclose(fp);
	glPointSize(5);
	for (int i = 0; i < G_line_num; i++) {

		//if (E[i].size() > 15) continue;
		for (int j = 0; j < E[i].size(); j++) {
			glBegin(GL_POINTS);
			glVertex3f(mesh.vtx(E[i][j], 0), mesh.vtx(E[i][j], 1), mesh.vtx(E[i][j], 2));
			glEnd();
		}
	}
	
	puts("asd");
	glLineWidth(5);
	for (int i = 0; i < G_line_num; i++)
		for (int j = 0; j < E[i].size()-1; j++) {
			glBegin(GL_LINES);
			glVertex3f(mesh.vtx(E[i][j], 0)*scale, mesh.vtx(E[i][j], 1)*scale, mesh.vtx(E[i][j], 2)*scale);
			glVertex3f(mesh.vtx(E[i][j+1], 0)*scale, mesh.vtx(E[i][j + 1], 1)*scale, mesh.vtx(E[i][j + 1], 2)*scale);
			
			glEnd();
		} 
#endif
#ifdef test_slt_drct

	glLineWidth(5);
	for (int i = 0; i < mesh.num_rect; ++i) {
		//printf("%d\n",i);		
		if (check_slt_line(mesh,i)) continue;
		float mi = 100;
		int mi1, mi2;
		for (int t = 0; t < 4; t++) {
			int idx1= mesh.rect(i, t), idx2= mesh.rect(i, (t+1)%4);
			if (fabs(mesh.vtx(idx1, 1) - mesh.vtx(idx2, 1)) < mi)
				mi = fabs(mesh.vtx(idx1, 1) - mesh.vtx(idx2, 1)), mi1 = idx1, mi2 = idx2;
			/*printf("%.5f %.5f\n", fabs(mesh.vtx(mesh.rect(i, 0), 1) - mesh.vtx(mesh.rect(i, 1), 1)),
				fabs(mesh.vtx(mesh.rect(i, 0), 1) - mesh.vtx(mesh.rect(i, 2), 1)));*/
		}
		glBegin(GL_LINES);
		glVertex3f(mesh.vtx(mi1, 0)*scale, mesh.vtx(mi1, 1)*scale, mesh.vtx(mi1, 2)*scale);
		glVertex3f(mesh.vtx(mi2, 0)*scale, mesh.vtx(mi2, 1)*scale, mesh.vtx(mi2, 2)*scale);
		glEnd();
		mi = 10;
		int mm = mi1;
		for (int t = 0; t < 4; t++) {
			int idx1 = mesh.rect(i, t), idx2 = mesh.rect(i, (t + 1) % 4);
			if (idx1 == mm) continue;
			if (fabs(mesh.vtx(idx1, 1) - mesh.vtx(idx2, 1)) < mi)
				mi = fabs(mesh.vtx(idx1, 1) - mesh.vtx(idx2, 1)), mi1 = idx1, mi2 = idx2;
			/*printf("%.5f %.5f\n", fabs(mesh.vtx(mesh.rect(i, 0), 1) - mesh.vtx(mesh.rect(i, 1), 1)),
			fabs(mesh.vtx(mesh.rect(i, 0), 1) - mesh.vtx(mesh.rect(i, 2), 1)));*/
		}
		glBegin(GL_LINES);
		glVertex3f(mesh.vtx(mi1, 0)*scale, mesh.vtx(mi1, 1)*scale, mesh.vtx(mi1, 2)*scale);
		glVertex3f(mesh.vtx(mi2, 0)*scale, mesh.vtx(mi2, 1)*scale, mesh.vtx(mi2, 2)*scale);
		glEnd();
	}
#endif // test_slt_drct
}

int f[20000] = { 0 };
bool use[20000] = { 0 };
#include<utility>
std::vector<std::pair<int,int> > slt_point_rect[20000];
int u_f(int x) {
	if (f[x] == 0) return x;
	f[x] = u_f(f[x]);
	return f[x];
}

void get_silhouette_vertex(Mesh_my &mesh) {
	for (int i = 0; i < mesh.num_rect; ++i) {
		//printf("%d\n",i);
		if (check_slt_line(mesh, i)) continue;
		float mi = 10;
		int mi1, mi2;
		for (int t = 0; t < 4; t++) {
			int idx1 = mesh.rect(i, t), idx2 = mesh.rect(i, (t + 1) % 4);
			if (fabs(mesh.vtx(idx1, 1) - mesh.vtx(idx2, 1)) < mi)
				mi = fabs(mesh.vtx(idx1, 1) - mesh.vtx(idx2, 1)), mi1 = idx1, mi2 = idx2;
			/*printf("%.5f %.5f\n", fabs(mesh.vtx(mesh.rect(i, 0), 1) - mesh.vtx(mesh.rect(i, 1), 1)),
			fabs(mesh.vtx(mesh.rect(i, 0), 1) - mesh.vtx(mesh.rect(i, 2), 1)));*/
		}
		int r1 = u_f(mi1), r2 = u_f(mi2);
		if (r1 != r2) f[r1] = r2;
		mi = 10;
		int mm = mi1;
		for (int t = 0; t < 4; t++) {
			int idx1 = mesh.rect(i, t), idx2 = mesh.rect(i, (t + 1) % 4);
			if (idx1 == mm) continue;
			if (fabs(mesh.vtx(idx1, 1) - mesh.vtx(idx2, 1)) < mi)
				mi = fabs(mesh.vtx(idx1, 1) - mesh.vtx(idx2, 1)), mi1 = idx1, mi2 = idx2;
			/*printf("%.5f %.5f\n", fabs(mesh.vtx(mesh.rect(i, 0), 1) - mesh.vtx(mesh.rect(i, 1), 1)),
			fabs(mesh.vtx(mesh.rect(i, 0), 1) - mesh.vtx(mesh.rect(i, 2), 1)));*/
		}
		r1 = u_f(mi1), r2 = u_f(mi2);
		if (r1 != r2) f[r1] = r2;
	}

	FILE *fp;
	fopen_s(&fp, "sillht.txt", "w");
	for (int i = 0; i < 20000; i++) {
		slt_point_rect[i].clear();
		int r = u_f(i);
		if (r != i && use[r] == 0) {
			std::vector<int> p;
			p.clear();
			for (int j = 0; j < 20000; j++)
				if (u_f(j) == r) p.push_back(j);
			use[r] = 1;
			//if (p.size() > 15) continue;
			fprintf(fp, "%d", p.size());
			for (int j = 0; j < p.size(); j++)
				fprintf(fp, " %d", p[j]) , use[p[j]] = 1;
			fprintf(fp, "\n");			
		}
	}
	fclose(fp);

	fopen_s(&fp, "slt_point_rect.txt", "w");
	mesh.vtx.row(1);
	for (int i=0;i<mesh.num_rect;i++)
		for (int j = 0; j < 4; j++) {
			int idx = mesh.rect(i, j);
			if (use[idx]) 
				if (slt_point_rect[idx].size()==0) 
					slt_point_rect[idx].push_back(
						std::make_pair(mesh.rect(i, (j+3)%4), mesh.rect(i, (j +1)%4 )));
				else {
					Eigen::RowVector3d V[2];
					cal_nor_vec(V[0], mesh.vtx.row(slt_point_rect[idx][0].first), mesh.vtx.row(slt_point_rect[idx][0].second),mesh.vtx.row(idx));
					cal_nor_vec(V[1], mesh.vtx.row(mesh.rect(i, (j + 3) % 4)), mesh.vtx.row(mesh.rect(i, (j + 1) % 4)), mesh.vtx.row(idx));
					if (V[0].dot(V[1]) > 0)
						slt_point_rect[idx].push_back(
							std::make_pair(mesh.rect(i, (j + 3) % 4), mesh.rect(i, (j + 1) % 4)));
					else
						slt_point_rect[idx].push_back(
							std::make_pair(mesh.rect(i, (j + 1) % 4), mesh.rect(i, (j + 3) % 4)));
				}
		}

	for (int i = 0; i < 20000; i++) {
		if (use[i]) {
			fprintf(fp, "%d %d ", i,slt_point_rect[i].size());
			for (int j = 0; j < slt_point_rect[i].size(); j++)
				fprintf(fp, " %d %d", slt_point_rect[i][j].first, slt_point_rect[i][j].second);
			fprintf(fp, "\n");
		}
	}
	fclose(fp);
} 

void cal_nor_vec(Eigen::RowVector3d &nor, Eigen::RowVector3d a, Eigen::RowVector3d b, Eigen::RowVector3d o) {
	nor = (a - o).cross(b - o);
}

void test_slt() {
	FILE *fp;
	fopen_s(&fp, "D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/test_slt_picked_out.txt", "r");
	float x, y, z;
	glPointSize(5);
	while (fscanf_s(fp,"%f",&x)>0)
	{
		fscanf_s(fp, "%f%f", &y, &z);
		glBegin(GL_POINTS);
		glVertex3f(x, y, z);
		glEnd();
	}
	fclose(fp);
}

void get_coef_land(Eigen::MatrixX3f &coef_land, int &test_coef_num_tot,std::string name) {
	puts("get_coef_land");
	FILE *fp;
	fopen_s(&fp, name.c_str(), "r");
	int num;
	fscanf_s(fp, "%d", &num);
	test_coef_num_tot = num;
	coef_land.resize(num * G_land_num, 3);
	for (int j = 0; j < num * G_land_num; j++)
		fscanf_s(fp, "%f%f%f", &coef_land(j, 0), &coef_land(j, 1), &coef_land(j, 2));
	fclose(fp);
}

float scale = 0.6;
void test_coef_land(Eigen::MatrixX3f &coef_land,int idx) {

	glPointSize(5);
	for (int i = idx* G_land_num; i < G_land_num*(idx+1); i++) {
		glBegin(GL_POINTS);
		glVertex3f(coef_land(i,0)*scale, coef_land(i, 1)*scale, coef_land(i, 2)*scale);
		glEnd();
	}
}

void get_coef_mesh(Eigen::MatrixX3f &coef_mesh,std::string name) {
	puts("get_coef_mesh");
	FILE *fp;
	int num = 0;
	num=fopen_s(&fp, name.c_str(), "r");
	std::cout << name << "\n";
	
	printf("%d\n", num);
	fscanf_s(fp, "%d", &num);
	printf("%d\n", num);
	coef_mesh.resize(num * G_nVerts, 3);
	for (int j = 0; j < num * G_nVerts; j++)
		fscanf_s(fp, "%f%f%f", &coef_mesh(j, 0), &coef_mesh(j, 1), &coef_mesh(j, 2));
	//fclose(fp);
	//fopen_s(&fp, "test_temp.txt", "w");
	//for (int j=0;j<G_nVerts;j++)
	//	fprintf(fp,"v %.6f %.6f %.6f\n", coef_mesh(j, 0), coef_mesh(j, 1), coef_mesh(j, 2));
	//fclose(fp);
}

void test_coef_mesh(Mesh_my &mesh, Eigen::MatrixX3f &coef_mesh, int idx) {
	for (int i = 0; i < mesh.num_vtx; i++)
		for (int j = 0; j < 3; j++)
			mesh.vtx(i, j) = coef_mesh(idx*G_nVerts + i, j) *scale;

	cal_norm(mesh);
}
int cnt_vtx[16000];
void smooth_mesh(Mesh_my &mesh, int iteration) {
	while (iteration--)
	{
		EigenMatrixXs temp = mesh.vtx;
		mesh.vtx.setZero();
		memset(cnt_vtx, 0, sizeof(cnt_vtx));
		for (int i_r = 0; i_r < mesh.num_rect; i_r++) {
			for (int l = 0; l < 4; l++) {
				cnt_vtx[mesh.rect(i_r, l)]++;
				int k = l - 1;
				if (k == -1) k = 3;
				mesh.vtx.row(mesh.rect(i_r, k)).array() += temp.row(mesh.rect(i_r, l)).array();
				k = l + 1;
				if (k == 4) k = 0;
				mesh.vtx.row(mesh.rect(i_r, k)).array() += temp.row(mesh.rect(i_r, l)).array();
			}
		}
		for (int i_v = 0; i_v < mesh.num_vtx;i_v++) mesh.vtx.row(i_v).array() /= cnt_vtx[i_v]*2;
	}
	cal_norm(mesh);
}
	
bool check_mouse_vtx(Mesh_my &mesh, int i) {
	bool fl = 0;
	if (mesh.vtx(i, 2) < -0.1) fl = 1;
	if (mesh.vtx(i, 1) > -0.05 || mesh.vtx(i, 1) < -0.25) fl = 1;
	if (mesh.vtx(i, 0) >0.3 || mesh.vtx(i, 0) < -0.2) fl = 1;
	
	return fl;
}
std:: vector<int> mouse_edge[11510];

void get_mouse_data(Mesh_my &mesh) {
	memset(use, 0, sizeof(use));
	for (int i_v = 0; i_v < mesh.num_vtx; i_v++) {
		use[i_v] = !check_mouse_vtx(mesh, i_v);
		mouse_edge[i_v].clear();
	}
	for (int i_e = 0; i_e < mesh.num_rect; i_e++)
		for (int j = 0; j < 4; j++) {
			int idx = mesh.rect(i_e, j);
			if (use[idx]) {
				mouse_edge[idx].push_back(mesh.rect(i_e, (j + 3) % 4));
				mouse_edge[idx].push_back(mesh.rect(i_e, (j + 1) % 4));
			}
		}
	FILE *fp;
	fopen_s(&fp, "mouse_point.txt", "w");
	for (int i_v = 0; i_v < mesh.num_vtx; i_v++) 
	if (mouse_edge[i_v].size()>0) {
		fprintf(fp, "%d %d",i_v, mouse_edge[i_v].size());
		for (int j = 0; j < mouse_edge[i_v].size() ;j++)
			fprintf(fp, " %d", mouse_edge[i_v][j]);
		fprintf(fp, "\n");
	}	
	fclose(fp);
}
void test_mouse(Mesh_my &mesh) {
	FILE *fp;
	fopen_s(&fp, "D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/mouse_point.txt", "r");
	glPointSize(5);
	int n ,x;
	fscanf_s(fp, "%d", &n);
	for (int i = 0; i < n; i++) {
		int t, x, y;
		fscanf_s(fp, "%d%d", &t, &x);
		for (int j = 0; j < x; j++) fscanf_s(fp, "%d", &y);
		glPointSize(3);
		glBegin(GL_POINTS);
		glVertex3f(mesh.vtx(t, 0), mesh.vtx(t, 1), mesh.vtx(t, 2));
		glEnd();
	}
	fclose(fp);
}

void check_3d_fan_corr(Mesh_my &mesh, Eigen::VectorXi &cor) {
	puts("loading 3d_fan point");
	FILE *fp;
	fopen_s(&fp, "D:\\sydney\\first\\code\\2017\\fan3d\\fan3d/inner_jaw/fan3d_land.txt", "r");
	cor.resize(68);
	for (int i = 0; i < 68; i++)
		fscanf_s(fp, "%d", &cor(i));
	fclose(fp);
	glPointSize(15);
	for (int i = 0; i < 68; i++) {
		glBegin(GL_POINTS);
		glVertex3f(mesh.vtx(cor(i), 0), mesh.vtx(cor(i), 1), mesh.vtx(cor(i), 2));
		//printf("%d ", cor(i));
		glEnd();
	}
}

void get_positive_point(Mesh_my &mesh) {
	FILE *fp;
	fopen_s(&fp, "over01_idx.txt", "w");
	for (int i=0;i<mesh.num_vtx;i++)
	if (mesh.vtx(i,2)>-0.1)
		fprintf(fp, "%d\n", i);
	fclose(fp);
}


void test_update_slt_norm(
	Mesh_my &mesh,Eigen::MatrixX3f &norm_line,Eigen::VectorXi &slt_cddt_idx,
	std::string slt_path, std::string rect_path) {
	////////////////////////////////project

	puts("calculating silhouette...");
	std::vector<int> slt_line[G_line_num];
	std::vector<std::pair<int, int> > slt_point_rect[G_nVerts];
	FILE *fp;
	fopen_s(&fp, slt_path.c_str(), "r");
	int num;
	fscanf_s(fp, "%d", &num);
	assert(num == G_line_num);
	for (int i = 0; i < G_line_num; i++) {
		int num;
		fscanf_s(fp, "%d", &num);
		slt_line[i].resize(num);
		for (int j = 0; j < num; j++)
			fscanf_s(fp, "%d", &slt_line[i][j]);
	}
	fclose(fp);
	fopen_s(&fp, rect_path.c_str(), "r");
	int vtx_num;
	fscanf_s(fp, "%d", &vtx_num);
	for (int i = 0; i < vtx_num; i++) {
		int idx, num;
		fscanf_s(fp, "%d%d", &idx, &num);
		printf("%d %d %d\n", i, idx, num);
		slt_point_rect[idx].resize(num);
		for (int j = 0; j < num; j++) fscanf_s(fp, "%d%d", &slt_point_rect[idx][j].first, &slt_point_rect[idx][j].second);
	}
	fclose(fp);


	int cnt_norm_line = 0;
	slt_cddt_idx.setZero();
	for (int i = 0; i < 10; i++) {
		printf("tst updt slt i %d\n", i);
		float min_v_n = 10000;
		int min_idx = 0;
		Eigen::Vector3f cdnt;
		for (int j = 0, sz = slt_line[i].size(); j < sz; j++) {
			//printf("j %d\n", j);
			int x = slt_line[i][j];
			//printf("x %d\n", x);
			Eigen::Vector3f nor;
			nor.setZero();
			Eigen::Vector3f V[2], point[3];
			for (int axis = 0; axis < 3; axis++)
				point[0](axis) = mesh.vtx(x, axis);
			//test															//////////////////////////////////debug
			//puts("A");
			//fprintf(fp, "%.6f %.6f %.6f \n", point[0](0), point[0](1), point[0](2));
			//printf("%.6f %.6f %.6f \n", point[0](0), point[0](1), point[0](2));
			//puts("B");


			////////////////////////////////////////////////////////////////////////////////////////////////////////

			for (int k = 0, sz = slt_point_rect[x].size(); k < sz; k++) {
				//printf("k %d\n", k);
				for (int axis = 0; axis < 3; axis++) {
					point[1](axis) = mesh.vtx(slt_point_rect[x][k].first, axis);
					point[2](axis) = mesh.vtx(slt_point_rect[x][k].second, axis);
				}
				
				V[0] = point[1] - point[0];
				V[1] = point[2] - point[0];
				//puts("C");
				V[0] = V[0].cross(V[1]);
				//puts("D");
				V[0].normalize();
				nor = nor + V[0];
				//printf("__ %.6f %.6f %.6f \n", point[0](0), point[0](1), point[0](2));
			}
			//printf("== %.6f %.6f %.6f \n", point[0](0), point[0](1), point[0](2));
			//puts("F");
			if (nor.norm() > EPSILON) nor.normalize();

			cnt_norm_line++;
			norm_line.conservativeResize(cnt_norm_line * 2, 3);
			norm_line.row(cnt_norm_line * 2 - 2) = point[0].transpose();
			norm_line.row(cnt_norm_line * 2 - 1) = ((point[0]-nor/10)).transpose();

			//std::cout << "nor++\n\n" << nor << "\n";
			//std::cout << "point--\n\n" << point[0].normalized() << "\n";
			//std::cout << "rltv--\n\n"<<x << ' ' << nor.dot(point[0].normalized()) << "\n";
			if (fabs(nor(2)) < min_v_n) min_v_n = fabs(nor(2)), min_idx = x, cdnt = point[0];// printf("%.6f %.6f %.6f \n", point[0](0), point[0](1), point[0](2));


/*
			cnt_norm_line++;
			norm_line.conservativeResize(cnt_norm_line * 2, 3);
			norm_line.row(cnt_norm_line * 2 - 2) = point[0].transpose();
			point[0].normalize();
			norm_line.row(cnt_norm_line * 2 - 1) = ((norm_line.row(cnt_norm_line * 2 - 2) + point[0].transpose() / 10)).transpose();
			if (fabs(point[0](2)) < min_v_n) min_v_n = fabs(point[0](2)), min_idx = x, cdnt = point[0];*/
		}
		//puts("H");
		//fprintf(fp, "%.6f %.6f %.6f \n", cdnt(0), cdnt(1), cdnt(2));
		slt_cddt_idx(i) = min_idx;

	}

}

void draw_test_slt_norm(Eigen::MatrixX3f test_slt_norm, Eigen::VectorXi slt_cddt_idx,Mesh_my &mesh) {
	std::cout << test_slt_norm.rows() << " " << test_slt_norm.cols() << "\n";
	glLineWidth(1);
	for (int i = 0; i < test_slt_norm.rows(); i += 2) {
		//printf("%d\n",i);
		glBegin(GL_LINES);
		glVertex3f(test_slt_norm(i, 0), test_slt_norm(i, 1), test_slt_norm(i, 2));
		glVertex3f(test_slt_norm(i + 1, 0), test_slt_norm(i + 1, 1), test_slt_norm(i + 1, 2));
		glEnd();
	}
	glPointSize(5);
	for (int i = 1; i < test_slt_norm.rows(); i += 2) {
		glBegin(GL_POINTS);
		glVertex3f(test_slt_norm(i, 0), test_slt_norm(i, 1), test_slt_norm(i, 2));
		glEnd();
	}
	glPointSize(10);
	for (int i = 0; i < slt_cddt_idx.rows(); i++) {
		glBegin(GL_POINTS);
		glVertex3f(mesh.vtx(slt_cddt_idx(i),0), mesh.vtx(slt_cddt_idx(i), 1), mesh.vtx(slt_cddt_idx(i), 2));
		glEnd();
	}
	
}

void show_scratch_line(int num, std::vector <int> *scrtch_line, Mesh_my &mesh) {
	float scale = 1.01;
	//printf("%d\n", scrtch_line[60][2]);
	//for (int idx = 0; idx < G_line_num; idx++) {
	//	glLineWidth(3);
	//	//printf("%d %d----\n", idx, scrtch_line[idx].size());
	//	for (int i = 2; i < scrtch_line[idx].size(); i++) {
	//		//printf("%d %d %d\n",i, scrtch_line[idx][i], scrtch_line[idx][i-1]);
	//		glBegin(GL_LINES);
	//		glVertex3f(mesh.vtx(scrtch_line[idx][i], 0)*scale, mesh.vtx(scrtch_line[idx][i], 1)*scale, mesh.vtx(scrtch_line[idx][i], 2)*scale);
	//		glVertex3f(mesh.vtx(scrtch_line[idx][i - 1], 0)*scale, mesh.vtx(scrtch_line[idx][i - 1], 1)*scale, mesh.vtx(scrtch_line[idx][i - 1], 2)*scale);
	//		glEnd();
	//	}
	//	glPointSize(10);
	//	for (int i = 1; i < scrtch_line[idx].size(); i++) {
	//		//printf("%d %d %d\n",i, scrtch_line[idx][i], scrtch_line[idx][i-1]);
	//		glBegin(GL_POINTS);
	//		glVertex3f(mesh.vtx(scrtch_line[idx][i], 0)*scale, mesh.vtx(scrtch_line[idx][i], 1)*scale, mesh.vtx(scrtch_line[idx][i], 2)*scale);
	//		glEnd();
	//	}
	//}
	//for (int idx = 48; idx < 84; idx++) {
	//	glLineWidth(3);
	//	//printf("%d %d----\n", idx, scrtch_line[idx].size());
	//	for (int i = 2; i < scrtch_line[idx].size(); i++) {
	//		//printf("%d %d %d\n",i, scrtch_line[idx][i], scrtch_line[idx][i-1]);
	//		glBegin(GL_LINES);
	//		glVertex3f(mesh.vtx(scrtch_line[idx][i], 0)*scale, mesh.vtx(scrtch_line[idx][i], 1)*scale, mesh.vtx(scrtch_line[idx][i], 2)*scale);
	//		glVertex3f(mesh.vtx(scrtch_line[idx][i - 1], 0)*scale, mesh.vtx(scrtch_line[idx][i - 1], 1)*scale, mesh.vtx(scrtch_line[idx][i - 1], 2)*scale);
	//		glEnd();
	//	}
	//	glPointSize(10);
	//	for (int i = 1; i < scrtch_line[idx].size()/2; i++) {
	//		//printf("%d %d %d\n",i, scrtch_line[idx][i], scrtch_line[idx][i-1]);
	//		glBegin(GL_POINTS);
	//		glVertex3f(mesh.vtx(scrtch_line[idx][i], 0)*scale, mesh.vtx(scrtch_line[idx][i], 1)*scale, mesh.vtx(scrtch_line[idx][i], 2)*scale);
	//		glEnd();
	//	}
	//}
	//for (int idx = 70; idx < 84; idx++) {
	//	glLineWidth(3);
	//	//printf("%d %d----\n", idx, scrtch_line[idx].size());
	//	for (int i = 2; i < scrtch_line[idx].size(); i++) {
	//		//printf("%d %d %d\n",i, scrtch_line[idx][i], scrtch_line[idx][i-1]);
	//		glBegin(GL_LINES);
	//		glVertex3f(mesh.vtx(scrtch_line[idx][i], 0)*scale, mesh.vtx(scrtch_line[idx][i], 1)*scale, mesh.vtx(scrtch_line[idx][i], 2)*scale);
	//		glVertex3f(mesh.vtx(scrtch_line[idx][i - 1], 0)*scale, mesh.vtx(scrtch_line[idx][i - 1], 1)*scale, mesh.vtx(scrtch_line[idx][i - 1], 2)*scale);
	//		glEnd();
	//	}
	//	glPointSize(10);
	//	for (int i = std::min((int)(scrtch_line[idx].size()), 15)-6; i < std::min((int)(scrtch_line[idx].size()), 15); i++) {
	//		//printf("%d %d %d\n",i, scrtch_line[idx][i], scrtch_line[idx][i-1]);
	//		glBegin(GL_POINTS);
	//		glVertex3f(mesh.vtx(scrtch_line[idx][i], 0)*scale, mesh.vtx(scrtch_line[idx][i], 1)*scale, mesh.vtx(scrtch_line[idx][i], 2)*scale);
	//		glEnd();
	//	}
	//}

	Eigen::VectorXi rdc(num);
	rdc.setZero();
	//rdc.block(8, 0, 6, 1) <<
	//	1, 1, 1, 3, 0, 3;
	//rdc.block(14, 0, 10, 1) <<
	//	6, 7, 7, 7, 5, 5, 5, 5, 5, 5;
	//rdc.block(24, 0, 10, 1) <<
	//	6, 9,9, 9, 9, 9, 8, 5, 4,3;
	printf("line num:%d\n", num);
	for (int idx = 0; idx < num; idx++) {
		glLineWidth(3);
		//printf("%d %d----\n", idx, scrtch_line[idx].size());
		for (int i = 2; i < scrtch_line[idx].size(); i++) {
			//printf("%d %d %d\n",i, scrtch_line[idx][i], scrtch_line[idx][i-1]);
			glBegin(GL_LINES);
			glVertex3f(mesh.vtx(scrtch_line[idx][i], 0)*scale, mesh.vtx(scrtch_line[idx][i], 1)*scale, mesh.vtx(scrtch_line[idx][i], 2)*scale);
			glVertex3f(mesh.vtx(scrtch_line[idx][i - 1], 0)*scale, mesh.vtx(scrtch_line[idx][i - 1], 1)*scale, mesh.vtx(scrtch_line[idx][i - 1], 2)*scale);
			glEnd();
		}
		glPointSize(10);
		for (int i = 1; i < scrtch_line[idx].size()- rdc(idx); i++) {
			//printf("%d %d %d\n",i, scrtch_line[idx][i], scrtch_line[idx][i-1]);
			glBegin(GL_POINTS);
			glVertex3f(mesh.vtx(scrtch_line[idx][i], 0)*scale, mesh.vtx(scrtch_line[idx][i], 1)*scale, mesh.vtx(scrtch_line[idx][i], 2)*scale);
			glEnd();
		}
	}
}

void get_silhouette_rect(Mesh_my &mesh,std::string path) {
	puts("getting scratch_line");
	FILE *fp;
	fopen_s(&fp, path.c_str(), "r");
	int num;
	fscanf_s(fp, "%d", &num);
	memset(use, 0, sizeof(use));
	for (int idx = 0; idx < num; idx++) {
		int p, n;
		fscanf_s(fp, "%d%d", &p, &n);
		printf("%d %d\n", p, n);
		for (int i = 0; i < n; i++) {
			int x;
			fscanf_s(fp, "%d", &x);
			use[x] = 1;
		}
	}

	for (int i = 0; i < 20000; i++) {
		slt_point_rect[i].clear();

	}

	fopen_s(&fp, "slt_rect_4_10.txt", "w");
	
	for (int i = 0; i < mesh.num_rect; i++)
		for (int j = 0; j < 4; j++) {
			int idx = mesh.rect(i, j);
			if (use[idx])
				
				if (slt_point_rect[idx].size() == 0)
					slt_point_rect[idx].push_back(
						std::make_pair(mesh.rect(i, (j + 3) % 4), mesh.rect(i, (j + 1) % 4)));
				else {
					Eigen::RowVector3d V[2];
					cal_nor_vec(V[0], mesh.vtx.row(slt_point_rect[idx][0].first), mesh.vtx.row(slt_point_rect[idx][0].second), mesh.vtx.row(idx));
					cal_nor_vec(V[1], mesh.vtx.row(mesh.rect(i, (j + 3) % 4)), mesh.vtx.row(mesh.rect(i, (j + 1) % 4)), mesh.vtx.row(idx));
					if (V[0].dot(V[1]) > 0)
						slt_point_rect[idx].push_back(
							std::make_pair(mesh.rect(i, (j + 3) % 4), mesh.rect(i, (j + 1) % 4)));
					else
						slt_point_rect[idx].push_back(
							std::make_pair(mesh.rect(i, (j + 1) % 4), mesh.rect(i, (j + 3) % 4)));
				}
		}
	num = 0;
	for (int i = 0; i < 20000; i++) 
		if (use[i]) num++;
	fprintf(fp, "%d\n", num);
	for (int i = 0; i < 20000; i++) {
		if (use[i]) {
			fprintf(fp, "%d %d ", i, slt_point_rect[i].size());
			for (int j = 0; j < slt_point_rect[i].size(); j++)
				fprintf(fp, " %d %d", slt_point_rect[i][j].first, slt_point_rect[i][j].second);
			fprintf(fp, "\n");
		}
	}
	fclose(fp);
}
void draw_tst_slt_pts(Eigen::MatrixXi &slt_pts, int idx, Mesh_my mesh) {
	float scale = 1.01;
	glPointSize(5);
	printf("now idx %d angle: %d\n", idx, (idx*10)-90);
	for (int i = 0; i < G_line_num; i++) {
		//printf("%d %d \n",i, slt_pts(idx, i));
		glBegin(GL_POINTS);
		glVertex3f(mesh.vtx(slt_pts(idx, i), 0)*scale, mesh.vtx(slt_pts(idx, i), 1)*scale, mesh.vtx(slt_pts(idx, i), 2)*scale);
		glEnd();
	}
	//for (int i = 70; i < 84; i++) {
	//	//printf("%d %d %d\n",i, scrtch_line[idx][i], scrtch_line[idx][i-1]);
	//	glBegin(GL_POINTS);
	//	glVertex3f(mesh.vtx(slt_pts(idx,i), 0)*scale, mesh.vtx(slt_pts(idx, i), 1)*scale, mesh.vtx(slt_pts(idx, i), 2)*scale);
	//	glEnd();
	//}
}