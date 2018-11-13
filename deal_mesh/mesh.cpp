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
		/*if (check(mesh,i)) continue;*/

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

void check_2d_3d_inner_jaw_corr(Mesh_my &mesh, Eigen::VectorXi &cor) {
	puts("loading inner point");
	FILE *fp;
	fopen_s(&fp, "D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/inner_jaw/inner_vertex_corr.txt", "r");
	cor.resize(G_jaw_land_num + G_inner_land_num);
	for (int i = 0; i < G_inner_land_num; i++)
		fscanf_s(fp, "%d", &cor(i));
	fclose(fp);
	puts("loading inner point");
	fopen_s(&fp, "D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/inner_jaw/jaw_vertex.txt", "r");
	for (int i = G_inner_land_num; i < G_inner_land_num + G_jaw_land_num; i++)
		fscanf_s(fp, "%d", &cor(i));
	fclose(fp);
	puts("loading inner point");
	glPointSize(15);
	for (int i = 0; i < G_inner_land_num + G_jaw_land_num; i++) {
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
	
	//puts("asd");
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

void get_coef_land(Eigen::MatrixX3f &coef_land,std::string name) {
	puts("get_coef_land");
	FILE *fp;
	fopen_s(&fp, name.c_str(), "r");
	int num;
	fscanf_s(fp, "%d", &num);
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