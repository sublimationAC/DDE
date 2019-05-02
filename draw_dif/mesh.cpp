#include "mesh.h"
//#define test_slt_drct
#define slt_file
#include "GL/glut.h"


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

void get_coef_land(Eigen::MatrixX3f &coef_land, std::string name) {
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

void test_coef_land(Eigen::MatrixX3f &coef_land, int idx) {

	glPointSize(5);
	for (int i = idx * G_land_num; i < G_land_num*(idx + 1); i++) {
		glBegin(GL_POINTS);
		glVertex3f(coef_land(i, 0), coef_land(i, 1), coef_land(i, 2));
		glEnd();
	}
}

void get_coef_mesh(Eigen::MatrixX3f &coef_mesh, std::string name) {
	puts("get_coef_mesh");
	FILE *fp;
	int num = 0;
	num = fopen_s(&fp, name.c_str(), "r");
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
			mesh.vtx(i, j) = coef_mesh(idx*G_nVerts + i, j);

	cal_norm(mesh);
}


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
	glPointSize(5);
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



void get_mima(Mesh_my &mesh, Mesh_my &mesh_ref, float &mi, float &ma, int axis) {
	mi = 1e9;
	ma = -10;
	for (int i = 0; i < mesh.num_vtx; i++) {
		mi = std::min(mi, (float) fabs(mesh.vtx(i, axis) - mesh_ref.vtx(i, axis)));
		ma = std::max(ma, (float)fabs(mesh.vtx(i, axis) - mesh_ref.vtx(i, axis)));
	}
}

