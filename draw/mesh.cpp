#include "mesh.h"
void init(std:: string &name,Mesh_my &mesh,double scale, std::string &lv_name, Eigen :: VectorXi &land_cor) {
	printf("Initiating mesh...\n");
	igl::readOBJ(name, mesh.vtx, mesh.rect);	
	mesh.num_rect = mesh.rect.rows();
	mesh.num_vtx = mesh.vtx.rows();
	mesh.vtx *= scale;
	cal_norm(mesh);
	load_land_cor_from_lv(lv_name, land_cor);
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
			if (i < 2) printf("-%d %d %.10f %.10f %.10f\n", i, l, v[2](0), v[2](1), v[2](2));
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

void itplt(Mesh_my *pre_mesh,Mesh_my *nex_mesh,Mesh_my &mesh,int num,int per_frame) {
	for (int i = 0, tot = pre_mesh->num_vtx; i < tot; i++) {
		Eigen::Vector3d temp = nex_mesh->vtx.row(i) - pre_mesh->vtx.row(i);
		for (int j = 0; j < 3; j++) temp(j) = temp(j)*num / per_frame;
		mesh.vtx.row(i) = pre_mesh->vtx.row(i)+temp.transpose();
	}
	cal_norm(mesh);
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

void draw_land(Mesh_my &mesh,Eigen::VectorXi &land_cor) {
	float scale = 1.01;
	glPointSize(8);
	for (int i = 0; i < G_land_num; i++) {
		glBegin(GL_POINTS);
		glVertex3f(mesh.vtx(land_cor(i), 0)*scale, mesh.vtx(land_cor(i), 1)*scale, mesh.vtx(land_cor(i), 2)*scale);
		glEnd();
	}

	glLineWidth(3);
	for (int i = 1; i < 15; i++) {
		glBegin(GL_LINES);
		glVertex3f(mesh.vtx(land_cor(i), 0)*scale, mesh.vtx(land_cor(i), 1)*scale, mesh.vtx(land_cor(i), 2)*scale);
		glVertex3f(mesh.vtx(land_cor(i - 1), 0)*scale, mesh.vtx(land_cor(i - 1), 1)*scale, mesh.vtx(land_cor(i - 1), 2)*scale);
		glEnd();
	}
}