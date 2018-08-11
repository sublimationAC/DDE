#include "mesh.h"


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
		if (mesh.vtx(mesh.rect(i,0), 2) < 0.1) continue;
		glBegin(GL_QUADS);

		for (int j = 0; j < 4; ++j) {
			//printf("%d %d\n", mesh.norm_vtx.rows(), mesh.norm_vtx.cols());
			//printf("%d %d %d\n", i, j, mesh.tri(i, j));
			int VertIndex = mesh.rect(i, j);

			/*if (i < 5) {
				printf("+%.10f %.10f %.10f\n", mesh.norm_vtx(VertIndex, 0), mesh.norm_vtx(VertIndex, 1), mesh.norm_vtx(VertIndex, 2));
				printf("-%.10f %.10f %.10f\n", mesh.vtx(VertIndex, 0), mesh.vtx(VertIndex, 1), mesh.vtx(VertIndex, 2));
			}*/
			GLdouble normal[3] = { mesh.norm_vtx(VertIndex, 0), mesh.norm_vtx(VertIndex, 1), mesh.norm_vtx(VertIndex, 2) };
			glNormal3dv(normal);

			glVertex3f(mesh.vtx(VertIndex, 0), mesh.vtx(VertIndex, 1), mesh.vtx(VertIndex, 2));
		}
		glEnd();

	}
}

void draw_line(Mesh_my &mesh,double agl){
	double scale = 1.0001;
	glLineWidth(5);
	for (int i = 0; i < mesh.num_rect; ++i) {
		//printf("%d\n",i);
		for (int t = 0; t < 4; t++) {
			int idx1=mesh.rect(i,t),idx2=mesh.rect(i,(t + 1) % 4);
			if (mesh.vtx(idx1, 1) - mesh.vtx(idx2, 1) <= 0.01 && mesh.vtx(idx1, 1) - mesh.vtx(idx2, 1) >= -0.01) {
				if (mesh.norm_vtx(idx1, 0)*sin(agl) + mesh.norm_vtx(idx1, 2)*cos(agl) > 0.1) continue;
				if (mesh.norm_vtx(idx1, 2)*sin(agl) + mesh.norm_vtx(idx1, 2)*cos(agl) > 0.1) continue;
				if (fabs(mesh.norm_vtx(idx1,0)*sin(agl)+mesh.norm_vtx(idx1, 2)*cos(agl)) > 0.6) continue;
				if (fabs(mesh.norm_vtx(idx2, 2)*sin(agl) +mesh.norm_vtx(idx2, 2)*cos(agl)) > 0.6) continue;
				//puts("asd");
				glBegin(GL_LINES);
				glVertex3f(mesh.vtx(idx1, 0)*scale, mesh.vtx(idx1, 1)*scale, mesh.vtx(idx1, 2)*scale);
				glVertex3f(mesh.vtx(idx2, 0)*scale, mesh.vtx(idx2, 1)*scale, mesh.vtx(idx2, 2)*scale);
				glEnd();
			}
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

void check_2d_3d_corr(Mesh_my &mesh) {
	FILE *fp;
	fopen_s(&fp, "inner_vertex_corr.txt", "r");
	Eigen::VectorXi cor(59);
	for (int i=0;i<59;i++)
		fscanf_s(fp,"%d",&cor(i));
	glPointSize(15);
	for (int i = 0; i < 59; i++) {
		glBegin(GL_POINTS);
		glVertex3f(mesh.vtx(cor(i), 0), mesh.vtx(cor(i), 1), mesh.vtx(cor(i), 2));
		glEnd();
	}
}