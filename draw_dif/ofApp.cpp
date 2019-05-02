#include "ofApp.h"
//#define server
#define server_lv

Mesh_my mesh, mesh_ref;

Eigen::MatrixX3f coef_land;
Eigen::MatrixX3f coef_mesh;

Eigen::Vector3f mi_rgb;
Eigen::Vector3f ma_rgb;

float mi_x, ma_x;

const int test_coef_num = 18;

const int show_color_ma = 10;

//--------------------------------------------------------------
void ofApp::setup() {
	
	
#ifdef server_lv
	init_mesh("G:/DDE/server_lv/TrainingPose/pose_0_0norm.obj", mesh);
#else
	init_mesh("D:\\sydney\\first\\data\\Tester_ (2)\\TrainingPose/pose_4.obj", mesh);
#ifdef server
	get_coef_land(coef_land,
		"D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/server/test_coef_land_olsgm_25.txt");
	get_coef_mesh(coef_mesh,
		"D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/server/test_coef_mesh_olsgm_25.txt");
	test_coef_mesh(mesh, coef_mesh, test_coef_num);
#else
	get_coef_land(coef_land,
		"D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/test_coef_land_olsgm_25_0.txt");
	get_coef_mesh(coef_mesh,
		"D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/test_coef_mesh_olsgm_25_0.txt");
#endif
	
#endif
	init_mesh("D:\\sydney\\first\\data\\Tester_ (32)\\TrainingPose/pose_0.obj", mesh_ref);
	/*for (int i = 0; i < 3; i++)
		get_mima(mesh, mesh_ref, mi_rgb(i), ma_rgb(i),i);*/
	//for (int i = 0; i < mesh_ref.num_vtx; i++)
	//	for (int j = 0; j < 3; j++)
	//		mesh_ref.vtx(i, j) = mesh_ref.vtx(i, j) *0.6;

	/*for (int axis = 0; axis < 3; axis++) {
		mi_rgb(axis) = 1e9;
		ma_rgb(axis) = -10;
		for (int i = 0; i < mesh.num_vtx; i++) {
			mi_rgb(axis) = std::min(mi_rgb(axis), (float)fabs(mesh.vtx(i, axis) - mesh_ref.vtx(i, axis)));
			ma_rgb(axis) = std::max(ma_rgb(axis), (float)fabs(mesh.vtx(i, axis) - mesh_ref.vtx(i, axis)));
		}
	}*/

	mi_x = 1e9, ma_x = -1e9;
	for (int i = 0; i < mesh.num_vtx; i++) {
		float t = mesh_ref.vtx(i,0);
		mi_x = std::min(mi_x, t);
		ma_x = std::max(ma_x, t);
	}

	for (int axis = 0; axis < 3; axis++) {
		mi_rgb(axis) = 1e9;
		ma_rgb(axis) = -10;
		for (int i = 0; i < mesh.num_vtx; i++) {
			float t = (mesh.vtx.row(i) - mesh_ref.vtx.row(i)).norm();
			mi_rgb(axis) = std::min(mi_rgb(axis), t);
			ma_rgb(axis) = std::max(ma_rgb(axis), t);
		}
		printf("blue: %.5f~%.5f\ngreen: %.5f~%.5f\nred: %.5f~%.5f\n",
			mi_rgb(axis), mi_rgb(axis) + (ma_rgb(axis) - mi_rgb(axis)) / 3,
			mi_rgb(axis) + (ma_rgb(axis) - mi_rgb(axis)) / 3, mi_rgb(axis) + 2 * (ma_rgb(axis) - mi_rgb(axis)) / 3,
			mi_rgb(axis) + 2 * (ma_rgb(axis) - mi_rgb(axis)) / 3, ma_rgb(axis));

	}

	printf("x range:%.5f %.5f\n", mi_x, ma_x);

	//FILE *fp;
	//fopen_s(&fp, "zbigger0_idx.txt", "w");
	//for (int i = 0; i < mesh_ref.num_rect; ++i) {
	//	bool fl = 0;
	//	for (int t = 0; t < 4; t++) {
	//		int VertIndex = mesh_ref.rect(i, t);
	//		if (mesh_ref.vtx(VertIndex, 2) < 0) fl = 1;
	//	}
	//	if (fl) continue;

	//	for (int t = 0; t < 4; t++) {
	//		int VertIndex = mesh_ref.rect(i, t);
	//		fprintf(fp, "%d\n", VertIndex);
	//	}
	//	
	//}
	//fclose(fp);

	system("pause");

	lights.resize(2);
	float light_distance = 5000.;
	lights[0].setPosition(2.0*light_distance, 1.0*light_distance, 0.);
	lights[1].setPosition(-1.0*light_distance, -1.0*light_distance, -1.0* light_distance);
}

//--------------------------------------------------------------
void ofApp::update() {
	//cameraOrbit += ofGetLastFrameTime() * 20.; // 20 degrees per second;
	//cam.orbitDeg(cameraOrbit, 0., cam.getDistance(), { 0., 0., 0. });
}

//--------------------------------------------------------------
void ofApp::draw() {
	ofBackgroundGradient(ofColor(200), ofColor(25));

	glEnable(GL_DEPTH_TEST);//?????

	ofEnableLighting();
	for (int i = 0; i < lights.size(); ++i) { lights[i].enable(); }
	cam.begin(); //?????????????


	//ofTranslate(500, 250);
#ifdef test_coef
	ofPushMatrix();
	ofScale(ofGetWidth() / 5);
	ofSetColor(ofColor(255, 0, 5));
#ifdef test_pspctv
	ofTranslate(0.5, 0);
#endif
	test_coef_land(coef_land, test_coef_num);
	ofPopMatrix();
#endif // test_coef

	





	//for (int i = 0; i < mesh.num_vtx; i++) {
	//	ofPushMatrix();

	//	ofScale(ofGetWidth() / 3);
	//	//ofSetColor(
	//	//	ofColor(
	//	//		(fabs(mesh.vtx(i, 0) - mesh_ref.vtx(i, 0))- mi_rgb(0))/(ma_rgb(0)-mi_rgb(0))*255,
	//	//		(fabs(mesh.vtx(i, 1) - mesh_ref.vtx(i, 1)) - mi_rgb(1)) / (ma_rgb(1) - mi_rgb(1)) * 255,
	//	//		(fabs(mesh.vtx(i, 2) - mesh_ref.vtx(i, 2)) - mi_rgb(2)) / (ma_rgb(2) - mi_rgb(2)) * 255));

	//	float t = (mesh.vtx.row(i) - mesh_ref.vtx.row(i)).norm();
	//	//printf("%d %.5f %.5f\n",i, t, (t - mi_rgb(0)) / (ma_rgb(0) - mi_rgb(0)) * 255);
	//	ofSetColor(
	//		ofColor(
	//			0,
	//			(t - mi_rgb(1)) / (ma_rgb(1) - mi_rgb(1)) * 255,
	//			0));
	//	float scale = 1.3;
	//	glPointSize(5);
	//	glBegin(GL_POINTS);
	//	glVertex3f(mesh_ref.vtx(i, 0)*scale, mesh_ref.vtx(i, 1)*scale, mesh_ref.vtx(i, 2)*scale);
	//	glEnd();
	//	ofPopMatrix();
	//}
	ofPushMatrix();
	ofScale(ofGetWidth() / 5);

	for (int i = 0; i < mesh_ref.num_rect; ++i) {
		bool fl = 0;
		for (int t = 0; t < 4; t++) {
			int VertIndex = mesh_ref.rect(i, t);
			if (mesh_ref.vtx(VertIndex, 2) < 0) fl = 1;
		}
		if (fl) continue;

		//printf(


		//glLineWidth(0.3);


		//float t = (mesh_ref.vtx.row(mesh_ref.rect(i, 0)) - mesh.vtx.row(mesh.rect(i, 0))).norm() - mi_rgb(0);		
		//printf("%d %.5f %.5f\n", i, t, t / (ma_rgb(0) - mi_rgb(0)) * 255);
		//if (t < (ma_rgb(0) - mi_rgb(0)) / 3) {
		//	ofSetColor(
		//		ofColor(
		//			0,
		//			0,
		//			t * 3 / (ma_rgb(0) - mi_rgb(0)) * 255
		//		));
		//}
		//else
		//	if (t < (ma_rgb(0) - mi_rgb(0)) * 2 / 3) {
		//		ofSetColor(
		//			ofColor(
		//				0,
		//				(t - (ma_rgb(0) - mi_rgb(0)) / 3) * 3 / (ma_rgb(0) - mi_rgb(0)) * 255,
		//				0
		//			));
		//	}
		//	else {
		//		ofSetColor(
		//			ofColor(
		//			(t - 2 * (ma_rgb(0) - mi_rgb(0)) / 3) * 3 / (ma_rgb(0) - mi_rgb(0)) * 255,
		//				0,
		//				0
		//			));
		//	}
		/*if (t < (ma_rgb(0) - mi_rgb(0)) / 3) {
			glColor3f(
				0,
				0,
				t * 3 / (ma_rgb(0) - mi_rgb(0))
			);
		}
		else
			if (t < (ma_rgb(0) - mi_rgb(0)) * 2 / 3) {
				glColor3f(
					0,
					(t - (ma_rgb(0) - mi_rgb(0)) / 3) * 3 / (ma_rgb(0) - mi_rgb(0)),
					1
				);
			}
			else {
				glColor3f(
					(t - 2 * (ma_rgb(0) - mi_rgb(0)) / 3) * 3 / (ma_rgb(0) - mi_rgb(0)),
					1,
					1
				);
			}*/
		float t = (mesh_ref.vtx.row(mesh_ref.rect(i, 0)) - mesh.vtx.row(mesh_ref.rect(i, 0))).norm() * 100;

		//assert(t <= show_color_ma * 2);
		if (t > 2 * show_color_ma) {
			glColor3f(
				1,
				0,
				0
			);
		}
		else

			if (t < show_color_ma) {
				glColor3f(
					0,
					t / show_color_ma,
					1 - t / show_color_ma
				);
			}
			else
			{
				glColor3f(
					(t - show_color_ma) / show_color_ma,
					1 - (t - show_color_ma) / show_color_ma,
					0
				);
			}

		float scale = 1.5;
		for (int t = 0; t < 1; t++) {
			glBegin(GL_TRIANGLES);
			for (int p = 0; p < 3; p++) {
				int VertIndex = mesh_ref.rect(i, (t + p) % 4);
				GLdouble normal[3] = { mesh_ref.norm_vtx(VertIndex, 0), mesh_ref.norm_vtx(VertIndex, 1), mesh_ref.norm_vtx(VertIndex, 2) };
				glNormal3dv(normal);
				glVertex3f(mesh_ref.vtx(VertIndex, 0)*scale, mesh_ref.vtx(VertIndex, 1)*scale, mesh_ref.vtx(VertIndex, 2)*scale);
			}
			glEnd();
		}
		for (int t = 2; t < 3; t++) {
			glBegin(GL_TRIANGLES);
			for (int p = 0; p < 3; p++) {
				int VertIndex = mesh_ref.rect(i, (t + p) % 4);
				GLdouble normal[3] = { mesh_ref.norm_vtx(VertIndex, 0), mesh_ref.norm_vtx(VertIndex, 1), mesh_ref.norm_vtx(VertIndex, 2) };
				glNormal3dv(normal);
				glVertex3f(mesh_ref.vtx(VertIndex, 0)*scale, mesh_ref.vtx(VertIndex, 1)*scale, mesh_ref.vtx(VertIndex, 2)*scale);
			}
			glEnd();
		}

	}
	ofPopMatrix();
	/*ofPushMatrix();
	ofScale(ofGetWidth() / 3);
	for (int
	ofSetColor(ofColor(0, 250, 0));

	ofPopMatrix();*/



	cam.end();

	for (int i = 0; i < lights.size(); i++) { lights[i].disable(); }
	ofDisableLighting();
	glDisable(GL_DEPTH_TEST);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key) {

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h) {

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg) {

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo) {

}
