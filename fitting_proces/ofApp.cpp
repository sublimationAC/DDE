#include "ofApp.h"





//const int per_frame = 5;
const int test_updt_slt_def_x = 2;
const int max_tot_obj = 210;

Mesh_my face_ref;
Mesh_my mesh_std, mesh_sq[max_tot_obj + 1];

Eigen::MatrixX3f land_sq[max_tot_obj + 1];
Eigen::MatrixXi range_slt_pts;

Eigen::MatrixX3f coef_land;
Eigen::MatrixX3f coef_mesh;

int tot_obj_num = 0;

//--------------------------------------------------------------
void ofApp::setup() {
	string name;


	init_mesh("D:\\sydney\\first\\data\\Tester_ (32)\\TrainingPose/pose_0.obj", mesh_std);
	float scale = 0.6;
	puts("A");
#ifdef test_svd_vec_def
	get_coef_mesh(coef_mesh, tot_obj_num,
		//"D:\\sydney\\first\\code\\2017\\deal_data_2\\py/test_bldshps_vector_c_20_-0.50.5.txt", scale);get_coef_mesh(coef_mesh, tot_obj_num,
		"D:\\sydney\\first\\code\\2017\\deal_data_2\\py/test_bldshps_one_vector_user0.txt", scale);

#else
	puts("A");
	get_coef_land(coef_land, 
		"D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/server/test_coef_land_olsgm_25.txt", scale);
	get_coef_mesh(coef_mesh, tot_obj_num,
		"D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/server/test_coef_mesh_olsgm_25.txt", scale);
#endif

	for (int i = 0; i < tot_obj_num; i++)
		test_coef_mesh(mesh_std, mesh_sq[i], land_sq[i], coef_mesh, coef_land, i);

	idx = 0;// set the beginning of the obj
	// lights setting 
	lights.resize(2);

	float light_distance = 5000.;
	lights[0].setPosition(2.0*light_distance, 1.0*light_distance, 0.);
	lights[1].setPosition(-1.0*light_distance, -1.0*light_distance, -1.0* light_distance);
	//	lights[2].setPosition(0, 0,- 5.0* light_distance);
}

//--------------------------------------------------------------

void ofApp::update() {
//	idx++;
//#ifdef test_updt_slt_def
//	if (idx >= 1) idx = 0;
//#else
//	if (idx >= tot_obj_num) idx = 0;
//#endif // test_updt_slt_def
//	printf("%d\n", idx);
	//cameraOrbit += ofGetLastFrameTime() * 20.; // 20 degrees per second;
	//cam.orbitDeg(cameraOrbit, 0., cam.getDistance(), { 0., 0., 0. });
	/*if (idx >= (tot_obj-1) * per_frame) idx = 0;
	int pre = idx / per_frame + 1, nex = pre + 1;
	int num = idx - (pre - 1)*per_frame;
	printf("%d %d %d %d\n",idx,pre,nex,num);
	itplt(&mesh_horse[pre],&mesh_horse[nex],cal_horse,num,per_frame);
	itplt(&mesh_camel[pre],&mesh_camel[nex], cal_camel,num, per_frame);*/
}

//--------------------------------------------------------------
//#include <Windows.h>
void ofApp::draw() {

	ofBackgroundGradient(ofColor(100), ofColor(25));

	glEnable(GL_DEPTH_TEST);//?????

	ofEnableLighting();
	for (int i = 0; i < lights.size(); ++i) { lights[i].enable(); }

	cam.begin(); //?????????????
	
	ofPushMatrix();
//	ofTranslate(0, -100 );
	ofScale(ofGetWidth() / 3.5);
	ofSetColor(ofColor(133, 180, 250));
	draw_mesh(mesh_sq[idx]);
	printf("%d\n", idx);
	
	ofPopMatrix();


#ifndef test_svd_vec_def
	ofPushMatrix();
	//	ofTranslate(0, -100 );
	ofScale(ofGetWidth() / 3.5);
	ofSetColor(ofColor(0, 250, 0));
	draw_land(land_sq[idx]);
	ofPopMatrix();
	//sleep(10);
	//ofDrawCircle(10, 100, 10);
#else
	printf("vector: %d  coef: %.2f\n", idx / 10, (idx % 10)*0.1 - 0.5);
#endif // test_svd_vec_def


	cam.end();

	for (int i = 0; i < lights.size(); i++) { lights[i].disable(); }
	ofDisableLighting();
	glDisable(GL_DEPTH_TEST);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
	//printf("vector: %d  coef: %.2f\n", idx/10,(idx%10)*0.1-0.5);
	printf("%d\n", key);
	if (key == 'n')
		idx = (idx + 1) % tot_obj_num;
	if (key == 'l')
		idx = (idx - 1+ tot_obj_num) % tot_obj_num;
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
