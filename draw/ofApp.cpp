#include "ofApp.h"
//#define test_fitting_lv
//#define test_updt_slt_def

//const int per_frame = 5;
const int test_updt_slt_def_x = 2;
const int tot_obj = 630;
Mesh_my face[tot_obj+1];
Mesh_my face_ref;
#ifdef perspective
#ifdef test_updt_slt_def
//std::string mesh_name = "G:/DDE/server_lv/Tester_88/X1_Left/pose_";
std::string mesh_name = "G:/DDE/server_lv/slt_test_obj/Tester_88_pose1_";
#else
std::string mesh_name = "G:/DDE/server_lv/lv_out_psp/lv_out_";
#endif

#endif
#ifdef  normalization
std::string mesh_name = "G:/DDE/server_lv/lv_out_l1l2/lv_out_";
#endif //  normalization


Eigen::VectorXi land_cor[tot_obj + 1];
Eigen::MatrixXi range_slt_pts;

//--------------------------------------------------------------
void ofApp::setup() {
	string name;
#ifdef test_fitting_lv
	
	printf("Loading %d now\n", test_updt_slt_def_x);
	name = mesh_name;
	name = name + to_string(test_updt_slt_def_x) + "_0norm.obj";
	
	cout << name << '\n';
#ifdef perspective
	string name_psp_f = mesh_name + to_string(test_updt_slt_def_x) + ".psp_f";
	init(name, face[0], 0.6, name_psp_f, land_cor[0]);
#endif // perspective

#ifdef normalization
	string name_lv = mesh_name + to_string(x) + ".lv";
	init(name, face[0], 0.6, name_lv, land_cor[0]);
#endif // normalization
	
	//system("pause");
#ifdef test_updt_slt_def
	get_tst_slt_pts(range_slt_pts);
#endif // test_updt_slt_def

#else
	for (int i = 0; i < tot_obj; i++) {
		printf("Loading %d now\n", i + 1);
		name = mesh_name;
		name = name + to_string(i+1) +"_0norm.obj";
#ifdef perspective
		string name_lv = mesh_name + to_string(i + 1) + ".psp_f";
#endif // perspective
#ifdef normalization
		string name_lv = mesh_name + to_string(i + 1) + ".lv";
#endif // normalization
		cout << name << '\n';
		//system("pause");
		init(name, face[i], 0.6, name_lv, land_cor[i]);
	}
#endif

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
	idx++;
#ifdef test_updt_slt_def
	if (idx >= 1) idx = 0;
#else
	if (idx >= tot_obj) idx = 0;
#endif // test_updt_slt_def
	printf("%d\n", idx);
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
	draw_mesh(face[idx]);
	ofPopMatrix();

	ofPushMatrix();
	//	ofTranslate(0, -100 );
	ofScale(ofGetWidth() / 3.5);
	ofSetColor(ofColor(0, 250, 0));
	draw_land(face[idx],land_cor[idx]);
	ofPopMatrix();
	//sleep(10);
	//ofDrawCircle(10, 100, 10);

#ifdef test_updt_slt_def
	ofPushMatrix();

	ofScale(ofGetWidth() / 3.5);
	ofSetColor(ofColor(255.0, 100, 100));
	draw_tst_slt_pts(range_slt_pts, test_updt_slt_def_x, face[idx]);
	//for (int i = 0; i < range_slt_pts.rows(); i+=4) {
	//	ofSetColor(ofColor(i*(255.0/ range_slt_pts.rows()), 100, 100));
	//	draw_tst_slt_pts(range_slt_pts,i,mesh);
	//}
	ofPopMatrix();

#endif


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
