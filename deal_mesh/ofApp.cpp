#include "ofApp.h"
//#define test_coef
#define test_bldshps_coef
//#define check_out_slt
//#define check_inner_jaw

Mesh_my mesh;
Eigen::VectorXi cor(G_inner_land_num);
Eigen::MatrixX3f coef_land;
Eigen::MatrixX3f coef_mesh;
const int test_coef_num = 2;
//--------------------------------------------------------------
void ofApp::setup(){
	init_mesh("D:\\sydney\\first\\data\\Tester_ (1)\\TrainingPose/pose_0.obj",mesh);
	

	
	//get_silhouette_vertex(mesh);
#ifdef test_coef
	
	get_coef_land(coef_land,
		"D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/server/test_coef_land_olsgm_25.txt");
	get_coef_mesh(coef_mesh,
		"D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/server/test_coef_mesh_olsgm_25.txt");
	test_coef_mesh(mesh, coef_mesh, test_coef_num);
#endif // test_coef
#ifdef test_bldshps_coef	
	get_coef_mesh(coef_mesh,
		"D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/test_svd_bldshps.txt");//server/test_exp_coeff_mesh.txt");
	test_coef_mesh(mesh, coef_mesh, test_coef_num);
#endif


	lights.resize(2);
	float light_distance = 5000.;
	lights[0].setPosition(2.0*light_distance, 1.0*light_distance, 0.);
	lights[1].setPosition(-1.0*light_distance, -1.0*light_distance, -1.0* light_distance);
}

//--------------------------------------------------------------
void ofApp::update(){
	//cameraOrbit += ofGetLastFrameTime() * 20.; // 20 degrees per second;
	//cam.orbitDeg(cameraOrbit, 0., cam.getDistance(), { 0., 0., 0. });
}

//--------------------------------------------------------------
void ofApp::draw(){
	ofBackgroundGradient(ofColor(200), ofColor(25));

	glEnable(GL_DEPTH_TEST);//?????

	ofEnableLighting();
	for (int i = 0; i < lights.size(); ++i) { lights[i].enable(); }
	cam.begin(); //?????????????
	ofScale(ofGetWidth() / 3);
	//ofPushMatrix();
	//ofTranslate(500, 250);

	//ofSetColor(ofColor(255, 5, 0));
#ifdef check_inner_jaw
	check_2d_3d_inner_jaw_corr(mesh, cor);
#endif
#ifdef check_out_slt
	
	ofSetColor(ofColor(255, 255, 250));
	check_2d_3d_out_corr(mesh);
#endif // check_out_slt

	
	
	//test_slt();

#ifdef test_coef
	ofSetColor(ofColor(255, 0, 5));
	test_coef_land(coef_land, test_coef_num);
#endif

	ofSetColor(ofColor(133, 180, 250));
	//ofScale(ofGetWidth() / 5);

	draw_mesh(mesh);
	//ofSetColor(ofColor(255, 255, 250));
	//ofScale(ofGetWidth() / 5);
	//check_2d_3d_out_corr(mesh);
	//draw_line(mesh,0);
	//check_2d_3d_corr(mesh);
	
	
	//ofPopMatrix();


	//sleep(10);
	//ofDrawCircle(10, 100, 10);
	cam.end();

	for (int i = 0; i < lights.size(); i++) { lights[i].disable(); }
	ofDisableLighting();
	glDisable(GL_DEPTH_TEST);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
