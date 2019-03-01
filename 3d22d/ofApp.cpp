#include "ofApp.h"
#define test_coef
//#define server
//#define test_bldshps_coef
//#define check_out_slt
//#define check_inner_jaw
//#define check_fan3d_land_cor
//#define test_pspctv
//#define smooth_mesh_def
#define test_update_slt

Mesh_my mesh,mesh_ref;
Eigen::VectorXi cor(G_land_num);
Eigen::MatrixX3f coef_land;
Eigen::MatrixX3f coef_mesh;
Eigen::MatrixX3f test_slt_norm;
const int test_coef_num = 0;
const int const_smooth_iteration = 2;
//--------------------------------------------------------------
void ofApp::setup(){
	init_mesh("D:\\sydney\\first\\data\\Tester_ (1)\\TrainingPose/pose_1.obj",mesh);

	//get_positive_point(mesh);

#ifdef test_pspctv
	init_mesh("D:\\sydney\\first\\data\\Tester_ (1)\\TrainingPose/pose_9.obj", mesh_ref);
#endif

	
	//get_silhouette_vertex(mesh);
#ifdef test_coef
#ifdef server
	get_coef_land(coef_land,
		"D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/server/test_coef_land_olsgm_25.txt");
	get_coef_mesh(coef_mesh,
		"D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/server/test_coef_mesh_olsgm_25.txt");
	test_coef_mesh(mesh, coef_mesh, test_coef_num);
#ifdef test_pspctv
	test_coef_mesh(mesh_ref, coef_mesh, test_coef_num+1);
#endif
#else
	get_coef_land(coef_land,
		"D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/test_coef_land_olsgm_25.txt");
	get_coef_mesh(coef_mesh,
		"D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/test_coef_mesh_olsgm_25.txt");
	//get_coef_land(coef_land,
	//	"D:\\sydney\\first\\code\\2017\\fan3d/fan3d/test_coef_land_olsgm_25.txt");
	//get_coef_mesh(coef_mesh,
	//	"D:\\sydney\\first\\code\\2017\\fan3d/fan3d/test_coef_mesh_olsgm_25.txt");
	test_coef_mesh(mesh, coef_mesh, test_coef_num);
#endif
#endif // test_coef
#ifdef test_bldshps_coef	
	get_coef_mesh(coef_mesh,
		"D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/test_svd_bldshps_test.txt");// server / test_exp_coeff_mesh.txt");//test_svd_bldshps_77_ide1.txt"); //
	test_coef_mesh(mesh, coef_mesh, test_coef_num);
#endif

#ifdef smooth_mesh_def
	smooth_mesh(mesh, const_smooth_iteration);
#endif // smooth_mesh_def
	//get_mouse_data(mesh);
#ifdef test_update_slt
	test_update_slt_norm(mesh, test_slt_norm);
#endif //test_update_slt
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
	
	
	//ofTranslate(500, 250);

	//ofSetColor(ofColor(255, 5, 0));
#ifdef check_fan3d_land_cor
	ofPushMatrix();
	ofScale(ofGetWidth() / 3);
	ofSetColor(ofColor(5, 5, 250));
	check_3d_fan_corr(mesh, cor);
	ofPopMatrix();
#endif // check_fan3d_land_cor

#ifdef check_inner_jaw
	ofPushMatrix();
	ofScale(ofGetWidth() / 3);
	ofSetColor(ofColor(5, 5, 250));
	check_2d_3d_inner_jaw_corr(mesh, cor);
	ofPopMatrix();
#endif
#ifdef check_out_slt
	
	ofSetColor(ofColor(255, 255, 250));
	check_2d_3d_out_corr(mesh);
#endif // check_out_slt

	
	
	//test_slt();

#ifdef test_coef
	ofPushMatrix();
	ofScale(ofGetWidth() / 3);
	ofSetColor(ofColor(255, 0, 5));
#ifdef test_pspctv
	ofTranslate(0.5, 0);
#endif
	test_coef_land(coef_land, test_coef_num);
	ofPopMatrix();
#endif
	

	ofPushMatrix();
	ofSetColor(ofColor(133, 180, 250));
	ofScale(ofGetWidth() / 3);
#ifdef test_pspctv
	ofTranslate(0.5,0);
#endif
	draw_mesh(mesh);
	//ofSetColor(ofColor(253, 0, 0));
	//draw_line(mesh, 0);
	//test_mouse(mesh);
	ofPopMatrix();

#ifdef test_pspctv
#ifdef test_coef
	ofPushMatrix();
	ofScale(ofGetWidth() / 3);
	ofSetColor(ofColor(255, 0, 5));
	ofTranslate(-0.5, 0);
	test_coef_land(coef_land, test_coef_num + 1);
	ofPopMatrix();
#endif

	ofPushMatrix();
	ofSetColor(ofColor(133, 180, 250));
	ofScale(ofGetWidth() / 3);
	ofTranslate(-0.5, 0);
	draw_mesh(mesh_ref);
	ofPopMatrix();
#endif // test_pspctv

#ifdef  test_update_slt
	ofPushMatrix();
	ofSetColor(ofColor(0, 0, 0));
	ofScale(ofGetWidth() / 3);
	draw_test_slt_norm(test_slt_norm);
	ofPopMatrix();
#endif //  test_updae_slt



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
