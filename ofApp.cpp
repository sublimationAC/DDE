#include "ofApp.h"
//#define test_coef

Mesh_my mesh;
Eigen::VectorXi cor(G_inner_land_num);
Eigen::MatrixX3f coef_land;
Eigen::MatrixX3f coef_mesh;
const int test_coef_num = 3;
//--------------------------------------------------------------
void ofApp::setup(){
	init_mesh("D:\\sydney\\first\\data\\Tester_ (1)\\TrainingPose/pose_1.obj",mesh);
	FILE *fp;
	fopen_s(&fp, "D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/inner_vertex_corr.txt", "r");
	
	for (int i = 0; i<G_inner_land_num; i++)
		fscanf_s(fp, "%d", &cor(i));
	fclose(fp);

#ifdef test_coef
	get_silhouette_vertex(mesh);
	get_coef_land(coef_land);
	get_coef_mesh(coef_mesh);
	test_coef_mesh(mesh, coef_mesh, test_coef_num);
#endif // test_coef

	


	lights.resize(2);
	float light_distance = 300.;
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

	//ofPushMatrix();
	//ofTranslate(500, 250);

	//ofSetColor(ofColor(255, 5, 0));
	ofScale(ofGetWidth() / 5);
	//check_2d_3d_corr(mesh, cor);
	ofSetColor(ofColor(255, 255, 250));
	check_2d_3d_out_corr(mesh);
	test_slt();

#ifdef test_coef
	ofSetColor(ofColor(255, 0, 5));
	ofScale(ofGetWidth() / 5);
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
