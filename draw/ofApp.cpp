#include "ofApp.h"


//const int per_frame = 5;
const int tot_obj = 1;
Mesh_my face[tot_obj+1];
Mesh_my face_ref;
std::string mesh_name = "D:\\sydney\\first\\code\\2017\\DDE\\FaceX\\photo_test\\video\\lv_mp4\\lv_file/lv_file/lv_mp4_1";



//--------------------------------------------------------------
void ofApp::setup() {
	string name;
	for (int i = 0; i < tot_obj; i++) {
		printf("Loading %d now\n", i * 2 + 1);
		name = mesh_name;
		name = name + to_string(i*2+1) +"_0norm.obj";

		//cout << name << '\n';
		//system("pause");
		init(name, face[i], 0.6);
	}
	/*init(name_horse, cal_horse, 300);
	init(name_camel, cal_camel, 1.5);*/
	idx = 0;// set the beginning of the obj
	// lights setting 
	lights.resize(3);
	float light_distance = 5000.;
	lights[0].setPosition(2.0*light_distance, 1.0*light_distance, 0.);
	lights[1].setPosition(-1.0*light_distance, -1.0*light_distance, -1.0* light_distance);
//	lights[2].setPosition(0, 0,- 5.0* light_distance);
}

//--------------------------------------------------------------

void ofApp::update() {
	idx++;
	if (idx >= tot_obj) idx = 0;
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
	ofScale(ofGetWidth() / 3);
	ofSetColor(ofColor(133, 180, 250));
	draw_mesh(face[idx]);
	ofPopMatrix();


	//sleep(10);
	//ofDrawCircle(10, 100, 10);
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
