#include "ofApp.h"
//#define test_coef
//#define server
//#define test_bldshps_coef
//#define check_out_slt
//#define check_inner
//#define check_fan3d_land_cor
//#define test_pspctv
//#define smooth_mesh_def
//#define test_update_slt_norm_def
#define deal_slt_line_def
//#define test_updt_slt_def
int slt_line_idx = 0, slt_line_total = 20;

Mesh_my mesh,mesh_ref;
Eigen::VectorXi all_land_cor(G_land_num);
Eigen::VectorXi slt_cddt_idx(G_line_num);
Eigen::MatrixX3f coef_land;
Eigen::MatrixX3f coef_mesh;
Eigen::MatrixX3f test_slt_norm;
Eigen::MatrixXi range_slt_pts;

Eigen::Vector3f mi_rgb, ma_rgb;

std::vector <int> scrth_slt_line[1000];
int scrth_line_num;
int test_coef_num = 0, test_coef_num_tot=10;
const int const_smooth_iteration = 2;



//--------------------------------------------------------------
void ofApp::setup(){
	init_mesh("D:\\sydney\\first\\data\\Tester_ (32)\\TrainingPose/pose_0.obj",mesh);

	//get_positive_point(mesh);

#ifdef test_pspctv
	init_mesh("D:\\sydney\\first\\data\\Tester_ (1)\\TrainingPose/pose_9.obj", mesh_ref);
#endif

	
	//get_silhouette_vertex(mesh);
#ifdef test_coef
#ifdef server
	get_coef_land(coef_land, test_coef_num_tot,
		"D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/server/test_coef_land_olsgm_25.txt");
	get_coef_mesh(coef_mesh,
		"D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/server/test_coef_mesh_olsgm_25.txt");
	test_coef_mesh(mesh, coef_mesh, test_coef_num);
#ifdef test_pspctv
	test_coef_mesh(mesh_ref, coef_mesh, test_coef_num+1);
#endif
#else
	get_coef_land(coef_land,
		"D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/test_coef_land_olsgm_25_0.txt");
	get_coef_mesh(coef_mesh,
		"D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/test_coef_mesh_olsgm_25_0.txt");
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
#ifdef test_update_slt_norm_def
	test_update_slt_norm(mesh, test_slt_norm, slt_cddt_idx, "slt_line_4_10.txt", "slt_rect_4_10.txt");
#endif //test_update_slt_norm_def

#ifdef deal_slt_line_def
	//slt_line_left_scratch
	get_scratch_line("slt_line_new.txt", scrth_line_num, scrth_slt_line);
	//get_scratch_line("slt_line_4_10.txt", scrth_line_num, scrth_slt_line);
	//deal_scratch_line(scrth_line_num, scrth_slt_line);
	//get_silhouette_rect(mesh, "slt_line_4_10.txt");
#endif // deal_slt_line_def

#ifdef test_updt_slt_def
	get_tst_slt_pts(range_slt_pts, slt_line_total);
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
	/*slt_line_idx = ((int)(ofGetLastFrameTime()) + 1) % (slt_line_total-5);
	if (slt_line_idx == 0) slt_line_idx = 5;*/
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
	check_3d_fan_corr(mesh, all_land_cor);
	ofPopMatrix();
#endif // check_fan3d_land_cor

#ifdef check_inner
	ofPushMatrix();
	ofScale(ofGetWidth() / 3);
	ofSetColor(ofColor(5, 5, 250));
	check_2d_3d_inner_corr(mesh);
	ofPopMatrix();
#endif
#ifdef check_out_slt
	ofPushMatrix();
	ofScale(ofGetWidth() / 3);
	ofSetColor(ofColor(255, 255, 250));
	check_2d_3d_out_corr(mesh);
	ofPopMatrix();
#endif // check_out_slt

#ifdef deal_slt_line_def
	ofPushMatrix();
	ofScale(ofGetWidth() / 3);
	ofSetColor(ofColor(0, 250,0));
	show_scratch_line(scrth_line_num, scrth_slt_line, mesh);
	ofPopMatrix();
#endif // deal_slt_line_def

	
	//test_slt();

#ifdef test_coef
	ofPushMatrix();
	ofScale(ofGetWidth() / 3);
	ofSetColor(ofColor(255, 0, 5));
#ifdef test_pspctv
	ofTranslate(0.5, 0);
#endif
	test_coef_land(coef_land, test_coef_num);
	printf("%d\n", test_coef_num);
	ofPopMatrix();
#endif
	

	ofPushMatrix();
	ofSetColor(ofColor(133, 180, 250));
	ofScale(ofGetWidth() / 3);
#ifdef test_pspctv
	ofTranslate(0.5,0);
#endif
	draw_mesh(mesh);
	ofSetColor(ofColor(233, 218, 225));
	draw_mesh_point(mesh);
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

#ifdef  test_update_slt_norm_def
	ofPushMatrix();
	ofSetColor(ofColor(0, 0, 0));
	ofScale(ofGetWidth() / 3);
	draw_test_slt_norm(test_slt_norm, slt_cddt_idx,mesh);
	ofPopMatrix();
#endif //  test_updae_slt


#ifdef test_updt_slt_def
	ofPushMatrix();
	
	ofScale(ofGetWidth() / 3);
	ofSetColor(ofColor(255.0, 100, 100));
	draw_tst_slt_pts(range_slt_pts, slt_line_idx, mesh);
	//for (int i = 0; i < range_slt_pts.rows(); i+=4) {
	//	ofSetColor(ofColor(i*(255.0/ range_slt_pts.rows()), 100, 100));
	//	draw_tst_slt_pts(range_slt_pts,i,mesh);
	//}
	ofPopMatrix();
	
#endif
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
	printf("%d\n", key);
#ifdef test_coef
	if (key == 'b') {
		test_coef_num = (test_coef_num + 1) % test_coef_num_tot;
		test_coef_mesh(mesh, coef_mesh, test_coef_num);
	}
#endif
#ifdef test_updt_slt_def
	if (key=='a')
		slt_line_idx = (slt_line_idx + 1) % (slt_line_total);
#endif // test_updt_slt_def
	
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
