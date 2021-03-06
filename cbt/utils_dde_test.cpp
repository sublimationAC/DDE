#include "utils_dde_test.hpp"


float cal_3d_vtx(
	Eigen::MatrixXf &bldshps,
	Eigen ::VectorXf &user, Eigen::VectorXf &exp, int vtx_idx, int axis) {

	//puts("calculating one vertex coordinate...");
	float ans = 0;

	for (int i_id = 0; i_id < G_iden_num; i_id++)
		for (int i_shape = 0; i_shape < G_nShape; i_shape++)
			if (i_shape == 0)
				ans += exp(i_shape)*user(i_id)
				*bldshps(i_id, vtx_idx * 3 + axis);
			else
				ans += exp(i_shape)*user(i_id)
				*(bldshps(i_id, 3 * G_nVerts*i_shape + vtx_idx * 3 + axis) - bldshps(i_id, vtx_idx * 3 + axis));
	return ans;
}
//void recal_dis(DataPoint &data, Eigen::MatrixXf &bldshps) {
//	puts("calculating displacement...");
//	Eigen::MatrixX2f land(G_land_num, 2);
//	Eigen::RowVector2f T = data.shape.tslt.block(0, 0, 1, 2);
//	for (int i_v = 0; i_v < G_land_num; i_v++) {
//		Eigen::Vector3f v;
//		for (int axis = 0; axis < 3; axis++)
//			v(axis) = cal_3d_vtx(bldshps, data.user, data.shape.exp, data.land_cor(i_v), axis);
//		land.row(i_v) = ((data.s) * ((data.shape.rot) * v)).transpose() + T;
//	}
//
//	data.shape.dis.array() = data.land_2d.array() - land.array();
//
//}

void recal_dis_ang(DataPoint &data, Eigen::MatrixXf &bldshps) {
	puts("calculating displacement...");
	Eigen::MatrixX2f land(G_land_num, 2);
	Eigen::Matrix3f rot = get_r_from_angle_zyx(data.shape.angle);
#ifdef perspective
	Eigen::Vector3f T = data.shape.tslt;
#endif // perspective

#ifdef normalization
	Eigen::RowVector2f T = data.shape.tslt.block(0, 0, 1, 2);
#endif // normalization
	for (int i_v = 0; i_v < G_land_num; i_v++) {
		Eigen::Vector3f v;
		for (int axis = 0; axis < 3; axis++)
			v(axis) = cal_3d_vtx(bldshps, data.user, data.shape.exp, data.land_cor(i_v), axis);
#ifdef perspective
		v = rot * v + T;

		land(i_v, 0) = v(0)*(data.fcs) / v(2) + data.center(0);
		land(i_v, 1) = v(1)*(data.fcs) / v(2) + data.center(1);
#endif // perspective
#ifdef normalization
		land.row(i_v) = ((data.s) * (rot * v)).transpose() + T;
#endif // normalization
		
	}
	data.shape.dis.resize(G_land_num, 2);
	data.shape.dis.array() = data.land_2d.array() - land.array();

}

void cal_mesh(DataPoint &data, Eigen::MatrixXf &bldshps, Eigen::MatrixX3f &mesh) {
	for (int i_v = 0; i_v < G_nVerts; i_v++) {
		for (int axis = 0; axis < 3; axis++)
			mesh(i_v, axis) = cal_3d_vtx(bldshps, data.user, data.shape.exp, i_v, axis);
	}
}
void cal_3d_land(DataPoint &data, Eigen::MatrixXf &bldshps, Eigen::MatrixX3f &land_3d) {
	for (int i_v = 0; i_v < G_land_num; i_v++) {
		printf("^^ %d %d\n", i_v, data.land_cor(i_v));
		for (int axis = 0; axis < 3; axis++)
			land_3d(i_v, axis) = cal_3d_vtx(bldshps, data.user, data.shape.exp, data.land_cor(i_v), axis);
	}
}

void test_load_mesh(std::vector <DataPoint> &data, Eigen::MatrixXf &bldshps, int idx, std::string path) {
	puts("testing mesh...");
	Eigen::MatrixX3f mesh(G_nVerts, 3);


	FILE *fp;
	fopen_s(&fp, path.c_str(), "w");
	int num = 5;
	fprintf(fp, "%d\n", num);
	cal_mesh(data[0], bldshps, mesh);
	for (int i = 0; i < G_nVerts; i++)
		fprintf(fp, "%.6f %.6f %.6f \n", mesh(i, 0), mesh(i, 1), mesh(i, 2));
	cal_mesh(data[3], bldshps, mesh);
	for (int i = 0; i < G_nVerts; i++)
		fprintf(fp, "%.6f %.6f %.6f \n", mesh(i, 0), mesh(i, 1), mesh(i, 2));
	cal_mesh(data[6], bldshps, mesh);
	for (int i = 0; i < G_nVerts; i++)
		fprintf(fp, "%.6f %.6f %.6f \n", mesh(i, 0), mesh(i, 1), mesh(i, 2)); cal_mesh(data[5], bldshps, mesh);
	cal_mesh(data[8], bldshps, mesh);
	for (int i = 0; i < G_nVerts; i++)
		fprintf(fp, "%.6f %.6f %.6f \n", mesh(i, 0), mesh(i, 1), mesh(i, 2));
	cal_mesh(data[15], bldshps, mesh);
	for (int i = 0; i < G_nVerts; i++)
		fprintf(fp, "%.6f %.6f %.6f \n", mesh(i, 0), mesh(i, 1), mesh(i, 2));

	fclose(fp);
}

void test_load_3d_land(std::vector <DataPoint> &data, Eigen::MatrixXf &bldshps, int idx, std::string path) {
	puts("testing land...");
	Eigen::MatrixX3f land_3d(G_land_num, 3);


	FILE *fp;
	fopen_s(&fp, path.c_str(), "w");
	int num = 5;
	fprintf(fp, "%d\n", num);
	cal_3d_land(data[0], bldshps, land_3d);
	for (int i = 0; i < G_land_num; i++)
		fprintf(fp, "%.6f %.6f %.6f \n", land_3d(i, 0), land_3d(i, 1), land_3d(i, 2));
	cal_3d_land(data[3], bldshps, land_3d);
	for (int i = 0; i < G_land_num; i++)
		fprintf(fp, "%.6f %.6f %.6f \n", land_3d(i, 0), land_3d(i, 1), land_3d(i, 2));
	cal_3d_land(data[6], bldshps, land_3d);
	for (int i = 0; i < G_land_num; i++)
		fprintf(fp, "%.6f %.6f %.6f \n", land_3d(i, 0), land_3d(i, 1), land_3d(i, 2));
	cal_3d_land(data[8], bldshps, land_3d);
	for (int i = 0; i < G_land_num; i++)
		fprintf(fp, "%.6f %.6f %.6f \n", land_3d(i, 0), land_3d(i, 1), land_3d(i, 2));
	cal_3d_land(data[15], bldshps, land_3d);
	for (int i = 0; i < G_land_num; i++)
		fprintf(fp, "%.6f %.6f %.6f \n", land_3d(i, 0), land_3d(i, 1), land_3d(i, 2));

	fclose(fp);
}


bool check_in_ccmc(Eigen::MatrixX2f &points, int i, int j, int k, int l) {
	Eigen::RowVector2f center;
	/*std::cout << points.row(i) << ' ' << points.row(j) << ' ' <<
		points.row(k)<< ' ' << points.row(l) << "\n";*/
	float X1 = points(i, 0), Y1 = points(i, 1);
	float X2 = points(j, 0), Y2 = points(j, 1);
	float X3 = points(k, 0), Y3 = points(k, 1);
	/*std::cout << X1 << ' ' << Y1 << "\n";
	std::cout << X2 << ' ' << Y2 << "\n";
	std::cout << X3 << ' ' << Y3 << "\n";*/
	center(0) = (pow(X2, 2)*Y1 - pow(X3, 2)*Y1 - pow(X1, 2)*Y2 + pow(X3, 2)*Y2 - pow(Y1, 2)*Y2 + pow(Y2, 2)*Y1 + pow(X1, 2)*Y3 - pow(X2, 2)*Y3 + pow(Y1, 2)*Y3 - pow(Y2, 2)*Y3 - pow(Y3, 2)*Y1 + pow(Y3, 2)*Y2) / (2 * (X2*Y1 - X3 * Y1 - X1 * Y2 + X3 * Y2 + X1 * Y3 - X2 * Y3));
	center(1) = -(-pow(X1, 2)*X2 + pow(X2, 2)*X1 + pow(X1, 2)*X3 - pow(X2, 2)*X3 - pow(X3, 2)*X1 + pow(X3, 2)*X2 - pow(Y1, 2)*X2 + pow(Y1, 2)*X3 + pow(Y2, 2)*X1 - pow(Y2, 2)*X3 - pow(Y3, 2)*X1 + pow(X3, 2)*X2) / (2 * (X2*Y1 - X3 * Y1 - X1 * Y2 + X3 * Y2 + X1 * Y3 - X2 * Y3));
	/*std::cout << center << "\n";
	std::cout << (center - points.row(i)).norm() << ' ' << (center - points.row(j)).norm() << ' ' <<
		(center - points.row(k)).norm() << ' ' << (center - points.row(l)).norm() << "\n";
	system("pause");*/
	if ((center - points.row(l)).norm() <= (center - points.row(i)).norm()) return 1;
	return 0;
}
// Draw delaunay triangles
void draw_point(cv::Mat& img, cv::Point2f fp, cv::Scalar color)
{
	circle(img, fp, 2, color, cv::FILLED, cv::LINE_AA, 0);
}


void draw_delaunay(cv::Mat& img, cv::Subdiv2D& subdiv, cv::Scalar delaunay_color)
{

	std::vector<cv::Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	std::vector<cv::Point> pt(3);
	cv::Size size = img.size();
	cv::Rect rect(0, 0, size.width, size.height);

	for (size_t i = 0; i < triangleList.size(); i++)
	{
		cv::Vec6f t = triangleList[i];
		pt[0] = cv::Point(cvRound(t[0]), cvRound(t[1]));
		pt[1] = cv::Point(cvRound(t[2]), cvRound(t[3]));
		pt[2] = cv::Point(cvRound(t[4]), cvRound(t[5]));

		// Draw rectangles completely inside the image.
		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
		{
			line(img, pt[0], pt[1], delaunay_color, 1, cv::LINE_AA, 0);
			line(img, pt[1], pt[2], delaunay_color, 1, cv::LINE_AA, 0);
			line(img, pt[2], pt[0], delaunay_color, 1, cv::LINE_AA, 0);
		}
	}
}
void draw_voronoi(cv::Mat& img, cv::Subdiv2D& subdiv)
{
	std::vector<std::vector<cv::Point2f> > facets;
	std::vector<cv::Point2f> centers;
	subdiv.getVoronoiFacetList(std::vector<int>(), facets, centers);

	std::vector<cv::Point> ifacet;
	std::vector<std::vector<cv::Point> > ifacets(1);

	for (size_t i = 0; i < facets.size(); i++)
	{
		ifacet.resize(facets[i].size());
		for (size_t j = 0; j < facets[i].size(); j++)
			ifacet[j] = facets[i][j];

		cv::Scalar color;
		color[0] = rand() & 255;
		color[1] = rand() & 255;
		color[2] = rand() & 255;
		fillConvexPoly(img, ifacet, color, 8, 0);

		ifacets[0] = ifacet;
		polylines(img, ifacets, true, cv::Scalar(), 1, cv::LINE_AA, 0);
		circle(img, centers[i], 3, cv::Scalar(), cv::FILLED, cv::LINE_AA, 0);
	}
}

void test_del_tri(Eigen::MatrixX2f &points, cv::Mat& img, cv::Subdiv2D subdiv) {
	cv::Scalar delaunay_color(255, 255, 255), points_color(0, 0, 255);
	std::string win_delaunay = "Delaunay Triangulation";
	std::string win_voronoi = "Voronoi Diagram";
	for (int i = 0; i < G_land_num; i++) {
		cv::Mat img_copy = img.clone();
		// Draw delaunay triangles
		draw_delaunay(img_copy, subdiv, delaunay_color);
		imshow(win_delaunay, img_copy);
		cv::waitKey(100);
	}
	// Draw delaunay triangles
	draw_delaunay(img, subdiv, delaunay_color);


	// Draw points
	for (int i = 0; i < G_land_num; i++) {
		draw_point(img, cv::Point2f(points(i, 0), points(i, 1)), points_color);
	}

	// Allocate space for Voronoi Diagram
	cv::Mat img_voronoi = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);

	// Draw Voronoi diagram
	draw_voronoi(img_voronoi, subdiv);

	// Show results.
	cv::imshow(win_delaunay, img);
	cv::imshow(win_voronoi, img_voronoi);
	cv::waitKey(0);

}


void cal_del_tri(
	Eigen::MatrixX2f &points, cv::Mat& img, std::vector<cv::Vec6f> &triangleList) {
	/*std ::cout << points << "\n";
	ans.clear();
	for (int i = 0; i < G_land_num; i++)
		for (int j = i + 1; j < G_land_num; j++)
			for (int k = j + 1; k < G_land_num; k++) {
				bool flag = 0;
				for (int l = 0; l < G_land_num; l++)
					if (check_in_ccmc(points, i, j, k, l)) {
						flag = 1;
						break;
					}
				if (!flag) ans.push_back(std::make_pair(i, std::make_pair(j,k) ));
			}*/

			// Create an instance of Subdiv2D
	cv::Subdiv2D subdiv(cv::Rect(0, 0, img.cols, img.rows));
	for (int i = 0; i < G_land_num; i++)
		subdiv.insert(cv::Point2f(points(i, 0), points(i, 1)));
	test_del_tri(points, img, subdiv);
	subdiv.getTriangleList(triangleList);
}


void cal_del_tri(
	const std::vector<cv::Point2d> &points, cv::Rect &rect,
	std::vector<cv::Vec6f> &triangleList, Eigen::MatrixX3i &tri_idx) {
	double left = 10000, right = -10000, top = 10000, bottom = -10000;
	for (cv::Point2d landmark : points) {
		left = std::min(left, landmark.x);
		right = std::max(right, landmark.x);
		top = std::min(top, landmark.y);
		bottom = std::max(bottom, landmark.y);
	}
	rect = cv::Rect(left - 10, top - 10, right - left + 25, bottom - top + 25);

	cv::Subdiv2D subdiv(rect);
	for (cv::Point2d pt : points)
		subdiv.insert(pt);
	subdiv.getTriangleList(triangleList);

	int flag = 1;
	while (flag)
	{
		flag = 0;
		for (auto it = triangleList.begin(); it != triangleList.end(); it++)
		{
			std::vector<cv::Point2d> pt(3);
			cv::Vec6f t = *it;
			pt[0] = cv::Point2d(t[0], t[1]);
			pt[1] = cv::Point2d(t[2], t[3]);
			pt[2] = cv::Point2d(t[4], t[5]);

			if (!(rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])))
			{
				triangleList.erase(it);
				flag = 1;
				break;
			}
		}
	}




	tri_idx.resize(triangleList.size(), 3);
	std::vector<cv::Point2d> pt(3);
	for (size_t i = 0; i < triangleList.size(); i++)
	{
		cv::Vec6f t = triangleList[i];
		pt[0] = cv::Point2d(t[0], t[1]);
		pt[1] = cv::Point2d(t[2], t[3]);
		pt[2] = cv::Point2d(t[4], t[5]);

		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
		{
			for (int k = 0; k < points.size(); k++)
				for (int j = 0; j < 3; j++) {
					//if (i==0 && k<50)
					//	printf("%d %d %d dis %.10f cha %.10lf pts %.10lf pt%.10lf chay %.10lf\n", 
					//		i, j, k, dis_cv_pt(points[k], pt[j]), points[k].x - pt[j].x, points[k].x, pt[j].x, points[k].y - pt[j].y);

					if (dis_cv_pt(points[k], pt[j]) < EPSILON) tri_idx(i, j) = k;
					//float t = sqrt((points[k].x - pt[j].x)*(points[k].x - pt[j].x)
					//	+ (points[k].y - pt[j].y)*(points[k].y - pt[j].y));

				}
			//printf("--++-- %d %d %d %d\n", i, tri_idx(i, 0), tri_idx(i, 1), tri_idx(i, 2));
		}
	}
}
//void test_del_tri(
//	std::vector < std::vector <cv::Mat_<uchar> > >& imgs,
//	iden *ide, std::vector<std::pair<int, std::pair <int, int> > > &del_tri) {
//	for (int i = 0; i < G_land_num; i++) {
//		cv::circle(
//			imgs[0][0],
//			cv::Point2f(ide[0].land_2d(i, 0), ide[0].land_2d(i, 1)),
//			1, cv::Scalar(244, 244, 244), -1, 8, 0);
//	}
//	for (int j = 0; j < del_tri.size(); j++) {
//		cv::line(imgs[0][0],
//			cv::Point2f(ide[0].land_2d(del_tri[j].first, 0), ide[0].land_2d(del_tri[j].first, 1)),
//			cv::Point2f(ide[0].land_2d(del_tri[j].second.first, 0), ide[0].land_2d(del_tri[j].second.first, 1)),
//			cv::Scalar(244, 244, 244), 1, 8, 0);
//		cv::line(imgs[0][0],
//			cv::Point2f(ide[0].land_2d(del_tri[j].first, 0), ide[0].land_2d(del_tri[j].first, 1)),
//			cv::Point2f(ide[0].land_2d(del_tri[j].second.second, 0), ide[0].land_2d(del_tri[j].second.second, 1)),
//			cv::Scalar(244, 244, 244), 1, 8, 0);
//		cv::line(imgs[0][0],
//			cv::Point2f(ide[0].land_2d(del_tri[j].second.second, 0), ide[0].land_2d(del_tri[j].second.second, 1)),
//			cv::Point2f(ide[0].land_2d(del_tri[j].second.first, 0), ide[0].land_2d(del_tri[j].second.first, 1)),
//			cv::Scalar(244, 244, 244), 1, 8, 0);
//	}
//	cv::imshow("test_image", imgs[0][0]);
//	cv::waitKey(0);
//}


double dis_cv_pt(cv::Point2d pointO, cv::Point2d pointA)
{
	double distance;
	distance = powf((pointO.x - pointA.x), 2) + powf((pointO.y - pointA.y), 2);
	distance = sqrtf(distance);

	return distance;
}

double cal_cv_area(cv::Point2d point0, cv::Point2d point1, cv::Point2d point2) {
	double a = dis_cv_pt(point0, point1), b = dis_cv_pt(point0, point2), c = dis_cv_pt(point1, point2);
	double q = (a + b + c) / 2;
	return sqrtf(q*(q - a)*(q - b)*(q - c));
}

//void update_2d_land(DataPoint &data, Eigen::MatrixXf &bldshps) {
//	data.landmarks.resize(G_land_num);
//	Eigen::RowVector2f T = data.shape.tslt.block(0, 0, 1, 2);
//	Eigen::VectorXf user = data.user;
//	Eigen::VectorXf init_exp = data.shape.exp;
//	data.center.setZero();
//	for (int i_v = 0; i_v < G_land_num; i_v++) {
//		Eigen::Vector3f v;
//		for (int axis = 0; axis < 3; axis++)
//			v(axis) = cal_3d_vtx(bldshps, user, init_exp, data.land_cor(i_v), axis);
//		Eigen::RowVector2f temp = ((data.s) * ((data.shape.rot) * v)).transpose() + T + data.shape.dis.row(i_v);
//		data.landmarks[i_v].x = temp(0); data.landmarks[i_v].y = data.image.rows-temp(1);
//		data.land_2d.row(i_v) = temp;
//		data.center += temp;
//	}
//	data.center /= G_land_num;
//}

//void update_2d_ang_land(DataPoint &data, Eigen::MatrixXf &bldshps) {
//	data.landmarks.resize(G_land_num);
//	Eigen::RowVector2f T = data.shape.tslt.block(0, 0, 1, 2);
//	Eigen::Matrix3f rot = get_r_from_angle_zyx(data.shape.angle);
//	Eigen::VectorXf user = data.user;
//	Eigen::VectorXf init_exp = data.shape.exp;
//	data.center.setZero();
//	for (int i_v = 0; i_v < G_land_num; i_v++) {
//		Eigen::Vector3f v;
//		for (int axis = 0; axis < 3; axis++)
//			v(axis) = cal_3d_vtx(bldshps, user, init_exp, data.land_cor(i_v), axis);
//		Eigen::RowVector2f temp = ((data.s) * ((rot) * v)).transpose() + T + data.shape.dis.row(i_v);
//		data.landmarks[i_v].x = temp(0); data.landmarks[i_v].y = data.image.rows - temp(1);
//		data.land_2d.row(i_v) = temp;
//		data.center += temp;
//	}
//	data.center /= G_land_num;
//}

//void update_2d_land_0ide(DataPoint &data, Eigen::MatrixXf &exp_r_t_all_matrix) {
//	data.landmarks.resize(G_land_num);
//	Eigen::RowVector2f T = data.shape.tslt.block(0, 0, 1, 2);
//	Eigen::VectorXf init_exp = data.shape.exp;
//	data.center.setZero();
//	for (int i_v = 0; i_v < G_land_num; i_v++) {
//		Eigen::Vector3f v;
//		for (int axis = 0; axis < 3; axis++)
//			v(axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix, init_exp, data.land_cor(i_v), axis);
//		Eigen::RowVector2f temp = ((data.s) * ((data.shape.rot) * v)).transpose() + T + data.shape.dis.row(i_v);
//		data.landmarks[i_v].x = temp(0); data.landmarks[i_v].y = data.image.rows - temp(1);
//		data.land_2d.row(i_v) = temp;
//		data.center += temp;
//	}
//	data.center /= G_land_num;
//}

void update_2d_land_ang_0ide(DataPoint &data, Eigen::MatrixXf &exp_r_t_all_matrix) {
	data.landmarks.resize(G_land_num);
#ifdef perspective
	Eigen::Vector3f T = data.shape.tslt;
#endif // perspective

#ifdef normalization
	Eigen::RowVector2f T = data.shape.tslt.block(0, 0, 1, 2);
#endif // normalization

	Eigen::Matrix3f rot = get_r_from_angle_zyx(data.shape.angle);
	Eigen::VectorXf exp = data.shape.exp;
	data.centeroid.setZero();
	for (int i_v = 0; i_v < G_land_num; i_v++) {
		Eigen::Vector3f v;
		for (int axis = 0; axis < 3; axis++)
			v(axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix, exp, data.land_cor(i_v), axis);
#ifdef perspective
		v = rot * v + T;
		data.landmarks[i_v].x=data.land_2d(i_v,0)= v(0)*data.fcs / v(2) + data.center(0)+data.shape.dis(i_v,0);
		data.land_2d(i_v, 1) = v(1)*data.fcs / v(2) + data.center(1) + data.shape.dis(i_v, 1);
		data.landmarks[i_v].y = data.image.rows - data.land_2d(i_v,1);
		data.centeroid += data.land_2d.row(i_v);
#endif // perspective
#ifdef normalization		
		Eigen::RowVector2f temp = ((data.s) * (rot * v)).transpose() + T;
		ans[i_v].x = temp(0); ans[i_v].y = data.image.rows - temp(1);
		Eigen::RowVector2f temp = ((data.s) * (rot * v)).transpose() + T + data.shape.dis.row(i_v);
		data.landmarks[i_v].x = temp(0); data.landmarks[i_v].y = data.image.rows - temp(1);
		data.land_2d.row(i_v) = temp;
		data.centeroid += temp;
		
#endif	
	}
	data.centeroid /= G_land_num;
}

void cal_2d_land_i_ang_0ide(
	std::vector<cv::Point2d> &ans, const Target_type &data, Eigen::MatrixXf &exp_r_t_all_matrix,
	const DataPoint &data_dp) {
	ans.resize(G_land_num);
#ifdef perspective
	Eigen::Vector3f T = data.tslt;
#endif // perspective

#ifdef normalization
	Eigen::RowVector2f T = data.shape.tslt.block(0, 0, 1, 2);
#endif // normalization
	Eigen::Matrix3f rot = get_r_from_angle_zyx(data.angle);
	Eigen::VectorXf exp = data.exp;
	for (int i_v = 0; i_v < G_land_num; i_v++) {
		Eigen::Vector3f v;
		for (int axis = 0; axis < 3; axis++)
			v(axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix, exp, data_dp.land_cor(i_v), axis);

#ifdef perspective
		v = rot * v + T;		
		ans[i_v].x = data_dp.fcs*v(0) / v(2) + data_dp.center(0)+ data.dis(i_v,0);
		ans[i_v].y = data_dp.fcs*v(1) / v(2) + data_dp.center(1) + data.dis(i_v, 1);
		ans[i_v].y = data_dp.image.rows - ans[i_v].y;
#endif // perspective

#ifdef normalization
		Eigen::RowVector2f temp = ((data_dp.s) * (rot * v)).transpose() + T + data.dis.row(i_v);
		ans[i_v].x = temp(0); ans[i_v].y = data_dp.image.rows - temp(1);
#endif // normalization
	}
}

//void cal_2d_land_i(
//	std::vector<cv::Point2d> &ans, const Target_type &data, Eigen::MatrixXf &bldshps, DataPoint &ini_data) {
//	ans.resize(G_land_num);
//	Eigen::RowVector2f T = data.tslt.block(0, 0, 1, 2);
//	Eigen::VectorXf user = ini_data.user;
//	Eigen::VectorXf exp = data.exp;
//	for (int i_v = 0; i_v < G_land_num; i_v++) {
//		Eigen::Vector3f v;
//		for (int axis = 0; axis < 3; axis++)
//			v(axis) = cal_3d_vtx(bldshps, user, exp, ini_data.land_cor(i_v), axis);
//		Eigen::RowVector2f temp = ((ini_data.s) * ((data.rot) * v)).transpose() + T + data.dis.row(i_v);
//		ans[i_v].x = temp(0); ans[i_v].y = ini_data.image.rows - temp(1);
//	}
//}

Target_type shape_difference(const Target_type &s1, const Target_type &s2)
{
	Target_type result;

	result.dis.resize(G_land_num, 2);
	result.dis.array() = s1.dis.array() - s2.dis.array();

	result.exp.resize(G_nShape);
	result.exp.array() = s1.exp.array() - s2.exp.array();

	//result.rot.array() = s1.rot.array() - s2.rot.array();
	result.angle.array() = s1.angle.array() - s2.angle.array();
	result.tslt.array() = s1.tslt.array() - s2.tslt.array();



	return result;
}

Target_type shape_adjustment(Target_type &shape, Target_type &offset)
{
	Target_type result;
	result.land_cor = shape.land_cor;

	result.dis.resize(G_land_num, 2);
	result.dis.array() = shape.dis.array() + offset.dis.array();

	result.exp.resize(G_nShape);
	result.exp.array() = shape.exp.array() + offset.exp.array();

	//	result.rot.array() = shape.rot.array() + offset.rot.array();
	result.angle.array() = shape.angle.array() + offset.angle.array();
	result.tslt.array() = shape.tslt.array() + offset.tslt.array();

	return result;
}

void show_image(cv::Mat img, cv::Rect rect, std::vector<cv::Point2d> landmarks) {
	cv::Mat image = img.clone();
	printf("landmarks number: %d\n", landmarks.size());
	cv::rectangle(image, rect, cv::Scalar(0, 0, 255), 2);
	for (cv::Point2d landmark : landmarks)
	{
		cv::circle(image, landmark, 0.1, cv::Scalar(0, 255, 0), 2);
	}
	//}
	cv::imshow("Alignment result", image);
	cv::waitKey();
	//system("pause");
}
void show_image_0rect(cv::Mat img,std::vector<cv::Point2d> landmarks) {
	cv::Mat image = img.clone();
	for (cv::Point2d landmark : landmarks)
	{
		cv::circle(image, landmark, 0.1, cv::Scalar(250, 250, 220), 2);
	}
	//}
	cv::imshow("dde result", image);
	cv::waitKey();
	//system("pause");
}
void show_image_land_2d(cv::Mat img, Eigen::MatrixX2f &land) {
	cv::Mat image = img.clone();
	for (int i = 0; i < G_land_num; i++)
	{
		cv::circle(image, cv::Point2d(land(i, 0), land(i, 1)), 3, cv::Scalar(10, 2, 2), 2);
	}
	//}
	cv::imshow("dde result", image);
	cv::waitKey();
	//system("pause");
}
void draw_land_img(cv::Mat image, std::vector<cv::Point2d> landmarks) {
	
	for (int i = 0; i < 15; i++)
		cv::circle(image, landmarks[i], 1, cv::Scalar(0, 250, 0), 2);
	for (int i = 1; i < 15; i++)
		cv::line(image, landmarks[i], landmarks[i - 1], cv::Scalar(255, 0, 0));

	for (cv::Point2d landmark : landmarks)
	{
		cv::circle(image, landmark, 0.1, cv::Scalar(0, 250, 0), 2);
	}
	
}

//void save_video(cv::Mat img, std::vector<cv::Point2d> landmarks, cv::VideoWriter &output_video) {
//	cv::Mat image = img.clone();
//	for (cv::Point2d landmark : landmarks)
//	{
//		cv::circle(image, landmark, 0.1, cv::Scalar(250, 250, 220), 2);
//	}
//	//cv::resize(image, image, cv::Size(640, 480 * 3));
//	output_video << image;
//}
void save_video(cv::Mat img, std::vector<cv::Point2d> landmarks, cv::VideoWriter &output_video) {
	cv::Mat image = img.clone();
	for (int i = 0; i < 15; i++)
		cv::circle(image, landmarks[i], 1, cv::Scalar(0, 250, 0), 2);
	for (int i = 1; i < 15; i++)
		cv::line(image, landmarks[i], landmarks[i - 1], cv::Scalar(255, 0, 0));

	for (cv::Point2d landmark : landmarks)
	{
		cv::circle(image, landmark, 0.1, cv::Scalar(0, 250, 0), 2);
	}
	//for (int i = 1; i < 15; i++)
	//	cv::line(image, landmarks[i], landmarks[i - 1], cv::Scalar(255, 0, 0));
	//cv::resize(image, image, cv::Size(640, 480 * 3));
	output_video << image;
}
void save_video(cv::Mat img, std::vector<cv::Point2f> landmarks, cv::VideoWriter &output_video) {
	cv::Mat image = img.clone();
	//for (int i = 0; i < 15; i++)
	//	cv::circle(image, landmarks[i], 1, cv::Scalar(0, 250, 0), 2);
	//for (int i = 1; i < 15; i++)
	//	cv::line(image, landmarks[i], landmarks[i - 1], cv::Scalar(255, 0, 0));

	for (cv::Point2f landmark : landmarks)
	{
		cv::circle(image, landmark, 0.1, cv::Scalar(0, 250, 0), 2);
	}
	//for (int i = 1; i < 15; i++)
	//	cv::line(image, landmarks[i], landmarks[i - 1], cv::Scalar(255, 0, 0));
	//cv::resize(image, image, cv::Size(640, 480 * 3));
	output_video << image;
}
//void cal_2d_land_i_0dis(
//	std::vector<cv::Point2d> &ans, Eigen::MatrixXf &bldshps, DataPoint &data) {
//	ans.resize(G_land_num);
//	Eigen::RowVector2f T = data.shape.tslt.block(0, 0, 1, 2);
//	Eigen::VectorXf user = data.user;
//	Eigen::VectorXf exp = data.shape.exp;
//	for (int i_v = 0; i_v < G_land_num; i_v++) {
//		Eigen::Vector3f v;
//		for (int axis = 0; axis < 3; axis++)
//			v(axis) = cal_3d_vtx(bldshps, user, exp, data.land_cor(i_v), axis);
//		Eigen::RowVector2f temp = ((data.s) * ((data.shape.rot) * v)).transpose() + T;
//		ans[i_v].x = temp(0); ans[i_v].y = data.image.rows - temp(1);
//	}
//}

void cal_2d_land_i_0dis_ang(
	std::vector<cv::Point2d> &ans, Eigen::MatrixXf &bldshps, DataPoint &data) {
	ans.resize(G_land_num);
#ifdef perspective
	Eigen::Vector3f T = data.shape.tslt;
#endif // perspective

#ifdef normalization
	Eigen::RowVector2f T = data.shape.tslt.block(0, 0, 1, 2);
#endif // normalization
	
	Eigen::VectorXf user = data.user;
	Eigen::VectorXf exp = data.shape.exp;
	Eigen::Matrix3f rot = get_r_from_angle_zyx(data.shape.angle);
	for (int i_v = 0; i_v < G_land_num; i_v++) {
		Eigen::Vector3f v;
		for (int axis = 0; axis < 3; axis++)
			v(axis) = cal_3d_vtx(bldshps, user, exp, data.land_cor(i_v), axis);
#ifdef perspective
		v = rot * v + T;
		ans[i_v].x = v(0)*data.fcs / v(2)+data.center(0);
		ans[i_v].y = v(1)*data.fcs / v(2) + data.center(1);
		ans[i_v].y = data.image.rows - ans[i_v].y;
#endif // perspective
#ifdef normalization		
		Eigen::RowVector2f temp = ((data.s) * (rot * v)).transpose() + T;
		ans[i_v].x = temp(0); ans[i_v].y = data.image.rows - temp(1);
#endif	
	}
}

void save_for_debug(DataPoint &temp,std::string name) {
	std::cout << "saving coefficients...file:" << name << "\n";
	FILE *fp;
	fopen_s(&fp, name.c_str(), "wb");
	for (int j = 0; j < G_iden_num; j++)
		fwrite(&temp.user(j), sizeof(float), 1, fp);

	for (int i_v = 0; i_v < G_land_num; i_v++) {
		fwrite(&temp.land_2d(i_v, 0), sizeof(float), 1, fp);
		fwrite(&temp.land_2d(i_v, 1), sizeof(float), 1, fp);
	}

	fwrite(&temp.center(0), sizeof(float), 1, fp);
	fwrite(&temp.center(1), sizeof(float), 1, fp);

	for (int i_shape = 0; i_shape < G_nShape; i_shape++)
		fwrite(&temp.shape.exp(i_shape), sizeof(float), 1, fp);

/*	for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
		fwrite(&temp.shape.rot(i, j), sizeof(float), 1, fp);*/
	Eigen::Matrix3f rot = get_r_from_angle_zyx(temp.shape.angle);

	for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
		fwrite(&rot(i, j), sizeof(float), 1, fp);

#ifdef normalization
	temp.shape.tslt(2) = 0;
#endif // normalization
	for (int i = 0; i < 3; i++) fwrite(&temp.shape.tslt(i), sizeof(float), 1, fp);

	for (int i_v = 0; i_v < G_land_num; i_v++) fwrite(&temp.land_cor(i_v), sizeof(int), 1, fp);

#ifdef perspective
	fwrite(&temp.fcs, sizeof(float), 1, fp);
#endif // perspective
#ifdef normalization
	for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++)
		fwrite(&temp.s(i, j), sizeof(float), 1, fp);
#endif // normalization

	//temp.shape.dis.rowwise() += temp.center;

	std::cout << temp.shape.dis << "\n";
	for (int i_v = 0; i_v < G_land_num; i_v++) {
		fwrite(&temp.shape.dis(i_v, 0), sizeof(float), 1, fp);
		fwrite(&temp.shape.dis(i_v, 1), sizeof(float), 1, fp);
	}

	fclose(fp);

	puts("save successful!");
}

void print_datapoint(DataPoint &data) {

	std::cout << "exp:" << data.shape.exp.transpose() << "\n";
	std::cout << "dis:" << data.shape.dis.transpose() << "\n";
	//std::cout << "rot:" << data.shape.rot << "\n";
	std::cout << "angle:" << data.shape.angle << "\n";
	Eigen::Vector3f temp = data.shape.angle;
	temp.array() *= 180 / pi;
	std::cout << "angle" << temp.transpose() << "\n";
	std::cout << "tslt:" << data.shape.tslt << "\n";
	std::cout << "user:" << data.user.transpose() << "\n";
	std::cout << "center:" << data.center << "\n";
	std::cout << "land:" << data.land_2d.transpose() << "\n";
	std::cout << "landmark:" << data.landmarks << "\n";

	for (int i = 0; i < data.user.rows(); i++)
		printf("%.3f,\n", data.user(i));

}

void print_target(Target_type &data) {

	std::cout << "exp:" << data.exp.transpose() << "\n";
	std::cout << "dis:" << data.dis.transpose() << "\n";
	//std::cout << "rot:" << data.rot << "\n";
	std::cout << "angle:" << data.angle << "\n";
	Eigen::Vector3f temp= data.angle;
	temp.array() *= 180.0 / pi;
	std::cout << "angle" << temp.transpose() << "\n";
	std::cout << "tslt:" << data.tslt << "\n";
}

void save_datapoint(DataPoint &temp, std::string save_name) {
	std::cout << "saving datapoint...file:" << save_name << "\n";
	FILE *fp;
	fopen_s(&fp, save_name.c_str(), "wb");
	for (int j = 0; j < G_iden_num; j++)
		fwrite(&temp.user(j), sizeof(float), 1, fp);

	for (int i_v = 0; i_v < G_land_num; i_v++) {
		fwrite(&temp.land_2d(i_v, 0), sizeof(float), 1, fp);
		fwrite(&temp.land_2d(i_v, 1), sizeof(float), 1, fp);
	}

	fwrite(&temp.center(0), sizeof(float), 1, fp);
	fwrite(&temp.center(1), sizeof(float), 1, fp);

	for (int i_shape = 0; i_shape < G_nShape; i_shape++)
		fwrite(&temp.shape.exp(i_shape), sizeof(float), 1, fp);

	//for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
	//	fwrite(&temp.shape.rot(i, j), sizeof(float), 1, fp);

	Eigen::Matrix3f rot = get_r_from_angle_zyx(temp.shape.angle);
	for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
		fwrite(&rot(i, j), sizeof(float), 1, fp);

	for (int i = 0; i < 3; i++) fwrite(&temp.shape.tslt(i), sizeof(float), 1, fp);

	for (int i_v = 0; i_v < G_land_num; i_v++) fwrite(&temp.land_cor(i_v), sizeof(int), 1, fp);

#ifdef perspective
	fwrite(&temp.fcs, sizeof(float), 1, fp);
#endif // perspective

#ifdef normalization
	for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++)
		fwrite(&temp.s(i, j), sizeof(float), 1, fp);
#endif // normalization

	//temp.shape.dis.rowwise() += temp.center;

	for (int i_v = 0; i_v < G_land_num; i_v++) {
		fwrite(&temp.shape.dis(i_v, 0), sizeof(float), 1, fp);
		fwrite(&temp.shape.dis(i_v, 1), sizeof(float), 1, fp);
	}

	fclose(fp);

	puts("save successful!");


}

void cal_exp_r_t_all_matrix(
	Eigen::MatrixXf &bldshps, DataPoint &data, Eigen::MatrixXf &result) {

	puts("prepare exp_point matrix for bfgs/ceres...");
	result.resize(G_nShape, 3 * G_nVerts);

	for (int i_shape = 0; i_shape < G_nShape; i_shape++)
		for (int i_v = 0; i_v < G_nVerts; i_v++) {
			Eigen::Vector3f V;
			V.setZero();
			for (int j = 0; j < 3; j++)
				for (int i_id = 0; i_id < G_iden_num; i_id++)
					if (i_shape == 0)
						V(j) += data.user(i_id)*bldshps(i_id, i_v * 3 + j);
					else
						V(j) += data.user(i_id)*
						(bldshps(i_id, i_shape*G_nVerts * 3 + i_v * 3 + j) - bldshps(i_id, i_v * 3 + j));

			for (int j = 0; j < 3; j++)
				result(i_shape, i_v * 3 + j) = V(j);
		}

}

void target2vector(Target_type &data, Eigen::VectorXf &ans) {
	ans.resize(G_target_type_size);
	for (int i = 1; i < G_nShape; i++) ans(i - 1) = data.exp(i);
	for (int i = 0; i < G_tslt_num; i++) ans(i + G_nShape - 1) = data.tslt(i);
	//for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) ans(G_nShape + 3 + i * 3 + j) = data.rot(i, j);
	for (int i = 0; i < G_angle_num; i++) ans(G_nShape - 1 + 2 + i) = data.angle(i);
	for (int i = 0; i < G_land_num; i++)for (int j = 0; j < 2; j++) ans(G_nShape - 1 + G_tslt_num + G_angle_num + i * 2 + j) = data.dis(i, j);
}
void vector2target(Eigen::VectorXf &data, Target_type &ans) {
	ans.exp.resize(G_nShape);
	ans.dis.resize(G_land_num, 2);
	for (int i = 1; i < G_nShape; i++) ans.exp(i) = data(i - 1);
	ans.exp(0) = 0;
#ifdef normalization
	ans.tslt(2) = 0;
#endif // normalization
	for (int i = 0; i < G_tslt_num; i++) ans.tslt(i) = data(i + G_nShape - 1);
	
	//for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) ans.rot(i, j) = data(G_nShape + 3 + i * 3 + j);
	for (int i = 0; i < G_angle_num; i++) ans.angle(i) = data(G_nShape - 1 + 2 + i);
	for (int i = 0; i < G_land_num; i++)for (int j = 0; j < 2; j++) ans.dis(i, j) = data(G_nShape - 1 + G_tslt_num + G_angle_num + i * 2 + j);
}


float cal_3d_vtx_0ide(
	Eigen::MatrixXf &exp_r_t_all_matrix, Eigen::VectorXf &exp, int vtx_idx, int axis) {

	//puts("calculating one vertex coordinate...");
	float ans = 0;

	for (int i_shape = 0; i_shape < G_nShape; i_shape++)
		ans += exp(i_shape)*(exp_r_t_all_matrix(i_shape, vtx_idx * 3 + axis));
	return ans;
}

//assume the be could not be more than 90
//Eigen::RowVector3f get_uler_angle(Eigen::Matrix3f R) {
//	Eigen::Vector3f x, y, z;
//	x = R.row(0).transpose();
//	y = R.row(1).transpose();
//	z = R.row(2).transpose();
//	float al, be, gaw;
//	if (fabs(1 - z(2)*z(2)) < 1e-3) {
//		gaw = be = 0;
//		al = acos(x(0));
//		if (y(0) < 0) al = 2 * pi - al;
//	}
//	else {
//
//		be = acos(z(2));
//		al = acos(std::max(std::min(float(1.0), z(1) / sqrt(1 - z(2)*z(2))), float(-1.0)));
//		if (z(0) < 0) al = 2 * pi - al;//according to the sin(al)
//
//		gaw = acos(std::max(std::min(float(1.0), -y(2) / sqrt(1 - z(2)*z(2))), float(-1.0)));
//
//		if (x(2) < 0) gaw = 2 * pi - gaw;//according to the sin(ga)
//	}
//	/*std::cout << R << "\n----------------------\n";
//	printf("%.10f %.10f %.10f %.10f %.10f\n",z(2), al/pi*180, be / pi * 180, ga / pi * 180, gaw / pi * 180);
//	system("pause");*/
//	Eigen::RowVector3f ans;
//	ans << al, be, gaw;
//	return ans;
//}
//
//Eigen::Matrix3f get_r_from_angle(float angle, int axis) {
//	Eigen::Matrix3f ans;
//	ans.setZero();
//	ans(axis, axis) = 1;
//	int idx_x = 0, idx_y = 1;
//	if (axis == 0)
//		idx_x = 1, idx_y = 2;
//	else
//		if (axis == 2)
//			idx_x = 0, idx_y = 1;
//		else
//			idx_x = 0, idx_y = 2;
//	ans(idx_x, idx_x) = cos(angle), ans(idx_x, idx_y) = -sin(angle), ans(idx_y, idx_x) = sin(angle), ans(idx_y, idx_y) = cos(angle);
//	return ans;
//}
//
//Eigen::Matrix3f get_r_from_angle(const Eigen::Vector3f &angle) {
//	Eigen::Matrix3f ans;
//	float Sa = sin(angle(0)), Ca = cos(angle(0)), Sb = sin(angle(1)), 
//		Cb = cos(angle(1)), Sc = sin(angle(2)), Cc = cos(angle(2));
//
//	ans(0, 0) = Ca * Cc - Sa * Cb*Sc;
//	ans(0, 1) = -Sa * Cc - Ca * Cb*Sc;
//	ans(0, 2) = Sb * Sc;
//	ans(1, 0) = Ca * Sc + Sa * Cb*Cc;
//	ans(1, 1) = -Sa * Sc + Ca * Cb*Cc;
//	ans(1, 2) = -Sb * Cc;
//	ans(2, 0) = Sa * Sb;
//	ans(2, 1) = Ca * Sb;
//	ans(2, 2) = Cb;
//	return ans;
//}

Eigen::Vector3f get_uler_angle_zyx(Eigen::Matrix3f R) {
	Eigen::Vector3f x, y, z, t;
	x = R.row(0).transpose();
	y = R.row(1).transpose();
	z = R.row(2).transpose();
	float al, be, ga;
	if (fabs(1 - x(2)*x(2)) < 1e-3) {
		be = asin(x(2));
		al = ga = 0;
		exit(1);
	}
	else {

		be = asin(std::max(std::min(1.0, double(x(2))), -1.0));
		al = asin(std::max(std::min(1.0, double(-x(1) / sqrt(1 - x(2)*x(2)))), -1.0));
		ga = asin(std::max(std::min(1.0, double(-y(2) / sqrt(1 - x(2)*x(2)))), -1.0));

	}
	//std::cout << R << "\n----------------------\n";
	//printf("%.10f %.10f %.10f %.10f\n", x(2), al / pi * 180, be / pi * 180, ga / pi * 180);
	Eigen::Vector3f ans;
	ans << al, be, ga;
	return ans;
	//system("pause");
}

Eigen::Matrix3f get_r_from_angle_zyx(const Eigen::Vector3f &angle) {
	Eigen::Matrix3f ans;
	float Sa = sin(angle(0)), Ca = cos(angle(0)), Sb = sin(angle(1)),
		Cb = cos(angle(1)), Sc = sin(angle(2)), Cc = cos(angle(2));

	ans(0, 0) = Ca * Cb;
	ans(0, 1) = -Sa * Cb;
	ans(0, 2) = Sb;
	ans(1, 0) = Sa * Cc + Ca * Sb*Sc;
	ans(1, 1) = Ca * Cc - Sa * Sb*Sc;
	ans(1, 2) = -Cb * Sc;
	ans(2, 0) = Sa * Sc - Ca * Sb*Cc;
	ans(2, 1) = Ca * Sc + Sa * Sb*Cc;
	ans(2, 2) = Cb * Cc;
	return ans;
}
void update_training_data(DataPoint &data, std::vector<DataPoint> &training_data, Eigen::MatrixXf &exp_r_t_all_matrix) {
	puts("updating train data.....");
	for (int i_train = 0; i_train < training_data.size(); i_train++) {
#ifdef perspective
		training_data[i_train].shape.tslt(2) = (training_data[i_train].shape.tslt(2) / training_data[i_train].fcs)*data.fcs;
		training_data[i_train].fcs = data.fcs;
		training_data[i_train].center = data.center;
#endif // perspective

#ifdef normalization
		training_data[i_train].s = data.s;
#endif // normalization
		update_2d_land_ang_0ide(training_data[i_train], exp_r_t_all_matrix);
	}

}

void rect_scale(cv::Rect &rect, double scale) {
	cv::Point2d center = cv::Point2d(rect.x + rect.width / 2, rect.y + rect.height / 2);
	rect.width *= scale;
	rect.height *= scale;
	rect.x = center.x - rect.width / 2;
	rect.y = center.y - rect.height / 2;
}
void normalize_gauss_face_rect(cv::Mat image, cv::Rect &rect) {
	float ave = 0;
	puts("A");
	int top = rect.y, bottom = rect.y + rect.height;
	int left = rect.x, right = rect.x + rect.width;
	for (int i_row = top; i_row < bottom; i_row++)
		for (int i_col = left; i_col < right; i_col++) {
			//std::cout << i_row << ' ' << i_col << "\n";
			//std::cout << image.rows << ' ' << image.cols << "\n";
			//std::cout << image.channels() << "\n";
			//printf("%d %d %d\n", i_row, i_col, image.at <uchar>(2,2));
			//printf("%d %d %d\n", i_row, i_col, image.at <uchar>(i_row, i_col));

			ave += image.at <uchar>(i_row, i_col);
		}
	ave /= (rect.height)*(rect.width);
	puts("B");
	float sig = 0;
	for (int i_row = top; i_row < bottom; i_row++)
		for (int i_col = left; i_col < right; i_col++)
			sig += (image.at <uchar>(i_row, i_col) - ave)*(image.at <uchar>(i_row, i_col) - ave);
	sig /= (rect.height + 1)*(rect.width + 1);
	sig = sqrt(sig);
	puts("C");
	for (int i_row = top; i_row < bottom; i_row++)
		for (int i_col = left; i_col < right; i_col++)
			image.at <uchar>(i_row, i_col) =
			(uchar)(((image.at <uchar>(i_row, i_col) - ave) / sig)*G_norm_face_rect_sig + G_norm_face_rect_ave);
}

uchar get_batch_feature(int patch_size, cv::Mat img, cv::Point p) {
	float ans = 0;
	for (int i_x = p.x - patch_size; i_x <= p.x + patch_size; i_x++)
		for (int i_y = p.y - patch_size; i_y <= p.y + patch_size; i_y++) {
			cv::Point pos(i_x, i_y);
			float r = dis_cv_pt(pos, p);

			pos.x = std::max(std::min(i_x, img.cols - 1), 0);
			pos.y = std::max(std::min(i_y, img.rows - 1), 0);

			ans += exp(-r * r / 2)*img.at<uchar>(pos);
		}
	ans = std::max(ans, (float)0);
	return (uchar)(ans);
}

void get_batch_feature_dvd(int patch_size, std::vector<cv::Mat> &channel, cv::Point p,Eigen::VectorXf &ans) {
	int C = channel.size();
	ans.resize((2 * patch_size + 1)*(2 * patch_size + 1) * C);
	puts("get_batch_feature_dvd...");
	//std::cout << p << " pp\n";
	for (int i_x = p.x - patch_size; i_x <= p.x + patch_size; i_x++)
		for (int i_y = p.y - patch_size; i_y <= p.y + patch_size; i_y++) {
			cv::Point pos(i_x, i_y);
			pos.x = std::max(std::min(pos.x, channel[0].cols - 1), 0);
			pos.y = std::max(std::min(pos.y, channel[0].rows - 1), 0);

			for (int ch = 0; ch < C; ch++) {				
				int row_idx = ((i_x - (p.x - patch_size))*(2 * patch_size + 1)
					+ i_y - (p.y - patch_size)) * C + ch;
				ans(row_idx)= channel[ch].at<uchar>(pos);
			}			
		}
}

const float sobel_x_wndw[3][3] = { {-0.125,0,0.125} ,{-0.25,0,0.25} ,{-0.125,0,0.125} };
const float sobel_y_wndw[3][3] = { {-0.125,-0.25,-0.125} ,{0,0,0} ,{0.125,0.25,0.125} };
std::pair<float, float> cal_sobel(cv::Mat img, cv::Point p) {
	//std::cout << p << " \n ";
	//system("pause");
	std::pair<float, float> ans;
	ans.first = 0;
	for (int i_x = -1; i_x < 2; i_x++)
		for (int i_y = -1; i_y < 2; i_y++) {
			cv::Point pos(p.x + i_x, p.y + i_y);
			//if (p.x==480) std::cout << p << " + " << pos << "\n";
			pos.x = std::max(std::min(pos.x, img.cols - 1), 0);
			pos.y = std::max(std::min(pos.y, img.rows - 1), 0);
			//if (p.x == 480) std::cout << p << " + " << pos << "\n";
			ans.first += sobel_x_wndw[i_x + 1][i_y + 1] * img.at<uchar>(pos);
		}
	ans.second = 0;
	for (int i_x = -1; i_x < 2; i_x++)
		for (int i_y = -1; i_y < 2; i_y++) {
			cv::Point pos(p.x + i_x, p.y + i_y);
			//if (p.x == 480) std::cout << p << " - " << pos << "\n";
			pos.x = std::max(std::min(pos.x, img.cols - 1), 0);
			pos.y = std::max(std::min(pos.y, img.rows - 1), 0);
			ans.second += sobel_y_wndw[i_x + 1][i_y + 1] * img.at<uchar>(pos);
		}
	//if (p.x == 480) std::cout << ans.first << " " << ans.second << "\n";
	return ans;
}
uchar get_sobel_batch_feature(cv::Mat img, cv::Point p) {
	float ans = 0;
	for (int i_x = p.x - G_pixel_batch_feature_size; i_x <= p.x + G_pixel_batch_feature_size; i_x++)
		for (int i_y = p.y - G_pixel_batch_feature_size; i_y <= p.y + G_pixel_batch_feature_size; i_y++) {
			cv::Point pos(i_x, i_y);
			float r = dis_cv_pt(pos, p);

			pos.x = std::max(std::min(i_x, img.cols), 0);
			pos.y = std::max(std::min(i_y, img.rows), 0);

			std::pair<float, float> sb = cal_sobel(img, cv::Point(i_x, i_y));
			ans += exp(-r * r / 2)*sqrt(sb.first*sb.first + sb.second*sb.second);
		}
	ans = std::max(ans, (float)0);
	return (uchar)(ans);
}
cv::Point lk_get_pos_next(int batch_feature_size, cv::Point X, cv::Mat frame_last, cv::Mat frame_now) {
	
	std::vector<cv::Mat> channel_last;
	cv::split(frame_last, channel_last);

	int C = frame_last.channels();
	assert(C == channel_last.size());

	Eigen::MatrixX2f J, aJ;
	J.resize((2 * batch_feature_size + 1)*(2 * batch_feature_size + 1) * C, 2);
	aJ.resize((2 * batch_feature_size + 1)*(2 * batch_feature_size + 1) * C, 2);
	
	//puts("lk A");
	for (int i_x = X.x - batch_feature_size; i_x <= X.x + batch_feature_size; i_x++)
		for (int i_y = X.y - batch_feature_size; i_y <= X.y + batch_feature_size; i_y++) {
			//printf("i_x_y %d %d\n", i_x, i_y);
			cv::Point pos(i_x, i_y);
			float r = dis_cv_pt(pos, X);
			//puts("lk C");
			pos.x = std::max(std::min(pos.x, frame_last.cols - 1), 0);
			pos.y = std::max(std::min(pos.y, frame_last.rows - 1), 0);
			for (int ch = 0; ch < C; ch++) {
				//printf("ch %d\n", ch);
				std::pair<float, float> temp = cal_sobel(channel_last[ch], pos);
				int row_idx = ((i_x - (X.x - batch_feature_size))*(2 * batch_feature_size + 1)
					+ i_y - (X.y - batch_feature_size)) * C + ch;
				//printf("row_idx %d %d %d %d\n", row_idx, (i_x - ((int)X.x - batch_feature_size)), (i_x - (X.x - batch_feature_size)), (X.x - batch_feature_size));
				//puts("lk D");
				//printf("%d %d %d\n", J.rows(), J.cols(), batch_feature_size);
				J(row_idx, 0) = temp.first;
				//puts("lk E");
				J(row_idx, 1) = temp.second;
				//puts("lk F");
				aJ.row(row_idx) = exp(-r * r / 2)*J.row(row_idx);
			}
			
		}
	//puts("lk B");
	Eigen::Matrix2f H_inv = (J.transpose()*aJ).inverse();
	std::cout << J << "\n";
	std::cout << H_inv << "\n";

	Eigen::VectorXf f_last;
	
	get_batch_feature_dvd(batch_feature_size, channel_last, X,f_last);

	std::vector<cv::Mat> channel_now;
	cv::split(frame_now, channel_now);
	assert(C == channel_now.size());

	cv::Point2d dp_x (0,0);
	for (int ite = 0; ite < G_lk_step; ite++) {
		Eigen::VectorXf f_now;
		get_batch_feature_dvd(batch_feature_size, channel_now, (cv::Point2d)X+ dp_x, f_now);
		Eigen::Vector2f dp = H_inv * aJ.transpose()*(f_now - f_last);
		//std::cout << f_now.transpose() << "<--  f_now \n";
		//std::cout << f_last.transpose() << "<--  f_last \n";
		//std::cout << dp.transpose() << "<--  dp \n";
		
		dp_x.x -= dp(0); dp_x.y -= dp(1);
		std::cout << dp_x << "<--    \n";
		//system("pause");
		if (dp.norm() < EPSILON) break;
	}	
	return X + (cv::Point)dp_x;
}

void shape_err_print(DataPoint &data, DataPoint &data_ref, Eigen::VectorXf &ave_er) {
	std::cout << "shape angle err :" << (data_ref.shape.angle - data.shape.angle).transpose() << "\n";
	std::cout << "shape tslt err :" << (data_ref.shape.tslt - data.shape.tslt).transpose() << "\n";
	std::cout << "shape exp err (norm) : "<< (data_ref.shape.exp - data.shape.exp).norm() <<"\n" << (data_ref.shape.exp - data.shape.exp).transpose() << "\n";
	std::cout << "shape dis err(ave one axis):" << sqrt((data_ref.shape.dis - data.shape.dis).squaredNorm()/G_land_num/2) << "\n";
	

	Eigen::MatrixX2f dis = data_ref.land_2d - data.land_2d;
	const int left_eye_num = 8;
	int idx_lft_eye[left_eye_num] = { 27,28,29,30,66,67,68,65 };
	const int right_eye_num = 8;
	int idx_rt_eye[right_eye_num] = { 31,32,33,34,70,71,72,69 };

	const int lft_bn_num = 6;
	int idx_lft_bn[lft_bn_num] = { 21,22,23,24,25,26 };
	const int rt_bn_num = 6;
	int idx_rt_bn[rt_bn_num] = { 15,16,17,18,19,20 };

	const int ms_be = 46, ms_ed = 64;
	const int ns_be = 35, ns_ed = 46;

	double err_slt = 0, err_eye_left = 0, err_eye_right = 0, err_lft_bn = 0, err_rt_bn = 0, err_ms = 0, err_ns = 0;
	for (int i = 0; i < 15; i++) err_slt += dis.row(i).squaredNorm();

	for (int i = 0; i < left_eye_num; i++) err_eye_left += dis.row(idx_lft_eye[i]).squaredNorm();
	for (int i = 0; i < right_eye_num; i++) err_eye_right += dis.row(idx_rt_eye[i]).squaredNorm();

	for (int i = 0; i < lft_bn_num; i++) err_lft_bn += dis.row(idx_lft_bn[i]).squaredNorm();
	for (int i = 0; i < rt_bn_num; i++) err_rt_bn += dis.row(idx_rt_bn[i]).squaredNorm();

	for (int i = ms_be; i < ms_ed; i++) err_ms += dis.row(i).squaredNorm();
	for (int i = ns_be; i < ns_ed; i++) err_ns += dis.row(i).squaredNorm();

	
	printf("slt err:%.5f\nlft eye:%.5f  rt eye:%.5f\n", err_slt / 15, err_eye_left / left_eye_num, err_eye_right / right_eye_num);
	printf("lft bn:%.5f  rt bn:%.5f\n", err_lft_bn / lft_bn_num, err_rt_bn / rt_bn_num);
	printf("ms:%.5f  ns:%.5f\n", err_ms / (ms_ed - ms_be + 1), err_ns / (ns_ed - ns_be + 1));
	puts("----------------");

	ave_er.resize(5, 1);
	ave_er.setZero();
	for (int i = 0; i < G_angle_num; i++) ave_er(0) += fabs(data_ref.shape.angle(i) - data.shape.angle(i));
	ave_er(0) /= G_angle_num;
	for (int i=0;i<G_tslt_num;i++)ave_er(1)+= fabs(data_ref.shape.tslt(i) - data.shape.tslt(i));
	ave_er(1) /= G_tslt_num;
	for (int i = 0; i < G_nShape; i++) ave_er(2) += fabs(data_ref.shape.exp(i) - data.shape.exp(i));
	ave_er(2) /= G_nShape;
	ave_er(3) = sqrt((data_ref.shape.dis - data.shape.dis).squaredNorm() / G_land_num / 2);
	ave_er(4) = sqrt(dis.squaredNorm() / G_land_num / 2);
}

float show_dis_part(Eigen::MatrixX2f &dis) {
	
	const int left_eye_num = 8;
	int idx_lft_eye[left_eye_num] = { 27,28,29,30,66,67,68,65 };
	const int right_eye_num = 8;
	int idx_rt_eye[right_eye_num] = { 31,32,33,34,70,71,72,69 };

	const int lft_bn_num = 6;
	int idx_lft_bn[lft_bn_num] = { 21,22,23,24,25,26 };
	const int rt_bn_num = 6;
	int idx_rt_bn[rt_bn_num] = { 15,16,17,18,19,20 };

	const int ms_be = 46, ms_ed = 64;
	const int ns_be = 35, ns_ed = 46;

	double err_slt = 0, err_eye_left = 0, err_eye_right = 0, err_lft_bn = 0, err_rt_bn = 0, err_ms = 0, err_ns = 0;
	for (int i = 0; i < 15; i++) err_slt += dis.row(i).squaredNorm();

	for (int i = 0; i < left_eye_num; i++) err_eye_left += dis.row(idx_lft_eye[i]).squaredNorm();
	for (int i = 0; i < right_eye_num; i++) err_eye_right += dis.row(idx_rt_eye[i]).squaredNorm();

	for (int i = 0; i < lft_bn_num; i++) err_lft_bn += dis.row(idx_lft_bn[i]).squaredNorm();
	for (int i = 0; i < rt_bn_num; i++) err_rt_bn += dis.row(idx_rt_bn[i]).squaredNorm();

	for (int i = ms_be; i < ms_ed; i++) err_ms += dis.row(i).squaredNorm();
	for (int i = ns_be; i < ns_ed; i++) err_ns += dis.row(i).squaredNorm();

	printf("slt err:%.5f\nlft eye:%.5f  rt eye:%.5f\n", err_slt / 15, err_eye_left / left_eye_num, err_eye_right / right_eye_num);
	printf("lft bn:%.5f  rt bn:%.5f\n", err_lft_bn / lft_bn_num, err_rt_bn / rt_bn_num);
	printf("ms:%.5f  ns:%.5f\n", err_ms / (ms_ed - ms_be + 1), err_ns / (ns_ed - ns_be + 1));


	for (int i = 0; i < G_land_num; i++)
		printf("%d %.5f %.5f\n", i, dis(i, 0), dis(i, 1));

	printf("tot average error:%.5f\n", sqrt(dis.squaredNorm() / G_land_num / 2));
	return sqrt(dis.squaredNorm() / G_land_num / 2);
}