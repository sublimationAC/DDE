#include "utils_dde.hpp"


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
void recal_dis(DataPoint &data, Eigen::MatrixXf &bldshps) {
	//puts("calculating displacement...");
	Eigen::MatrixX2f land(G_land_num, 2);
	Eigen::RowVector2f T = data.shape.tslt.block(0, 0, 1, 2);
	for (int i_v = 0; i_v < G_land_num; i_v++) {
		Eigen::Vector3f v;
		for (int axis = 0; axis < 3; axis++)
			v(axis) = cal_3d_vtx(bldshps, data.user, data.shape.exp, data.land_cor(i_v), axis);
		land.row(i_v) = ((data.s) * ((data.shape.rot) * v)).transpose() + T;
	}

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

void test_load_mesh(std::vector <DataPoint> &data, Eigen::MatrixXf &bldshps, int idx, std::string &path) {
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

void test_load_3d_land(std::vector <DataPoint> &data, Eigen::MatrixXf &bldshps, int idx, std::string &path) {
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
	circle(img, fp, 2, color, CV_FILLED, CV_AA, 0);
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
			line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
			line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
			line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
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
		polylines(img, ifacets, true, cv::Scalar(), 1, CV_AA, 0);
		circle(img, centers[i], 3, cv::Scalar(), CV_FILLED, CV_AA, 0);
	}
}

void test_del_tri(Eigen::MatrixX2f &points, cv::Mat& img, cv::Subdiv2D &subdiv) {
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
	std::vector<cv::Vec6f> &triangleList,Eigen::MatrixX3i &tri_idx) {
	double left = 10000, right = -10000, top = 10000, bottom = -10000;
	for (cv::Point2d landmark : points) {
		left = std::min(left, landmark.x);
		right = std::max(right, landmark.x);
		top = std::min(top, landmark.y);
		bottom = std::max(bottom, landmark.y);
	}
	rect = cv::Rect(left - 10, top - 10, right - left + 25, bottom - top + 25);

	cv::Subdiv2D subdiv(rect);
	for (cv::Point2d pt: points)
		subdiv.insert(pt);
	subdiv.getTriangleList(triangleList);

	int flag = 1;
	while (flag)
	{
		flag = 0;
		for (auto it=triangleList.begin();it!=triangleList.end();it++)
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
					if (dis_cv_pt(points[k], pt[j]) < EPSILON) tri_idx(i, j) = k;
					float t = sqrt((points[k].x - pt[j].x)*(points[k].x - pt[j].x)
						+ (points[k].y - pt[j].y)*(points[k].y - pt[j].y));

				}
			printf("--++-- %d %d %d %d\n", i, tri_idx(i, 0), tri_idx(i, 1), tri_idx(i, 2));
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

void cal_init_2d_land_i(std::vector<cv::Point2d> &ans, const DataPoint &data, Eigen::MatrixXf &bldshps) {
	ans.resize(G_land_num);
	Eigen::RowVector2f T = data.init_shape.tslt.block(0, 0, 1, 2);
	Eigen::VectorXf user = data.user;
	Eigen::VectorXf init_exp = data.init_shape.exp;
	for (int i_v = 0; i_v < G_land_num; i_v++) {
		Eigen::Vector3f v;
		for (int axis = 0; axis < 3; axis++)
			v(axis) = cal_3d_vtx(bldshps, user, init_exp, data.land_cor(i_v), axis);
		Eigen::RowVector2f temp = ((data.s) * ((data.init_shape.rot) * v)).transpose() + T + data.init_shape.dis.row(i_v);
		ans[i_v].x = temp(0); ans[i_v].y = data.image.rows-temp(1);
	}

}

Target_type shape_difference(const Target_type &s1, const Target_type &s2)
{
	Target_type result;

	result.dis.resize(G_land_num, 2);
	result.dis.array() = s1.dis.array() - s2.dis.array();

	result.exp.resize(G_nShape);
	result.exp.array() = s1.exp.array() - s2.exp.array();

	result.rot.array() = s1.rot.array() - s2.rot.array();
	result.tslt.array() = s1.tslt.array() - s2.tslt.array();

	return result;
}

Target_type shape_adjustment(Target_type &shape, Target_type &offset)
{
	Target_type result;

	result.dis.resize(G_land_num, 2);
	result.dis.array() = shape.dis.array() + offset.dis.array();

	result.exp.resize(G_nShape);
	result.exp.array() = shape.exp.array() + offset.exp.array();

	result.rot.array() = shape.rot.array() + offset.rot.array();
	result.tslt.array() = shape.tslt.array() + offset.tslt.array();

	return result;
}

std::vector<cv::Point2d> mean_shape(std::vector<std::vector<cv::Point2d>> shapes,
	const TrainingParameters &tp)
{
	std::vector<cv::Point2d> mean_shape = shapes[0];
	for (cv::Point2d & p : mean_shape)
		p.x = p.y = 0;

	for (const std::vector<cv::Point2d> & shape : shapes)
		for (int j = 0; j < mean_shape.size(); ++j)
		{
			mean_shape[j].x += shape[j].x;
			mean_shape[j].y += shape[j].y;
		}

	for (cv::Point2d & p : mean_shape)
		p *= 1.0 / shapes.size();


	return mean_shape;
}

void print_datapoint(DataPoint &data) {

	std::cout << "exp:" << data.shape.exp.transpose() << "\n";
	std::cout << "dis:" << data.shape.dis.transpose() << "\n";
	std::cout << "rot:" << data.shape.rot << "\n";
	std::cout << "tslt:" << data.shape.tslt << "\n";
	std::cout << "init_exp:" << data.init_shape.exp.transpose() << "\n";
	std::cout << "init_dis:" << data.init_shape.dis.transpose() << "\n";
	std::cout << "init_rot:" << data.init_shape.rot << "\n";
	std::cout << "init_tslt:" << data.init_shape.tslt << "\n";
	std::cout << "user:" << data.user.transpose() << "\n";
	std::cout << "center:" << data.center << "\n";
	std::cout << "land:" << data.land_2d.transpose() << "\n";
	std::cout << "landmark:" << data.landmarks << "\n";
}

void print_target(Target_type &data) {

	std::cout << "exp:" << data.exp.transpose() << "\n";
	std::cout << "dis:" << data.dis.transpose() << "\n";
	std::cout << "rot:" << data.rot << "\n";
	std::cout << "tslt:" << data.tslt << "\n";
}