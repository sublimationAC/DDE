
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

void test_del_tri(Eigen::MatrixX2f &points, cv::Mat_<uchar>& img, cv::Subdiv2D subdiv) {
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
	Eigen::MatrixX2f points, cv::Mat_<uchar>& img, std::vector<cv::Vec6f> &triangleList) {
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


void test_del_tri(
	std::vector < std::vector <cv::Mat_<uchar> > >& imgs,
	iden *ide, std::vector<std::pair<int, std::pair <int, int> > > &del_tri) {
	for (int i = 0; i < G_land_num; i++) {
		cv::circle(
			imgs[0][0],
			cv::Point2f(ide[0].land_2d(i, 0), ide[0].land_2d(i, 1)),
			1, cv::Scalar(244, 244, 244), -1, 8, 0);
	}
	for (int j = 0; j < del_tri.size(); j++) {
		cv::line(imgs[0][0],
			cv::Point2f(ide[0].land_2d(del_tri[j].first, 0), ide[0].land_2d(del_tri[j].first, 1)),
			cv::Point2f(ide[0].land_2d(del_tri[j].second.first, 0), ide[0].land_2d(del_tri[j].second.first, 1)),
			cv::Scalar(244, 244, 244), 1, 8, 0);
		cv::line(imgs[0][0],
			cv::Point2f(ide[0].land_2d(del_tri[j].first, 0), ide[0].land_2d(del_tri[j].first, 1)),
			cv::Point2f(ide[0].land_2d(del_tri[j].second.second, 0), ide[0].land_2d(del_tri[j].second.second, 1)),
			cv::Scalar(244, 244, 244), 1, 8, 0);
		cv::line(imgs[0][0],
			cv::Point2f(ide[0].land_2d(del_tri[j].second.second, 0), ide[0].land_2d(del_tri[j].second.second, 1)),
			cv::Point2f(ide[0].land_2d(del_tri[j].second.first, 0), ide[0].land_2d(del_tri[j].second.first, 1)),
			cv::Scalar(244, 244, 244), 1, 8, 0);
	}
	cv::imshow("test_image", imgs[0][0]);
	cv::waitKey(0);
}
