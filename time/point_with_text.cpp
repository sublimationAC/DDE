//#include "opencv2/imgproc.hpp"
//#include "opencv2/highgui.hpp"
//#include <iostream>
//#include <ctype.h>
//using namespace cv;
//using namespace std;
//const int width = 1640;
//const int height = 980;
//const int nVerts = 11510;
//double x[nVerts], y[nVerts], z[nVerts];
//int ini_idx[nVerts];
//struct Point_my
//{
//	cv::Point2d pt2d;
//	int v_idx;
//};
//
//std::vector <Point_my> slt_line[1000];
//
//int cmp_y(Point_my x, Point_my y) {
//	return ((x.pt2d.y < y.pt2d.y) || (((x.pt2d.y == y.pt2d.y)) && (x.pt2d.x < y.pt2d.x)));
//}
//int cmp_x(Point_my x, Point_my y) {
//	return (x.pt2d.x < y.pt2d.x);
//}
//
//int cmp_tp_btm(std::vector <Point_my> x, std::vector <Point_my> y) {
//	return (x[0].pt2d.y < y[0].pt2d.y);
//}
//
//bool bj[nVerts];
//void get_line(std::vector <Point_my> *slt_line, int &cnt_line, std::vector <Point_my> &pts) {
//
//	std::sort(pts.begin(), pts.end(), cmp_x);
//
//	memset(bj, 0, sizeof(bj));
//	for (int now = 0; now < pts.size(); now++) {
//		while (bj[now] && now < pts.size()) now++;
//		if (now == pts.size()) break;
//		printf("%d %d\n", now, cnt_line);
//		slt_line[cnt_line].clear();
//
//		slt_line[cnt_line].push_back(pts[now]); bj[now] = 1;
//
//		for (int x = now;;) {
//			int nx = -1, mi = 9999;
//			for (int i = x + 1; (i < pts.size()) && (pts[i].pt2d.x <= pts[x].pt2d.x + 50); i++)
//				if ((bj[i] == 0) && (fabs(pts[i].pt2d.y - pts[x].pt2d.y) < mi)) mi = fabs(pts[i].pt2d.y - pts[x].pt2d.y), nx = i;
//			if (nx == -1 || mi > 8) break;
//			slt_line[cnt_line].push_back(pts[nx]); bj[nx] = 1;
//			x = nx;
//		}
//		//system("pause");
//		cnt_line++;
//	}
//}
//
//
//int main()
//{
//	FILE *fp;
//	fopen_s(&fp, "D:\\sydney\\first\\data\\Tester_ (18)\\TrainingPose/pose_0.obj", "r");
//
//	double mi_x = 99999, ma_x = -999999, mi_y = 999999, ma_y = -999999;
//	int i = 0,tot_v_num=0;
//	for (int j=0; j < nVerts; j++) {
//		fscanf_s(fp, "v %lf%lf%lf\n", &x[i], &y[i], &z[i]);
//		//x[i] > 0.7 || x[i] < 0.5 || 
//		if (x[i] < 0.58 || y[i] > -0 || y[i] <-1.5 || z[i] < -0.15) {
//			continue;
//		}
//		ini_idx[i] = j;
//		mi_x = min(mi_x, x[i]), mi_y = min(mi_y, y[i]);
//		ma_x = max(ma_x, x[i]), ma_y = max(ma_y, y[i]);
//		i++;
//		//printf("%d %.5f %.5f %.5f\n", i, x[i], y[i], z[i]);
//		//system("pause");
//	}
//	fclose(fp);
//	tot_v_num = i;
//	double rg_x = ma_x - mi_x, rg_y = ma_y - mi_y;
//	mi_x -= rg_x * 0.1; ma_x += rg_x * 0.1;
//	mi_y -= rg_y * 0.1; ma_y += rg_y * 0.1;
//	double mid_x = (mi_x + ma_x) / 2;// , mid_y = (mi_y + ma_y) / 2;
//	std::vector<Point_my> pts;
//	pts.clear();
//	for (int i = 0; i < tot_v_num; i++)
//	{
//		pts.push_back(Point_my{
//			Point2d(
//				(x[i] - mi_x) / (ma_x - mi_x)*(width - 1),
//				height - 1 - (y[i] - mi_y) / (ma_y - mi_y)*(height - 1)),
//			ini_idx[i] });
//	}
//
//
//	Mat img(height, width, CV_32FC3, cv::Scalar(0, 0, 0));
//
//	std::sort(pts.begin(), pts.end(), cmp_y);
//	
//	for (int i = 0; i < pts.size(); i++) {
//		//cout << "i: " << pts[i].pt2d << "\n";
//		//img.at<Vec3f>(pts[i].pt2d) = Vec3f(255, 255, 255);
//		cv::circle(img, pts[i].pt2d, 3, cv::Scalar(255, 255, 255));
//		cv::putText(img, to_string(pts[i].v_idx), pts[i].pt2d, cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0));
//	}
//
//	cv::imshow("result", img);
//	cv::waitKey(0);
//	cv::imwrite("jaw.jpg", img);
//
//	return 0;
//}