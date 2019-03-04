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
//
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
//			for (int i = x + 1; (i < pts.size()) && (pts[i].pt2d.x <= pts[x].pt2d.x + 5); i++)
//				if ((bj[i]==0)&&(fabs(pts[i].pt2d.y - pts[x].pt2d.y) < mi)) mi = fabs(pts[i].pt2d.y - pts[x].pt2d.y), nx = i;
//			if (nx == -1 || mi>2) break;
//			slt_line[cnt_line].push_back(pts[nx]); bj[nx] = 1;
//			x = nx;
//		}
//		//system("pause");
//		cnt_line++;
//	}
//}
//
//void get_y_nearest(int x, double range, int &nx, int &mi, std::vector <Point_my> &pts) {
//	for (int i = x + 1; (i < pts.size()) && (pts[i].pt2d.x <= pts[x].pt2d.x + range); i++)
//		//if ((bj[i] == 0) && pts[i].pt2d.y >= pts[x].pt2d.y- range/3 && (fabs(pts[i].pt2d.y - pts[x].pt2d.y) < mi))
//		if ((bj[i] == 0) && pts[i].pt2d.y >= pts[x].pt2d.y-5 && (fabs(pts[i].pt2d.y - pts[x].pt2d.y) < mi))
//			mi = fabs(pts[i].pt2d.y - pts[x].pt2d.y), nx = i;
//}
//void cal_line_idx_based_on_first_nsty(std::vector <Point_my> *slt_line, int &cnt_line, std::vector <Point_my> &pts) {
//
//	std::sort(pts.begin(), pts.end(), cmp_x);
//	FILE *fp;
//	fopen_s(&fp, "start_pt_idx.txt", "r");
//	memset(bj, 0, sizeof(bj));
//	int st;
//	while (fscanf_s(fp,"%d",&st)>0){
//		int now = -1;
//		for (int i=0;i<pts.size();i++)
//			if (st == pts[i].v_idx) {
//				now = i;
//				break;
//			}
//		printf("cal_line_idx_based_on_first %d %d\n", now, cnt_line);
//		if (now == -1) {
//			puts("NO....shit...");
//			printf("%d %d\n", st,now);
//			system("pause");
//			exit(-2);
//		}
//		slt_line[cnt_line].clear();
//
//		slt_line[cnt_line].push_back(pts[now]); bj[now] = 1;
//
//		for (int x = now;;) {
//
//
//			int nx = -1, mi = 9999;
//			get_y_nearest(x, 40, nx, mi, pts);
//			//if (nx == -1 || mi > 8) get_y_nearest(x, 10, nx, mi, pts);
//			//if (nx == -1 || mi > 8) get_y_nearest(x, 20, nx, mi, pts);
//			if (nx == -1 || mi > 8) get_y_nearest(x, 30, nx, mi, pts);
//
//			if (nx == -1 || mi > 8) break;
//			slt_line[cnt_line].push_back(pts[nx]); bj[nx] = 1;
//			x = nx;
//		}
//		//system("pause");
//		cnt_line++;
//
//	}
//	fclose(fp);
//}
//void cal_line_idx_based_on_first_nstdis(std::vector <Point_my> *slt_line, int &cnt_line, std::vector <Point_my> &pts) {
//
//	std::sort(pts.begin(), pts.end(), cmp_x);
//	FILE *fp;
//	fopen_s(&fp, "start_pt_idx.txt", "r");
//	memset(bj, 0, sizeof(bj));
//	int st;
//	while (fscanf_s(fp, "%d", &st) > 0) {
//		int now = -1;
//		for (int i = 0; i < pts.size(); i++)
//			if (st == pts[i].v_idx) {
//				now = i;
//				break;
//			}
//		printf("cal_line_idx_based_on_first %d %d\n", now, cnt_line);
//		if (now == -1) {
//			puts("NO....shit...");
//			printf("%d %d\n", st, now);
//			system("pause");
//			exit(-2);
//		}
//		slt_line[cnt_line].clear();
//
//		slt_line[cnt_line].push_back(pts[now]); bj[now] = 1;
//
//		for (int x = now;;) {
//
//
//			int nx = -1, mi = 9999;
//			get_y_nearest(x, 40, nx, mi, pts);
//			//if (nx == -1 || mi > 8) get_y_nearest(x, 10, nx, mi, pts);
//			//if (nx == -1 || mi > 8) get_y_nearest(x, 20, nx, mi, pts);
//			if (nx == -1 || mi > 8) get_y_nearest(x, 30, nx, mi, pts);
//
//			if (nx == -1 || mi > 8) break;
//			slt_line[cnt_line].push_back(pts[nx]); bj[nx] = 1;
//			x = nx;
//		}
//		//system("pause");
//		cnt_line++;
//
//	}
//	fclose(fp);
//}
//
//int main()
//{
//	FILE *fp;
//	fopen_s(&fp,"D:\\sydney\\first\\data\\Tester_ (18)\\TrainingPose/pose_0.obj", "r");
//
//	double mi_x = 99999, ma_x = -999999, mi_y = 999999, ma_y = -999999;
//	for (int i = 0; i < nVerts; i++) {
//		fscanf_s(fp, "v %lf%lf%lf\n", &x[i], &y[i], &z[i]);
//		mi_x = min(mi_x, x[i]), mi_y = min(mi_y, y[i]);
//		ma_x = max(ma_x, x[i]), ma_y = max(ma_y, y[i]);
//		//printf("%d %.5f %.5f %.5f\n", i, x[i], y[i], z[i]);
//		//system("pause");
//	}
//	fclose(fp);
//	mi_x -= 0.1; ma_x += 0.1;
//	mi_y -= 0.1; ma_y += 0.1;
//	double mid_x = (mi_x + ma_x) / 2;// , mid_y = (mi_y + ma_y) / 2;
//	std::vector<Point_my> pts_lft,pts_rht;
//	pts_lft.clear(); pts_rht.clear();
//	for (int i=0;i<nVerts;i++)
//		if (z[i] > -0.1) {
//			if (x[i] < mid_x - 0.1 && x[i]<-0.19)
//				pts_lft.push_back(Point_my{
//					Point2d(
//						(x[i] - mi_x) / (ma_x - mi_x)*(width - 1),
//						height - 1-(y[i] - mi_y) / (ma_y - mi_y)*(height - 1)),
//					i });
//			if (x[i] > mid_x + 0.1)
//				pts_rht.push_back(Point_my{
//					Point2d(
//						(x[i] - mi_x) / (ma_x - mi_x)*(width - 1),
//						height - 1 - (y[i] - mi_y) / (ma_y - mi_y)*(height - 1)),
//					i });
//		}
//
//
//	Mat img(height, width, CV_32FC3,cv::Scalar(0,0,0));
//	
//	std::sort(pts_lft.begin(), pts_lft.end(), cmp_y);
//	std::sort(pts_rht.begin(), pts_rht.end(), cmp_y);
//	for (int i = 0; i < pts_lft.size(); i++) {
//		//cout << "i: " << pts[i].pt2d << "\n";
//		//img.at<Vec3f>(pts[i].pt2d) = Vec3f(255, 255, 255);
//		cv::circle(img, pts_lft[i].pt2d, 3, cv::Scalar(255, 0, 0));
//		//cv::putText(img, to_string(pts_lft[i].v_idx), pts_lft[i].pt2d, cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0));
//	}
//	for (int i = 0; i < pts_rht.size(); i++) {
//		//cout << "i: " << pts[i].pt2d << "\n";
//		//img.at<Vec3f>(pts[i].pt2d) = Vec3f(255, 255, 255);
//		cv::circle(img, pts_rht[i].pt2d, 3, cv::Scalar(0, 255, 0));
//	}
//
//	int cnt_line = 0;
//	
//	//get_line(slt_line, cnt_line, pts_lft);
//	//get_line(slt_line, cnt_line, pts_rht);
//	cal_line_idx_based_on_first_nsty(slt_line, cnt_line, pts_lft);
//
//	for (int i = 0; i < cnt_line; i++) {
//		printf("%d cnt_line %d\n", i, cnt_line);
//		if (slt_line[i].size()>1)
//			for (int j = 0; j < slt_line[i].size() - 1; j++)
//				cv::line(img, slt_line[i][j].pt2d, slt_line[i][j + 1].pt2d, cv::Scalar(0, 0, 255),2);
//	}
//	//GaussianBlur(img, img, Size(3, 3), 0, 0);
//	//blur(img, img, Size(10, 10));
//	imshow("test.jpg", img);
//	cv::waitKey(0);
//
//
//
//	fopen_s(&fp, "slt_line_left_scratch.txt", "w");
//
//	std::sort(slt_line, slt_line + cnt_line, cmp_tp_btm);
//	fprintf(fp, "%d\n", cnt_line);
//	for (int i = 0; i < cnt_line; i++) {
//		fprintf(fp, "%d %d ",i, slt_line[i].size());
//		for (int j = 0; j < slt_line[i].size(); j++)
//			fprintf(fp, "%d ", slt_line[i][j].v_idx);
//		fprintf(fp,"\n");
//	}
//	fclose(fp);
//
//
//
//	return 0;
//}