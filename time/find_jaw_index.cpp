#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <ctype.h>
using namespace cv;
using namespace std;
const int width = 1640;
const int height = 980;
const int nVerts = 11510;
double x[nVerts], y[nVerts], z[nVerts];

struct Point_my
{
	cv::Point2d pt2d;
	int v_idx;
};

std::vector <Point_my> slt_line[1000];

int cmp_y(Point_my x, Point_my y) {
	return ((x.pt2d.y < y.pt2d.y) || (((x.pt2d.y == y.pt2d.y)) && (x.pt2d.x < y.pt2d.x)));
}
int cmp_x(Point_my x, Point_my y) {
	return (x.pt2d.x < y.pt2d.x);
}

int cmp_tp_btm(std::vector <Point_my> x, std::vector <Point_my> y) {
	return (x[0].pt2d.y < y[0].pt2d.y);
}

bool bj[nVerts];
void get_line(std::vector <Point_my> *slt_line, int &cnt_line, std::vector <Point_my> &pts) {

	std::sort(pts.begin(), pts.end(), cmp_x);

	memset(bj, 0, sizeof(bj));
	for (int now = 0; now < pts.size(); now++) {
		while (bj[now] && now < pts.size()) now++;
		if (now == pts.size()) break;
		printf("%d %d\n", now, cnt_line);
		slt_line[cnt_line].clear();

		slt_line[cnt_line].push_back(pts[now]); bj[now] = 1;

		for (int x = now;;) {
			int nx = -1, mi = 9999;
			for (int i = x + 1; (i < pts.size()) && (pts[i].pt2d.x <= pts[x].pt2d.x + 5); i++)
				if ((bj[i]==0)&&(fabs(pts[i].pt2d.y - pts[x].pt2d.y) < mi)) mi = fabs(pts[i].pt2d.y - pts[x].pt2d.y), nx = i;
			if (nx == -1 || mi>2) break;
			slt_line[cnt_line].push_back(pts[nx]); bj[nx] = 1;
			x = nx;
		}
		//system("pause");
		cnt_line++;
	}
}

void get_y_nearest(int x, double range, int &nx, int &mi, std::vector <Point_my> &pts) {
	for (int i = x + 1; (i < pts.size()) && (pts[i].pt2d.x <= pts[x].pt2d.x + range); i++)
		//if ((bj[i] == 0) && pts[i].pt2d.y >= pts[x].pt2d.y- range/3 && (fabs(pts[i].pt2d.y - pts[x].pt2d.y) < mi))
		if ((bj[i] == 0) && pts[i].pt2d.y >= pts[x].pt2d.y-5 && (fabs(pts[i].pt2d.y - pts[x].pt2d.y) < mi))
			mi = fabs(pts[i].pt2d.y - pts[x].pt2d.y), nx = i;
}
double dis_cv_pt(cv::Point2d pointO, cv::Point2d pointA)
{
	double distance;
	distance = powf((pointO.x - pointA.x), 2) + powf((pointO.y - pointA.y), 2);
	distance = sqrtf(distance);

	return distance;
}
void get_dis_nrst(int x, int num, int &nx, double &mi, std::vector <Point_my> &pts) {

	for (int i = 0; i < pts.size(); i++) {
		if (bj[i]) continue;
		//if (pts[i].pt2d.y > pts[x].pt2d.y) continue;
		if (dis_cv_pt(pts[i].pt2d, pts[x].pt2d) > 50) continue;
		if (num > 10 && pts[i].pt2d.x > pts[x].pt2d.x) continue;
		//if ((bj[i] == 0) && pts[i].pt2d.y >= pts[x].pt2d.y- range/3 && (fabs(pts[i].pt2d.y - pts[x].pt2d.y) < mi))
		if (dis_cv_pt(pts[i].pt2d, pts[x].pt2d) < mi)
			mi = dis_cv_pt(pts[i].pt2d, pts[x].pt2d), nx = i;
	}
}
void cal_line_idx_based_on_first_nstdis(std::vector <Point_my> *slt_line, int &cnt_line, std::vector <Point_my> &pts) {

	std::sort(pts.begin(), pts.end(), cmp_x);
	FILE *fp;
	fopen_s(&fp, "start_pt_idx_jaw.txt", "r");
	memset(bj, 0, sizeof(bj));
	int st;
	while (fscanf_s(fp, "%d", &st) > 0) {
		int now = -1;
		for (int i = 0; i < pts.size(); i++)
			if (st == pts[i].v_idx) {
				now = i;
				break;
			}
		printf("cal_line_idx_based_on_first_nstdis %d %d %d\n",st, now, cnt_line);
		if (now == -1) {
			puts("NO....shit...");
			printf("%d %d\n", st, now);
			system("pause");
			exit(-2);
		}
		slt_line[cnt_line].clear();

		slt_line[cnt_line].push_back(pts[now]); bj[now] = 1;

		for (int x = now;;) {


			int nx = -1;
			double mi = 99999999;
			get_dis_nrst(x, slt_line[cnt_line].size(), nx, mi, pts);

			//if (nx == -1 || mi > 8) get_y_nearest(x, 10, nx, mi, pts);
			//if (nx == -1 || mi > 8) get_y_nearest(x, 20, nx, mi, pts);
		

			if (nx == -1 || slt_line[cnt_line].size()>15) break;
			slt_line[cnt_line].push_back(pts[nx]); bj[nx] = 1;
			x = nx;
		}
		//system("pause");
		cnt_line++;

	}
	fclose(fp);
}
int ini_idx[nVerts];
int main()
{
	FILE *fp;
	fopen_s(&fp, "D:\\sydney\\first\\data\\Tester_ (18)\\TrainingPose/pose_0.obj", "r");

	double mi_x = 99999, ma_x = -999999, mi_y = 999999, ma_y = -999999;
	int i = 0, tot_v_num = 0;
	for (int j = 0; j < nVerts; j++) {
		fscanf_s(fp, "v %lf%lf%lf\n", &x[i], &y[i], &z[i]);
		//x[i] > 0.7 || x[i] < 0.5 ||
		//if (x[i] < 0.1 || y[i] > -0.17 || y[i] < -1.5 || z[i] < -0.15) {
		if ( y[i] < -1.5 || z[i] < -0.15) {
			continue;
		}
		//if (y[i] > -0.17 || y[i] < -1.5 || z[i] < -0.15) {
		//	continue;
		//}
		ini_idx[i] = j;
		mi_x = min(mi_x, x[i]), mi_y = min(mi_y, y[i]);
		ma_x = max(ma_x, x[i]), ma_y = max(ma_y, y[i]);
		i++;
		//printf("%d %.5f %.5f %.5f\n", i, x[i], y[i], z[i]);
		//system("pause");

	}
	fclose(fp);
	tot_v_num = i;
	double rg_x = ma_x - mi_x, rg_y = ma_y - mi_y;
	mi_x -= rg_x * 0.1; ma_x += rg_x * 0.1;
	mi_y -= rg_y * 0.1; ma_y += rg_y * 0.1;
	double mid_x = (mi_x + ma_x) / 2;// , mid_y = (mi_y + ma_y) / 2;
	std::vector<Point_my> pts;
	pts.clear();
	for (int i = 0; i < tot_v_num; i++)
	{
		pts.push_back(Point_my{
			Point2d(
				(x[i] - mi_x) / (ma_x - mi_x)*(width - 1),
				height - 1 - (y[i] - mi_y) / (ma_y - mi_y)*(height - 1)),
			ini_idx[i] });
	}


	Mat img(height, width, CV_32FC3, cv::Scalar(0, 0, 0));
	
	for (int i = 0; i < pts.size(); i++) {
		//cout << "i: " << pts[i].pt2d << "\n";
		//img.at<Vec3f>(pts[i].pt2d) = Vec3f(255, 255, 255);
		cv::circle(img, pts[i].pt2d, 3, cv::Scalar(255, 255, 255));
		//cv::putText(img, to_string(pts_lft[i].v_idx), pts_lft[i].pt2d, cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0));
	}

	int cnt_line = 0;

	//get_line(slt_line, cnt_line, pts_lft);
	//get_line(slt_line, cnt_line, pts_rht);
	cal_line_idx_based_on_first_nstdis(slt_line, cnt_line, pts);

	for (int i = 0; i < cnt_line; i++) {
		printf("%d cnt_line %d ith size%d\n", i, cnt_line, slt_line[i].size());
		if (slt_line[i].size() > 1)
			for (int j = 0; j < slt_line[i].size() - 1; j++)
				cv::line(img, slt_line[i][j].pt2d, slt_line[i][j + 1].pt2d, cv::Scalar(0, 0, 255), 2);
	}
	//GaussianBlur(img, img, Size(3, 3), 0, 0);
	//blur(img, img, Size(10, 10));
	imshow("test.jpg", img);
	cv::waitKey(0);



	fopen_s(&fp, "slt_line_jaw_scratch.txt", "w");

	//std::sort(slt_line, slt_line + cnt_line, cmp_tp_btm);
	fprintf(fp, "%d\n", cnt_line);
	for (int i = 0; i < cnt_line; i++) {
		fprintf(fp, "%d %d ", i, slt_line[i].size());
		for (int j = 0; j < slt_line[i].size(); j++)
			fprintf(fp, "%d ", slt_line[i][j].v_idx);
		fprintf(fp, "\n");
	}
	fclose(fp);



	return 0;
}