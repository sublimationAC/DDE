#include<stdio.h>
#include<cstring>
#include<iostream>
#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/SVD"
#include <Eigen/Eigenvalues> 
#include <unsupported/Eigen/SparseExtra>
#include <unsupported/Eigen/KroneckerProduct>

using namespace Eigen;
std :: string filename = "D:/sydney/first/data/tester_ (";
MatrixXf A,result;
const int nShape = 47;
const int nVerts = 11510;
const int nFaces = 11540;
const int test_num = 77;
const int iden_num = 10;
std::vector<int> mouse_edge[1000];
using namespace std;
//int cnt_vtx[nVerts];
void smooth_mesh(MatrixX3f &mesh, int iteration, std::vector<int> *mouse_edge) {
	while (iteration--)
	{
		MatrixX3f temp = mesh;

		for (int i_e = 0; mouse_edge[i_e].size() >0; i_e++) {
			if (mouse_edge[i_e].size() < 6) continue;
			int v = mouse_edge[i_e][0];
			mesh.block(v, 0, 1, 3).setZero();
			for (int j = 1; j < mouse_edge[i_e].size(); j++)
				mesh.row(v).array() += temp.row(mouse_edge[i_e][j]).array() / (mouse_edge[i_e].size() - 1);
		}
	}
}
void load_smooth_edge(std::vector<int> *mouse_edge) {
	puts("loading mouse edge...");
	FILE *fp;
	fopen_s(&fp, "D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3\\cal_coeffience_Q_M_u_e_3/mouse_point.txt", "r");
	int n;
	fscanf_s(fp, "%d", &n);
	for (int i = 0; i < n; i++) {
		mouse_edge[i].clear();
		int t, num;
		fscanf_s(fp, "%d%d", &t, &num);
		mouse_edge[i].push_back(t);
		for (int j = 0; j < num; j++) {
			fscanf_s(fp, "%d", &t);
			mouse_edge[i].push_back(t);
		}
	}
	fclose(fp);
}

int main() {
	A.resize(test_num, nShape*nVerts * 3);
	for (int i = 0; i < test_num; i++) {
		std::string name = filename + std::to_string(i + 1) + ")/Blendshape/shape.bs";
		std :: cout << name << std :: endl;
		FILE *fp;
		fopen_s(&fp,name.c_str(), "rb");
		int nShapes = 0, nVerts = 0, nFaces = 0;
		fread(&nShapes, sizeof(int), 1, fp);			// nShape = 46
		fread(&nVerts, sizeof(int), 1, fp);			// nVerts = 11510
		fread(&nFaces, sizeof(int), 1, fp);			// nFaces = 11540
		printf("%d %d %d\n", nShapes, nVerts, nFaces);

													// Load neutral expression B_0
		float temp;
		for (int j = 0; j < nVerts * 3; j++) {
			fread(&temp, sizeof(float), 1, fp);
			A(i, j) = temp;
		}
		//for (int j = 0; j < 10; j++)
		//	printf("%.10f ", A(i, j));
		//puts("");

		// Load other expressions B_i ( 1 <= i <= 46 )
		for (int exprId = 0; exprId < nShapes; exprId++) 	
			for (int j = 0; j < nVerts * 3; j++) {
				fread(&temp, sizeof(float), 1, fp);
				A(i, 3 * nVerts*(exprId + 1) + j) = temp;
			}
		
		fclose(fp);
	}
	puts("loading initial bldshps complete!...");
	load_smooth_edge(mouse_edge);
	for (int i_id=0;i_id< iden_num;i_id++)
		for (int i_exp = 0; i_exp < nShape; i_exp++) {
			printf("smoothing id:%d exp:%d\n",i_id,i_exp);
			Eigen::MatrixX3f temp(nVerts, 3);
			for (int i_v = 0; i_v < nVerts; i_v++)
				for (int axis = 0; axis < 3; axis++)
					temp(i_v, axis) = A(i_id, i_exp*nVerts*3 + i_v * 3 + axis);
			smooth_mesh(temp, 25, mouse_edge);
			for (int i_v = 0; i_v < nVerts; i_v++)
				for (int axis = 0; axis < 3; axis++)
					A(i_id, i_exp*nVerts * 3 + i_v * 3 + axis) = temp(i_v, axis);
		}
			

	puts("saving...");
	FILE *fp;
	fopen_s(&fp, "blendshape_ide_svd_77_ite25_bound.lv", "wb");
	for (int i = 0; i < iden_num; i++)
		for (int j = 0; j < 3 * nVerts*nShape; j++)
			fwrite(&A(i, j), sizeof(float), 1, fp);
	fclose(fp);

	//MatrixXf mean = A.rowwise().mean();
	//cout << mean.rows() << ' ' << mean.cols() << '\n';
	//cout << mean;
	//cout << A(32, 22301) << "\n";
	//cout << A(24, 11144) << "\n";
	//cout << A(53, 32333) << "\n";
	//cout << A(60, 523441) << "\n";
	//cout << A(8, 765201) << "\n";

	//system("pause");
	//VectorXf vmean = mean;
	//A.colwise() -= vmean;
	//BDCSVD<Eigen::MatrixXf> svd((A * A.transpose()).array()/ A.cols(), ComputeThinU);//*A.transpose()
	//puts("asd");
	////printf("%d %d %d %d\n", U.rows(), U.cols(), V.rows(), V.cols());
	////cout <<"U:\n"<< U << endl;
	////cout << "V:\n" << V << endl;
	//MatrixXf  S = svd.singularValues();
	////printf("%d %d\n",S.cols(),S.rows());
	//float tot = 0,temp=0;
	//for (int i = 0; i < test_num; i++) tot += sqrt(S(i));
	//for (int i = 0; i < test_num; i++, puts("")) {
	//	temp += sqrt(S(i));
	//	printf("%d %.10f %.2f", i, sqrt(S(i)), temp / tot * 100);
	//}
	//A.colwise() += vmean;
	//result = svd.matrixU().block(0, 0, test_num, iden_num).transpose()*A;
	//FILE *fp;
	//fopen_s(&fp, "blendshape_ide_svd_50.lv", "wb");
	//for (int i = 0; i < iden_num; i++)
	//	for (int j = 0; j < 3 * nVerts*nShape; j++)
	//		fwrite(&result(i,j),sizeof(float),1,fp);
	//fclose(fp);
	//fopen_s(&fp, "blendshape_ide_svd_value_sqrt_50.txt", "w");
	//for (int i = 0; i < iden_num; i++)
	//	fprintf(fp, "%.10f\n", sqrt(S(i)));
	//fclose(fp);





	system("pause");
	return 0;
}


/*
test svd

MatrixXf m = MatrixXf::Random(3, 2);
m.resize(3, 2);
cout << "Here is the matrix m:" << endl << m << endl;
JacobiSVD<MatrixXf> svd(m, ComputeThinU | ComputeThinV);
cout << "Its singular values are:" << endl << svd.singularValues() << endl;
cout << "Its left singular vectors are the columns of the thin U matrix:" << endl << svd.matrixU() << endl;
cout << "Its right singular vectors are the columns of the thin V matrix:" << endl << svd.matrixV() << endl;*/