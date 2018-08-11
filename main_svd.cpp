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
const int iden_num = 25;
using namespace std;
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
	MatrixXf mean = A.rowwise().mean();
	cout << mean.rows() << ' ' << mean.cols() << '\n';
	VectorXf vmean = mean;
	A.colwise() -= vmean;
	BDCSVD<Eigen::MatrixXf> svd((A * A.transpose()).array()/ A.cols(), ComputeThinU);//*A.transpose()
	puts("asd");
	//printf("%d %d %d %d\n", U.rows(), U.cols(), V.rows(), V.cols());
	//cout <<"U:\n"<< U << endl;
	//cout << "V:\n" << V << endl;
	MatrixXf  S = svd.singularValues();
	//printf("%d %d\n",S.cols(),S.rows());
	float tot = 0,temp=0;
	for (int i = 0; i < test_num; i++) tot += sqrt(S(i));
	for (int i = 0; i < test_num; i++, puts("")) {
		temp += sqrt(S(i));
		printf("%d %.10f %.2f", i, sqrt(S(i)), temp / tot * 100);
	}
	A.colwise() += vmean;
	result = svd.matrixU().block(0, 0, iden_num, test_num)*A;
	FILE *fp;
	fopen_s(&fp, "blendshape_ide_svd.lv", "wb");
	for (int i = 0; i < iden_num; i++)
		for (int j = 0; j < 3 * nVerts*nShape; j++)
			fwrite(&result(i,j),sizeof(float),1,fp);
	fclose(fp);
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