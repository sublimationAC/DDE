/*
FaceX-Train is a tool to train model file for FaceX, which is an open
source face alignment library.

Copyright(C) 2015  Yang Cao

This program is free software : you can redistribute it and / or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <map>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#include <opencv2/opencv.hpp>

#include "regressor_train.h"
#include "utils_dde.hpp"
#include "load_data.hpp"

//#define win64
#define linux
#define changed

#ifdef win64

std::string fwhs_path = "D:/sydney/first/data_me/FaceWarehouse";
std::string lfw_path = "D:/sydney/first/data_me/lfw_image";
std::string gtav_path = "D:/sydney/first/data_me/GTAV_image";
std::string test_path = "D:/sydney/first/data_me/test";
std::string test_path_one = "D:/sydney/first/data_me/test_only_one";
std::string coef_path = "D:/sydney/first/data_me/fitting_coef/ide_fw_p1.lv";


#endif // win64
#ifdef linux
std::string fwhs_path = "/home/weiliu/DDE/cal_coeff/data_me/FaceWarehouse";
std::string lfw_path = "/home/weiliu/DDE/cal_coeff/data_me/lfw_image";
std::string gtav_path = "/home/weiliu/DDE/cal_coeff/data_me/GTAV_image";
std::string test_path = "./test";
std::string test_path_one = "/home/weiliu/DDE/cal_coeff/data_me/test_only_one";
std::string coef_path = "../fitting_coef/ide_fw_p1.lv";
#endif // linux

using namespace std;

vector<DataPoint> GetTrainingData()
{
	vector<DataPoint> result;
	load_img_land_coef(test_path, ".jpg", result);
	return result;
}


Eigen::MatrixXf bldshps(G_iden_num, G_nShape * 3 * G_nVerts);
std::string bldshps_path = "D:\\sydney\\first\\code\\2017\\deal_data_2\\deal_data/blendshape_ide_svd_77.lv";

std::string cal_coef_land_name = "test_coef_land_olsgm_25.txt";
std::string cal_coef_mesh_name = "test_coef_mesh_olsgm_25.txt";
int main(int argc, char *argv[])
{
	cout << "Testing begin." << endl;
	load_bldshps(bldshps, bldshps_path);
	vector<DataPoint> training_data = GetTrainingData();
	//system("pause");
	Eigen::MatrixX3f mesh;
	test_load_mesh(training_data, bldshps, 0, cal_coef_mesh_name);
	test_load_3d_land(training_data, bldshps, 0, cal_coef_land_name);

	return 0;
}
