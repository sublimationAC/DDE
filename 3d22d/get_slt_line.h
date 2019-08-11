#pragma once
#include "math_headers.h"

void get_scratch_line(std::string path, int &num, std::vector <int> *scrtch_line);
void deal_scratch_line(int num, std::vector <int> *scrtch_line);
void get_tst_slt_pts(Eigen::MatrixXi &slt_pts,int &slt_line_total);