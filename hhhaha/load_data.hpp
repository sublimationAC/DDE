#pragma once
#include "utils_train.h"

void load_img_land(std::string path, std::string sfx, std::vector<DataPoint> &img);

void load_land(std::string p, DataPoint &temp);

void load_img(std::string p, DataPoint &temp);

void test_data_2dland(DataPoint &temp);

void cal_rect(DataPoint &temp);