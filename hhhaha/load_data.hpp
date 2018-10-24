#pragma once
#include "utils_train.h"

void load_img_land_coef(std::string path, std::string sfx, std::vector<DataPoint> &img);

void load_land(std::string p, DataPoint &temp);

void load_img(std::string p, DataPoint &temp);

void load_land(std::string p, DataPoint &temp);

void cal_rect(DataPoint &temp);

void load_fitting_coef_one(std::string name, DataPoint &temp);