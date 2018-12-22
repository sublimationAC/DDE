CFLAGS = -std=c++11 -O3
objects = main.o face_x.o fern.o regressor.o utils.o ceres_cal_exp_ide_noconstrain_dde.o calculate_coeff_dde.o post_processing.o load_data_test.o dde_x.o regressor_dde.o fern_dde.o utils_dde_test.o

ddex: $(objects)
	g++ $(CFLAGS) -o ddex $(objects) -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lceres -lglog -lgflags -lcholmod -llapack -lcxsparse -lblas  -lpthread -fopenmp
	@echo To run the demo program, make sure haarcascade_frontalface_alt2.xml model.xml.gz test.jpg are in the working directory.

main.o: main.cpp face_x.h regressor.h fern.h utils.h calculate_coeff_dde.hpp post_processing.hpp load_data_test.hpp dde_x.hpp regressor_dde.hpp fern_dde.hpp utils_dde_test.hpp
	g++ $(CFLAGS) -c main.cpp

face_x.o: face_x.cpp face_x.h regressor.h fern.h utils.h
	g++ $(CFLAGS) -c face_x.cpp

fern.o: fern.cpp fern.h utils.h
	g++ $(CFLAGS) -c fern.cpp

regressor.o: regressor.cpp regressor.h fern.h utils.h
	g++ $(CFLAGS) -c regressor.cpp

utils.o: utils.cpp utils.h
	g++ $(CFLAGS) -c utils.cpp 

ceres_cal_exp_ide_noconstrain_dde.o: ceres_cal_exp_ide_noconstrain_dde.cpp calculate_coeff_dde.hpp post_processing.hpp load_data_test.hpp dde_x.hpp regressor_dde.hpp fern_dde.hpp utils_dde_test.hpp
	g++ $(CFLAGS) -c ceres_cal_exp_ide_noconstrain_dde.cpp

calculate_coeff_dde.o: calculate_coeff_dde.cpp calculate_coeff_dde.hpp post_processing.hpp load_data_test.hpp dde_x.hpp regressor_dde.hpp fern_dde.hpp utils_dde_test.hpp
	g++ $(CFLAGS) -c calculate_coeff_dde.cpp

post_processing.o: post_processing.cpp post_processing.hpp load_data_test.hpp dde_x.hpp regressor_dde.hpp fern_dde.hpp utils_dde_test.hpp
	g++ $(CFLAGS) -c post_processing.cpp

load_data_test.o: load_data_test.cpp load_data_test.hpp dde_x.hpp regressor_dde.hpp fern_dde.hpp utils_dde_test.hpp
	g++ $(CFLAGS) -c load_data_test.cpp

dde_x.o: dde_x.cpp dde_x.hpp regressor_dde.hpp fern_dde.hpp utils_dde_test.hpp
	g++ $(CFLAGS) -c dde_x.cpp

regressor_dde.o: regressor_dde.cpp regressor_dde.hpp fern_dde.hpp utils_dde_test.hpp
	g++ $(CFLAGS) -c regressor_dde.cpp

fern_dde.o: fern_dde.cpp fern_dde.hpp utils_dde_test.hpp
	g++ $(CFLAGS) -c fern_dde.cpp

utils_dde_test.o: utils_dde_test.cpp utils_dde_test.hpp
	g++ $(CFLAGS) -c utils_dde_test.cpp

clean:
	-rm ddex $(objects)	