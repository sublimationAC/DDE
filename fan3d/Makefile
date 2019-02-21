CFLAGS = -std=c++11 -O3

objects = main.o ceres_cal_exp_ide_noconstrain.o calculate_coeff.o 2dland.o

fitting-3dland: $(objects)
	g++ $(CFLAGS) -o fitting-3dland $(objects) -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lceres -lcholmod -llapack -lblas -lcxsparse -lglog -lpthread -fopenmp

main.o: main.cpp calculate_coeff.h 2dland.h
	g++ $(CFLAGS) -c main.cpp

ceres_cal_exp_ide_noconstrain.o: ceres_cal_exp_ide_noconstrain.cpp calculate_coeff.h 2dland.h
	g++ $(CFLAGS) -c ceres_cal_exp_ide_noconstrain.cpp

calculate_coeff.o: calculate_coeff.cpp calculate_coeff.h 2dland.h
	g++ $(CFLAGS) -c calculate_coeff.cpp

2dland.o: 2dland.cpp 2dland.h
	g++ $(CFLAGS) -c 2dland.cpp

clean:
	-rm fitting-3dland $(objects)
