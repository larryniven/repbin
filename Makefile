CXXFLAGS += -I ../ -std=c++11 -L ../nn -L ../autodiff -L ../la -L ../ebt -L ../speech -L ../opt

bin = \
    k-means-learn

.PHONY: all clean

all: $(bin)

clean:
	-rm $(bin)
	-rm *.o

k-means-learn: k-means-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lopt -lla -lspeech -lebt -lblas

