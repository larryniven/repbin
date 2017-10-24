CXXFLAGS += -I ../ -std=c++11 -L ../nn -L ../autodiff -L ../la -L ../ebt -L ../speech -L ../opt

bin = \
    k-means-learn \
    utt-autoenc-patch-learn \
    utt-autoenc-patch-recon

.PHONY: all clean

all: $(bin)

clean:
	-rm $(bin)
	-rm *.o

k-means-learn: k-means-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lopt -lla -lspeech -lebt -lblas

utt-autoenc-patch-learn: utt-autoenc-patch-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lopt -lla -lspeech -lebt -lblas

utt-autoenc-patch-recon: utt-autoenc-patch-recon.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lopt -lla -lspeech -lebt -lblas

