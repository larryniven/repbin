CXXFLAGS += -I ../ -std=c++11 -L ../nn -L ../autodiff -L ../la -L ../ebt -L ../speech -L ../opt

bin = \
    k-means-learn \
    utt-autoenc-patch-learn \
    utt-autoenc-patch-recon \
    utt-autoenc-lstm-learn \
    utt-autoenc-lstm-recon

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

utt-autoenc-lstm-learn: utt-autoenc-lstm-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lopt -lla -lspeech -lebt -lblas

utt-autoenc-lstm-recon: utt-autoenc-lstm-recon.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lopt -lla -lspeech -lebt -lblas

