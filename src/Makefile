CC ?= gcc
CFLAGS ?= -O3 -march=native -Wall
LIBS = -llapack -lblas -lm -lpthread

qap_bb: qap_bb.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f qap_bb
