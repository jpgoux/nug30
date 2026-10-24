CC ?= gcc
CFLAGS ?= -O3 -march=native -funroll-loops -fno-math-errno -fno-trapping-math -Wall -Wno-misleading-indentation -Wno-unused-function
LIBS ?= -llapack -lblas -lm -lpthread

all: qap_bb

qap_bb: qap_bb.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

test_qap: test_qap.c qap_bb.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

# unit tests against brute force, then regression bench (optimum and tree fingerprint)
test: qap_bb test_qap
	./test_qap
	../tests/regress.sh ./qap_bb | cut -d' ' -f1-5 > regress.out
	grep -v '^#' ../tests/expected.txt | diff - regress.out && echo "regression: OK"

clean:
	rm -f qap_bb test_qap regress.out
