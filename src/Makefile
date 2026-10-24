CC ?= gcc
CFLAGS ?= -O3 -march=native -funroll-loops -fno-math-errno -fno-trapping-math -Wall -Wno-misleading-indentation -Wno-unused-function
LIBS ?= -llapack -lblas -lm -lpthread

all: qap_bb

qap_bb: qap_bb.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

test_qap: test_qap.c qap_bb.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

# unit tests against brute force, regression bench (optimum and tree fingerprint),
# then checkpoint and resume (same tree as an uninterrupted run)
test: qap_bb test_qap
	./test_qap
	../tests/regress.sh ./qap_bb | cut -d' ' -f1-5 > regress.out
	grep -v '^#' ../tests/expected.txt | diff - regress.out && echo "regression: OK"
	../tests/ckpt_test.sh ./qap_bb

clean:
	rm -f qap_bb test_qap regress.out
