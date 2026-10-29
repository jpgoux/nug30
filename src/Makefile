CC ?= gcc
CFLAGS ?= -O3 -march=native -funroll-loops -fno-math-errno -fno-trapping-math -Wall -Wno-misleading-indentation -Wno-unused-function
LIBS ?= -llapack -lopenblas -lm -lpthread

all: qap_bb

qap_bb: qap_bb.c dnn.c dnn.h
	$(CC) $(CFLAGS) -o $@ qap_bb.c dnn.c $(LIBS)

test_qap: test_qap.c qap_bb.c dnn.c dnn.h
	$(CC) $(CFLAGS) -o $@ test_qap.c dnn.c $(LIBS)

dnntest: dnntest.c dnn.c dnn.h
	$(CC) $(CFLAGS) -o $@ dnntest.c dnn.c $(LIBS)

# unit tests against brute force, regression bench (optimum and node count; FW iteration
# counts are kept in the expected files but not compared, they move with the processor),
# then checkpoint and resume (same tree as an uninterrupted run)
test: qap_bb test_qap
	./test_qap
	../tests/regress.sh ./qap_bb | cut -d' ' -f1-4 > regress.out
	grep -v '^#' ../tests/expected.txt | cut -d' ' -f1-4 | diff - regress.out && echo "regression: OK"
	../tests/regress.sh ./qap_bb -dom | cut -d' ' -f1-4 > regress_dom.out
	grep -v '^#' ../tests/expected_dom.txt | cut -d' ' -f1-4 | diff - regress_dom.out && echo "regression -dom: OK"
	../tests/ckpt_test.sh ./qap_bb

clean:
	rm -f qap_bb test_qap regress.out regress_dom.out
