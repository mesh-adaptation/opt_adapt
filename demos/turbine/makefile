all: uniform hessian

uniform:
	for n in 1 2 3 4 5 6 7 8; do \
		python3 opt_uniform.py --n $$n; \
	done

hessian:
	for target in 1000 2000 3000 4000 5000 6000 7000 8000; do \
		python3 opt_hessian.py --target $$target; \
	done

go:
	for target in 1000 2000 3000 4000 5000 6000 7000 8000; do \
		python3 opt_go.py --target $$target; \
	done

clean:
	rm *.log
