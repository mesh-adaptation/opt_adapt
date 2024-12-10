all: uniform hessian go plot_qoi

uniform:
	cd .. && \
	for n in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do \
		python3 opt_uniform.py turbine --n $$n; \
	done

hessian:
	cd .. && \
	for target in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000; do \
		python3 opt_hessian.py turbine --n 2 --target $$target; \
	done

go:
	cd .. && \
	for target in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000; do \
		python3 opt_go.py turbine --n 2 --target $$target; \
	done

plot:
	python3 plot_qoi.py

clean:
	rm *.log
