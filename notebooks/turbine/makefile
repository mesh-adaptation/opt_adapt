all: uniform hessian

uniform:
	python3 opt_uniform.py --n 1
	python3 opt_uniform.py --n 2
	python3 opt_uniform.py --n 4

hessian:
	python3 opt_hessian.py --target 1000
	python3 opt_hessian.py --target 2000
	python3 opt_hessian.py --target 4000
	python3 opt_hessian.py --target 8000
