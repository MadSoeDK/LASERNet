init:
	export LASERNET_LOG_LEVEL=INFO
	@if [ -d /dtu ]; then export BLACKHOLE=/dtu/blackhole/06/168550; if [ ! -L data ]; then rm -rf data && ln -s $$BLACKHOLE/data data; fi; fi

preprocess:
	uv run src/lasernet/preprocess.py

train:
	uv run src/lasernet/train.py

evaluate:
	uv run src/lasernet/evaluate.py 

predict:
	uv run src/lasernet/predict.py --timestep 18 --slice-index 47

tensorboard:
	uv run tensorboard --logdir lightning_logs/ --host=0.0.0.0 --port=6006

test:
	uv run tests/test_utils.py

hpc:
	uv run src/lasernet/preprocess.py
	uv run src/lasernet/train.py  --network temperaturecnn --num-workers 31 --max-epochs 100
	uv run src/lasernet/evaluate.py --network temperaturecnn
	uv run src/lasernet/predict.py --network temperaturecnn --timestep 18 --slice-index 47
	uv run src/lasernet/predict.py --network temperaturecnn --timestep 19 --slice-index 47
	uv run src/lasernet/predict.py --network temperaturecnn --timestep 20 --slice-index 47
	uv run src/lasernet/predict.py --network temperaturecnn --timestep 21 --slice-index 47

	uv run src/lasernet/train.py --network microstructurecnn --num-workers 31 --max-epochs 100
	uv run src/lasernet/evaluate.py --network microstructurecnn
	uv run src/lasernet/predict.py --network microstructurecnn --timestep 18 --slice-index 47
	uv run src/lasernet/predict.py --network microstructurecnn --timestep 19 --slice-index 47
	uv run src/lasernet/predict.py --network microstructurecnn --timestep 20 --slice-index 47
	uv run src/lasernet/predict.py --network microstructurecnn --timestep 21 --slice-index 47


