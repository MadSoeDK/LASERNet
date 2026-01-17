init:
	export LASERNET_LOG_LEVEL=INFO

preprocess:
	uv run src/lasernet/preprocess.py

train:
	uv run src/lasernet/train.py

tensorboard:
	tensorboard --logdir lightning_logs/ --host=0.0.0.0 --port=6006
