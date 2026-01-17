init:
	export LASERNET_LOG_LEVEL=INFO

preprocess:
	uv run src/lasernet/preprocess.py
