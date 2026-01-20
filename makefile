init:
	export LASERNET_LOG_LEVEL=INFO

preprocess:
	uv run src/lasernet/preprocess.py

train:
	uv run src/lasernet/train.py --network deep_cnn_lstm_large --field-type microstructure --max-epochs 25

evaluate:
	uv run src/lasernet/evaluate.py --network deep_cnn_lstm_large --field-type microstructure

predict:
	uv run src/lasernet/predict.py --timestep 18 --network deep_cnn_lstm_large --field-type microstructure

tensorboard:
	uv run tensorboard --logdir lightning_logs/ --host=0.0.0.0 --port=6006

test:
	uv run tests/test_utils.py

run-exp:
	uv run src/lasernet/experiments.py

hpc:
	uv run src/lasernet/preprocess.py

	uv run src/lasernet/train.py  --network deep_cnn_lstm_large --field-type temperature --num-workers 4 --max-epochs 100
	uv run src/lasernet/evaluate.py --network deep_cnn_lstm_large --field-type temperature
	uv run src/lasernet/predict.py --network deep_cnn_lstm_large --field-type temperature --timestep 18 

	uv run src/lasernet/train.py  --network deep_cnn_lstm_large --field-type temperature --num-workers 4 --max-epochs 100 --loss loss-front-combined
	uv run src/lasernet/evaluate.py --network deep_cnn_lstm_large --field-type temperature --loss loss-front-combined
	uv run src/lasernet/predict.py --network deep_cnn_lstm_large --field-type temperature --timestep 18 --loss loss-front-combined

	uv run src/lasernet/train.py  --network deep_cnn_lstm_large --field-type microstructure --num-workers 4 --max-epochs 100
	uv run src/lasernet/evaluate.py --network deep_cnn_lstm_large --field-type microstructure
	uv run src/lasernet/predict.py --network deep_cnn_lstm_large --field-type microstructure --timestep 18 

	uv run src/lasernet/train.py  --network deep_cnn_lstm_large --field-type microstructure --num-workers 4 --max-epochs 100 --loss loss-front-combined
	uv run src/lasernet/evaluate.py --network deep_cnn_lstm_large --field-type microstructure --loss loss-front-combined
	uv run src/lasernet/predict.py --network deep_cnn_lstm_large --field-type microstructure --timestep 18 --loss loss-front-combined

	uv run src/lasernet/train.py  --network transformer_unet_large --field-type temperature --num-workers 4 --max-epochs 100
	uv run src/lasernet/evaluate.py --network transformer_unet_large --field-type temperature
	uv run src/lasernet/predict.py --network transformer_unet_large --field-type temperature --timestep 18 

	uv run src/lasernet/train.py  --network transformer_unet_large --field-type temperature --num-workers 4 --max-epochs 100 --loss loss-front-combined
	uv run src/lasernet/evaluate.py --network transformer_unet_large --field-type temperature --loss loss-front-combined
	uv run src/lasernet/predict.py --network transformer_unet_large --field-type temperature --timestep 18 --loss loss-front-combined

	uv run src/lasernet/train.py  --network transformer_unet_large --field-type microstructure --num-workers 4 --max-epochs 100
	uv run src/lasernet/evaluate.py --network transformer_unet_large --field-type microstructure
	uv run src/lasernet/predict.py --network transformer_unet_large --field-type microstructure --timestep 18 

	uv run src/lasernet/train.py  --network transformer_unet_large --field-type microstructure --num-workers 4 --max-epochs 100 --loss loss-front-combined
	uv run src/lasernet/evaluate.py --network transformer_unet_large --field-type microstructure --loss loss-front-combined
	uv run src/lasernet/predict.py --network transformer_unet_large --field-type microstructure --timestep 18 --loss loss-front-combined

notebook:
	uv run jupyter nbconvert --to notebook --execute --inplace --debug notebooks/cascaded-prediction-demo.ipynb
