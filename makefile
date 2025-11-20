.PHONY: train

train:
	cd $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
	bsub < batch/scripts/train.sh
