SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

.DEFAULT_GOAL := help

UV ?= uv
BUILD_DIR ?= build
REPOS_FILE ?= $(BUILD_DIR)/repos.txt
BEST_PARAMS_FILE ?= $(BUILD_DIR)/best_params.pkl
MODEL_DIR ?= $(BUILD_DIR)/model
PSF_FILE ?= $(BUILD_DIR)/upgen.psf
NUM_GENERATED ?= 1
ARGS ?=

help:
	@echo "Generated artifacts default to $(BUILD_DIR)/"
	@echo
	@echo "make sync        Install the base package"
	@echo "make sync-model  Install the package with model dependencies"
	@echo "make download    Download repository names to $(REPOS_FILE)"
	@echo "make tune        Find model hyperparameters"
	@echo "make train       Train the greeting model"
	@echo "make predict     Sample greetings from the trained model"
	@echo "make generate    Generate PSFs in $(PSF_FILE)"
	@echo "make check       Check the lockfile and environment"
.PHONY: help

sync:
	$(UV) sync
.PHONY: sync

sync-model:
	$(UV) sync --extra model
.PHONY: sync-model

download:
	mkdir -p "$(dir $(REPOS_FILE))"
	$(UV) run upgen-download-repos --output "$(REPOS_FILE)" $(ARGS)
.PHONY: download

tune:
	mkdir -p "$(dir $(BEST_PARAMS_FILE))"
	$(UV) run --extra model upgen-train \
		--hyperparam_tune \
		--output_filepath "$(BEST_PARAMS_FILE)" \
		"$(REPOS_FILE)" $(ARGS)
.PHONY: tune

train:
	mkdir -p "$(MODEL_DIR)"
	$(UV) run --extra model upgen-train \
		--best_params_filepath "$(BEST_PARAMS_FILE)" \
		--output_dirpath "$(MODEL_DIR)" \
		"$(REPOS_FILE)" $(ARGS)
.PHONY: train

predict:
	$(UV) run --extra model upgen-predict \
		"$(BEST_PARAMS_FILE)" \
		"$(MODEL_DIR)/encoder.pkl" \
		"$(MODEL_DIR)/model.torch" $(ARGS)
.PHONY: predict

generate:
	mkdir -p "$(dir $(PSF_FILE))"
	$(UV) run --extra model upgen-generate \
		"$(BEST_PARAMS_FILE)" \
		"$(MODEL_DIR)/encoder.pkl" \
		"$(MODEL_DIR)/model.torch" \
		--num_generated "$(NUM_GENERATED)" \
		--output_filepath "$(PSF_FILE)" $(ARGS)
.PHONY: generate

check:
	$(UV) lock --check
	$(UV) sync --locked --check
.PHONY: check
