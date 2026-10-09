SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

.DEFAULT_GOAL := help

UV ?= uv
CURL ?= curl
TAR ?= tar
CUDA ?=
UV_BACKEND_OPTIONS := $(if $(strip $(CUDA)),--no-group cpu-mps --group cuda)
CUDA_OPTION := $(if $(strip $(CUDA)),--cuda "$(CUDA)")
BUILD_DIR ?= build
CONFIG_FILE ?= assets/config.json
REPOS_FILE ?= $(BUILD_DIR)/repos.txt
MODEL_VERSION ?= 0.1.0
MODEL_RELEASE_TAG ?= v$(MODEL_VERSION)
MODEL_ARCHIVE_NAME ?= upgen_example_greeting_string_model_v$(MODEL_VERSION).tar.gz
MODEL_URL ?= https://github.com/unblockable/upgen/releases/download/$(MODEL_RELEASE_TAG)/$(MODEL_ARCHIVE_NAME)
MODEL_ARCHIVE ?= $(BUILD_DIR)/$(MODEL_ARCHIVE_NAME)
BEST_PARAMS_FILE ?= $(BUILD_DIR)/best_params.pkl
MODEL_DIR ?= $(BUILD_DIR)/model
PSF_FILE ?= $(BUILD_DIR)/upgen.psf
NUM_GENERATED ?= 1
NUM_HYPERPARAM_SAMPLES ?= 5000
NUM_TRAINING_SAMPLES ?= 10000
ARGS ?=

help:
	@echo "Generated artifacts default to $(BUILD_DIR)/"
	@echo "Set CUDA=N to use CUDA device N; otherwise models use MPS or the CPU"
	@echo
	@echo "make sync            Install UPGen and its model dependencies"
	@echo "make download-model  Download and extract the pretrained greeting model"
	@echo "make download        Download repository names for optional training"
	@echo "make tune            Tune model hyperparameters (optional training)"
	@echo "make train           Train the greeting model (optional)"
	@echo "make predict         Sample greetings from the trained model"
	@echo "make generate        Generate PSFs in $(PSF_FILE)"
	@echo "make format          Format Python source code"
	@echo "make lint            Check Python formatting and run Pylint"
	@echo "make check           Check the lockfile and environment"
	@echo "make clean           Deletes the build/ dir."
.PHONY: help

sync:
	$(UV) sync $(UV_BACKEND_OPTIONS)
.PHONY: sync

$(MODEL_ARCHIVE):
	mkdir -p "$(dir $(MODEL_ARCHIVE))"
	$(CURL) --fail --location --show-error \
		--output "$(MODEL_ARCHIVE)" "$(MODEL_URL)"

download-model: $(MODEL_ARCHIVE)
	$(TAR) -xzf "$(MODEL_ARCHIVE)" -C "$(BUILD_DIR)"
	test -f "$(BEST_PARAMS_FILE)"
	test -f "$(MODEL_DIR)/encoder.pkl"
	test -f "$(MODEL_DIR)/model.torch"
	@echo "Pretrained greeting model installed under $(BUILD_DIR)/"
.PHONY: download-model

download:
	mkdir -p "$(dir $(REPOS_FILE))"
	$(UV) run $(UV_BACKEND_OPTIONS) upgen-download-repos \
		--output "$(REPOS_FILE)" $(ARGS)
.PHONY: download

tune:
	mkdir -p "$(dir $(BEST_PARAMS_FILE))"
	$(UV) run $(UV_BACKEND_OPTIONS) upgen-train \
		--hyperparam_tune \
		--num_hyperparam_samples "$(NUM_HYPERPARAM_SAMPLES)" \
		--output_filepath "$(BEST_PARAMS_FILE)" \
		"$(REPOS_FILE)" $(CUDA_OPTION) $(ARGS)
.PHONY: tune

train:
	mkdir -p "$(MODEL_DIR)"
	$(UV) run $(UV_BACKEND_OPTIONS) upgen-train \
		--best_params_filepath "$(BEST_PARAMS_FILE)" \
		--output_dirpath "$(MODEL_DIR)" \
		--num_training_samples "$(NUM_TRAINING_SAMPLES)" \
		"$(REPOS_FILE)" $(CUDA_OPTION) $(ARGS)
.PHONY: train

predict:
	$(UV) run $(UV_BACKEND_OPTIONS) upgen-predict \
		"$(BEST_PARAMS_FILE)" \
		"$(MODEL_DIR)/encoder.pkl" \
		"$(MODEL_DIR)/model.torch" $(CUDA_OPTION) $(ARGS)
.PHONY: predict

generate:
	mkdir -p "$(dir $(PSF_FILE))"
	$(UV) run $(UV_BACKEND_OPTIONS) upgen-generate \
		"$(CONFIG_FILE)" \
		"$(BEST_PARAMS_FILE)" \
		"$(MODEL_DIR)/encoder.pkl" \
		"$(MODEL_DIR)/model.torch" \
		--num_generated "$(NUM_GENERATED)" \
		--output_filepath "$(PSF_FILE)" $(CUDA_OPTION) $(ARGS)
.PHONY: generate

format:
	$(UV) run $(UV_BACKEND_OPTIONS) black src
.PHONY: format

lint:
	$(UV) run $(UV_BACKEND_OPTIONS) black --check src
	$(UV) run $(UV_BACKEND_OPTIONS) pylint src/upgen
.PHONY: lint

check:
	$(UV) lock --check
	$(UV) sync --locked --check $(UV_BACKEND_OPTIONS)
.PHONY: check

clean:
	rm -rf $(BUILD_DIR)
.PHONY: clean
