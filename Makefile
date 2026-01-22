SHELL := bash
.DELETE_ON_ERROR:
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-print-directory

build:
	@echo "Building project..."
	@cargo build

build_mac:
	@echo "Building project for macOS..."
	@cargo build --features metal  --release

test:
	@echo "Running tests..."
	@cargo test

clean:
	@echo "Cleaning project..."
	@cargo clean

fmt:
	@echo "Formatting code..."
	@cargo fmt --all -- --check

lint:
	@echo "Linting code..."
	@cargo clippy --no-deps --all-targets -- -D warnings

help:
	@echo "Available commands:"
	@echo "  build         - Build the project"
	@echo "  test          - Run tests"
	@echo "  clean         - Clean the project"
	@echo "  fmt           - Format the code"
	@echo "  lint          - Lint the code"
	@echo "  help          - Show this help message"

.PHONY: build test clean fmt lint help