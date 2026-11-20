# BatchLAS: `make`, `make install`, `make check` from the source root.
#
# A thin layer over CMakePresets.json; every target is one cmake/ctest call, so the
# preset stays the single place a build shape is defined. `make help` lists the knobs.
#
#   make                          # configure (once) + build the release library
#   make install PREFIX=~/.local  # build + install (DESTDIR=... is honoured)
#   make check                    # install to a scratch prefix and build/run a consumer
#   make PRESET=dev               # any configure preset; the tree is build-<preset>

SHELL := bash
.SHELLFLAGS := -o pipefail -c
.NOTPARALLEL:
.DEFAULT_GOAL := all

UNAME_S := $(shell uname -s)

# macOS has one SYCL toolchain (AdaptiveCpp, Metal); elsewhere the compiler is auto-detected.
ifeq ($(UNAME_S),Darwin)
PRESET ?= release-metal
else
PRESET ?= release
endif

PREFIX ?= /usr/local
CMAKE ?= cmake
CMAKE_ARGS ?=
BUILD_DIR := build-$(PRESET)

# A cold DPC++ build peaks above 14 GB per translation unit (the release preset was
# OOM-killed at -j8 on a 125 GB box), so Linux gets one job per 24 GB of RAM. The
# AdaptiveCpp TUs on macOS are ~300 MB each: one job per core.
ifeq ($(UNAME_S),Darwin)
JOBS ?= $(shell sysctl -n hw.ncpu)
else
JOBS ?= $(shell n=$$(( $$(awk '/MemTotal/ {print $$2}' /proc/meminfo) / 25165824 )); \
	c=$$(nproc); [ $$n -lt 1 ] && n=1; [ $$n -gt $$c ] && n=$$c; echo $$n)
endif

CACHE := $(BUILD_DIR)/CMakeCache.txt

.PHONY: all build configure reconfigure install uninstall check clean distclean help

all: build

$(CACHE):
	@mkdir -p $(BUILD_DIR)
	$(CMAKE) --preset $(PRESET) -DCMAKE_INSTALL_PREFIX=$(PREFIX) $(CMAKE_ARGS) 2>&1 \
		| tee $(BUILD_DIR)/configure.log || { rm -f $(CACHE); exit 1; }
	@targets=$$(sed -n 's/^-- Using SYCL targets: //p' $(BUILD_DIR)/configure.log | tail -n 1); \
	compiler=$$(sed -n 's/^CMAKE_CXX_COMPILER:[A-Z]*=//p' $(CACHE)); \
	echo "batchlas: compiler $$compiler"; \
	echo "batchlas: SYCL targets $${targets:-(see $(BUILD_DIR)/configure.log)}"; \
	if command -v nvidia-smi >/dev/null 2>&1 && [ -n "$$targets" ] \
	   && ! echo "$$targets" | grep -qE 'nvptx|nvidia|cuda|generic'; then \
		echo "batchlas: WARNING: an NVIDIA GPU is present but this build has no CUDA target;"; \
		echo "batchlas:          it will run on the CPU only. Point CMAKE_ARGS=-DCMAKE_CXX_COMPILER=..."; \
		echo "batchlas:          at a CUDA-enabled DPC++ (or AdaptiveCpp) and run 'make reconfigure'."; \
	fi

configure: $(CACHE)

reconfigure:
	rm -f $(CACHE)
	$(MAKE) configure

build: $(CACHE)
	$(CMAKE) --build --preset $(PRESET) --parallel $(JOBS)

install: build
	$(CMAKE) --install $(BUILD_DIR) --prefix $(PREFIX)

uninstall:
	@test -f $(BUILD_DIR)/install_manifest.txt || { echo "batchlas: nothing installed from $(BUILD_DIR)"; exit 1; }
	tr '\n' '\0' < $(BUILD_DIR)/install_manifest.txt | xargs -0 rm -f

check: build
	bash examples/consumer_test.sh --build-dir $(BUILD_DIR) \
		--compiler "$$(sed -n 's/^CMAKE_CXX_COMPILER:[A-Z]*=//p' $(CACHE))"

clean:
	@test ! -f $(CACHE) || $(CMAKE) --build $(BUILD_DIR) --target clean

distclean:
	rm -rf $(BUILD_DIR)

help:
	@echo "targets: all (default) configure reconfigure install uninstall check clean distclean"
	@echo "PRESET=$(PRESET)  (cmake --list-presets; tree: $(BUILD_DIR))"
	@echo "PREFIX=$(PREFIX)  JOBS=$(JOBS)  CMAKE=$(CMAKE)"
	@echo "CMAKE_ARGS=...    extra configure arguments, e.g. -DCMAKE_CXX_COMPILER=/opt/dpcpp/bin/clang++"
