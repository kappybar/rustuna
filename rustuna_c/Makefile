.DEFAULT_GOAL := build

CFLAGS=-std=c11 -Wno-incompatible-pointer-types
CXXFLAGS=-std=c++11 -Wall
LDFLAGS=-L . -lrustuna
TARGET=rustuna.h librustuna.so

SRCS=$(wildcard src/*.rs)
C_TESTS=$(wildcard tests_c/*.c)
CPP_TESTS=$(wildcard tests_cpp/*.cpp)
TESTS=$(C_TESTS:.c=.exe) $(CPP_TESTS:.cpp=.exe)

# Rename the shared library to librustuna.so.
# https://github.com/rust-lang/cargo/issues/1970#issuecomment-138692829
librustuna.so: $(SRCS) Cargo.toml
	cargo build
	cp ../target/debug/librustuna_c.so ./librustuna.so

rustuna.h: $(SRCS) cbindgen.toml
	cbindgen --config cbindgen.toml --crate rustuna_c --output $@

tests_c/%.exe: $(TARGET) tests_c/%.c
	$(CC) -o $@ tests_c/$*.c $(CFLAGS) $(LDFLAGS)

tests_cpp/%.exe: $(TARGET) rustuna.hpp tests_cpp/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ tests_cpp/$*.cpp $(LDFLAGS)

.PHONY: build
build: $(TARGET)

.PHONY: test
test: $(TESTS)
	@export LD_LIBRARY_PATH=. ; \
	for i in $^; do ./$$i > ./$$i.log; \
	if [ $$? -eq 0 ]; then echo "  passed: $$i"; \
	else echo "  failed: $$i  (stdout: ./$$i.log)"; exit 1; \
	fi; done

.PHONY: clean
clean:
	cargo clean
	rm -f *.so rustuna.h tests_c/*.exe tests_c/*.log tests_cpp/*.exe tests_cpp/*.log

.PHONY: fmt
fmt:
	cargo fmt
	clang-format -i tests_c/*.c tests_cpp/*.cpp *.hpp
