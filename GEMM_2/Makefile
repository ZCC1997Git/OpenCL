compiler 	=g++
SrcDir   	=./src 
ObjDir   	=./obj
Target   	=./bin/gemm
source   	=$(foreach dir,$(SrcDir),$(wildcard $(dir)/*.cpp))
object   	=$(patsubst %.cpp,$(ObjDir)/%.o,$(notdir $(source)))
headfile 	=$(wildcard ./include/*.hpp)
cflags   	+=-std=c++17  -O3  -fno-tree-vectorize  -mavx2 -mfma
NO_COLOR    =\033[0m
OK_COLOR    =\033[32;01m

all:$(Target) $(Target_cpu)

$(Target):$(object)
	@mkdir -p ./bin
	$(compiler) $^ -o $@ -lOpenCL 
	@printf "$(OK_COLOR) OpenCL executable file $(Target) has been created\n$(NO_COLOR)" 


$(ObjDir)/%.o:./src/%.cpp $(headfile)
	@mkdir -p $(ObjDir)
	$(compiler) -c $(cflags) $< -o $@ -I./include

.PHONY:clean
clean:
	rm $(ObjDir)/* $(Target) -rf

.PHONY:run 
run:$(Target) $(Target_cpu)
	@printf "$(OK_COLOR)Runing using OpenCL\n$(NO_COLOR)"
	./$(Target)