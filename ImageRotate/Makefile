compiler 	=g++
SrcDir   	=./src 
ObjDir   	=./obj
Target   	=ImageRotate
Target_cpu   =ImageRotate_cpu
source   	=$(foreach dir,$(SrcDir),$(wildcard $(dir)/*.cpp))
source_cpu  =$(wildcard ./CPU_src/*.cpp)
object   	=$(patsubst %.cpp,$(ObjDir)/%.o,$(notdir $(source)))
object_cpu  =$(patsubst %.cpp,$(ObjDir)/cpu/%.o,$(notdir $(source_cpu)))
cflags   	+=-std=c++17  -O3
NO_COLOR    =\033[0m
OK_COLOR    =\033[32;01m

all:$(Target) $(Target_cpu)

$(Target):$(object)
	$(compiler) $^ -o $@ -lOpenCL  -L./FreeImage/usr/lib -lfreeimage
	@printf "$(OK_COLOR) OpenCL executable file $(Target) has been created\n$(NO_COLOR)" 

$(Target_cpu):$(object_cpu)
	$(compiler) $^ -o $@  -L./FreeImage/usr/lib -lfreeimage
	@printf "$(OK_COLOR) CPU executable file $(Target_cpu) has been created\n$(NO_COLOR)" 

$(ObjDir)/%.o:./src/%.cpp
	@mkdir -p $(ObjDir)
	$(compiler) -c $(cflags) $< -o $@ -I./include

$(ObjDir)/cpu/%.o:./CPU_src/%.cpp
	@mkdir -p $(ObjDir)/cpu
	$(compiler) -c $(cflags) $< -o $@ -I./include

.PHONY:clean
clean:
	rm $(ObjDir)/* $(Target) $(Target_cpu) -rf

.PHONY:run 
run:$(Target) $(Target_cpu)
	@printf "$(OK_COLOR)Runing using OpenCL\n$(NO_COLOR)"
	export LD_LIBRARY_PATH=./FreeImage/usr/lib:$LD_LIBRARY_PATH && ./$(Target)
	@printf "$(OK_COLOR)Runing using CPU\n$(NO_COLOR)" 
	export LD_LIBRARY_PATH=./FreeImage/usr/lib:$LD_LIBRARY_PATH && ./$(Target_cpu)