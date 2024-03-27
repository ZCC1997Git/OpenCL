compiler =g++
SrcDir   =./src 
ObjDir   =./obj
Target   =gemm
source   =$(foreach dir,$(SrcDir),$(wildcard $(dir)/*.cpp))
object   =$(patsubst %.cpp,$(ObjDir)/%.o,$(notdir $(source)))
cflags   +=-std=c++17 -fopenmp -O3

$(Target):$(object)
	$(compiler) $^ -o $@ -lOpenCL -fopenmp -L./FreeImage/usr/lib -lfreeimage

$(ObjDir)/%.o:./src/%.cpp
	@mkdir -p $(ObjDir)
	$(compiler) -c $(cflags) $< -o $@ -I./include

.PHONY:clean
clean:
	rm $(ObjDir)/*.o $(Target)

.PHONY:run 
run:$(Target)
	./$(Target)