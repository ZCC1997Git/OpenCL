compiler =g++
SrcDir   =./src 
ObjDir   =./obj
Target   =gemm
source   =$(foreach dir,$(SrcDir),$(wildcard $(dir)/*.cpp))
object   =$(patsubst %.cpp,$(ObjDir)/%.o,$(notdir $(source)))


$(Target):$(object)
	$(compiler) $^ -o $@ -lOpenCL

$(ObjDir)/%.o:./src/%.cpp 
	$(compiler) -c $(cflags) $< -o $@ -I./include

.PHONY:clean
clean:
	rm $(ObjDir)/*.o $(Target)

.PHONY:run 
run:$(Target)
	./$(Target)