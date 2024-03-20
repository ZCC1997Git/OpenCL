compiler =g++
SrcDir   =./src 
ObjDir   =./obj

source   =$(foreach dir,$(SrcDir),$(wildcard $(dir)/*.cpp))
object   =$(patsubst %.cpp,$(ObjDir)/%.o,$(notdir $(source)))


VecAdd:$(object)
	$(compiler) $^ -o $@ -lOpenCL

$(ObjDir)/%.o:./src/%.cpp 
	$(compiler) -c $(cflags) $< -o $@ -I./include

.PHONY:clean
clean:
	rm $(ObjDir)/*.o VecAdd

.PHONY:run 
run:VecAdd
	./VecAdd