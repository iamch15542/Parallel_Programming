default: conv

CC = gcc-10
FLAGS = -O3 -lOpenCL -m64 -ffloat-store -w -g -fPIC -lm

OBJS = main.o bmpfuncs.o hostFE.o serialConv.o helper.o

CUDA_LINK_FLAGS =  -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' 
CUDA_COMPILE_FLAGS = --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3

conv: $(OBJS)
	# $(CC) -o $@ $(OBJS) $(FLAGS)
	nvcc ${CUDA_LINK_FLAGS} -o $@ $(OBJS)

%.o: %.c
	$(CC) -c $(FLAGS) $< -o $@

hostFE.o: hostFE.cu hostFE.h
	nvcc ${CUDA_COMPILE_FLAGS} -c hostFE.cu -o $@

clean:
	rm -f conv *.o output.bmp ref.bmp