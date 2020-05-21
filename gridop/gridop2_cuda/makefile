TARGET = gridop2_cuda

CC = nvcc
RM = rm -rf 

BOOST_LIBS = -L ${BOOST_LIBRARY_PATH} \
	-lboost_timer -lboost_chrono

HPC_LIB = -L ../../lib -lhpc
 
OPTS = -std=c++11 -I ../.. -I ${BOOST_INCLUDE_PATH}

SRC = gridop2.cu

OBJS = $(SRC:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(BOOST_LIBS) $(HPC_LIB)

%.o: %.cu
	$(CC) $(OPTS) -c $< -o $@

clean:
	-$(RM) $(TARGET) $(OBJS)
