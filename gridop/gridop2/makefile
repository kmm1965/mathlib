TARGET = gridop2

CC = g++
RM = rm -rf 

OPTS = -std=c++14 -I ~/.local/include
BOOST_LIBS = -lboost_timer

SRC = $(TARGET).cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(OPTS) -o $(TARGET) $(SRC) $(BOOST_LIBS)

clean:
	-$(RM) $(TARGET)
