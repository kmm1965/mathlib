TARGET = gridop1_dim

CC = g++
RM = rm -rf 

OPTS = -std=c++14 -I ~/.local/include

SRC = $(TARGET).cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(OPTS) -o $(TARGET) $(SRC)

clean:
	-$(RM) $(TARGET)
