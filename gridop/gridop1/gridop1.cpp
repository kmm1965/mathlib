#include "pch.h"
#include "../gridop1_src/gridop1_cpp.h"
#include "../gridop1_src/catch.h"

int main(int argc, char* argv[])
{
    try {
        return main1(argc, argv);
    } CATCH_EXCEPTIONS()
}
