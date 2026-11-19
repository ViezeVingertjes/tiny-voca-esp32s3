// One sentence per stdin line -> "ids<TAB>stress" (comma-separated) per stdout line,
// or "ERR <message>". Uses the C++ dizione port in third_party/dizione.
//   dizione_cli /path/to/phonemizer < sentences.txt
// Build (speak_s3.py looks for out/bin/dizione_cli):
//   mkdir -p out/bin
//   c++ -O2 -std=c++17 -Ithird_party/dizione -o out/bin/dizione_cli tools/dizione_cli.cpp third_party/dizione/*.cpp -lpthread
#include "dizione.h"
#include <iostream>
#include <string>

int main(int argc, char **argv) {
    if (argc < 2) { std::cerr << "usage: dizione_cli PHONEMIZER_DIR\n"; return 2; }
    dizione::Phonemizer phonemizer(dizione::load_directory(argv[1]));
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            dizione::Phonemized p = phonemizer.phonemize(line);
            for (size_t i = 0; i < p.ids.size(); i++) std::cout << (i ? "," : "") << p.ids[i];
            std::cout << '\t';
            for (size_t i = 0; i < p.stress.size(); i++) std::cout << (i ? "," : "") << p.stress[i];
            std::cout << '\n';
        } catch (const std::exception &e) {
            std::cout << "ERR " << e.what() << '\n';
        }
        std::cout.flush();
    }
    return 0;
}
