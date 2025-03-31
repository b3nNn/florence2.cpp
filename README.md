# Florence2.cpp 🐾

## Software Requirements
- CMake 3.10 or higher
- C++17 compatible compiler
- Rust toolchain
- Git

## Installation
```bash
git clone https://github.com/b3nNn/florence2.cpp
cd florence2.cpp
```

## Compilation
Windows
```powershell
cmake -S . -B build -G "Visual Studio 16 2019"
cmake --build build --config Release
```

Mac/Linux
```bash
cmake -S . -B build
cmake --build build --config Release
```

## Execution

Windows
```powershell
.\build\Release\florence2.exe
```

Mac
```bash
DYLD_LIBRARY_PATH=$PWD/build ./build/florence2
```

Linux
```bash
LD_LIBRARY_PATH=$PWD/build ./build/florence2
```
