#!/bin/bash

cmake -S . -B build
cmake --build build --config Release -j 8
