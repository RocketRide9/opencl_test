#!/usr/bin/env bash

# Ручная сборка и запуск программы из директории с исполняемым файлом
# Скрипт нужно запускать из его директории
dotnet build --configuration Release && (cd ./bin/Release/net8.0 && ./ocl_test)
