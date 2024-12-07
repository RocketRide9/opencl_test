#!/usr/bin/env bash

set -euxo pipefail

if [ $# -eq 0 ]; then
    dotnet build --configuration Release && (cd "$(dirname "$0")" && cd ./bin/Release/net9.0 && ./ocl_test)
    exit 0
elif [ $# -gt 1 ]; then
    echo "Ожидается один аргумент, либо нисколько. Получено: $#"
    exit 1
fi

# Ручная сборка и запуск программы из директории с исполняемым файлом
if [[ "$1" == "--debug" ]]; then
    dotnet build --configuration Debug && (cd "$(dirname "$0")" && cd ./bin/Debug/net9.0 && ./ocl_test)
elif [[ "$1" == "--debug" ]]; then
    dotnet build --configuration Release && (cd "$(dirname "$0")" && cd ./bin/Release/net9.0 && ./ocl_test)
fi
