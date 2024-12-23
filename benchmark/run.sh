#!/usr/bin/env bash

set -euxo pipefail

BIN_MAME="benchmark"
DOTNET_VER="net8.0"

# Ручная сборка и запуск программы из директории с исполняемым файлом
# если аргумент не указан - собрать и запустить в релизной конфигурации
if [ $# -eq 0 ]; then
    dotnet build --configuration Release && (cd "$(dirname "$0")" && cd "./bin/Release/$DOTNET_VER" && "./$BIN_MAME")
    exit 0
elif [ $# -gt 1 ]; then
    echo "Ожидается один аргумент, либо нисколько. Получено: $#"
    exit 1
fi

if [[ "$1" == "--debug" ]]; then
    dotnet build --configuration Debug && (cd "$(dirname "$0")" && cd "./bin/Debug/$DOTNET_VER" && "./$BIN_MAME")
elif [[ "$1" == "--release" ]]; then
    dotnet build --configuration Release && (cd "$(dirname "$0")" && cd "./bin/Release/$DOTNET_VER" && "./$BIN_MAME")
fi
