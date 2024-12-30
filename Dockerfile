# Используем официальный образ с компилятором C++
FROM gcc:latest

# Устанавливаем зависимости для CMake
RUN apt-get update && \
    apt-get install -y wget build-essential && \
    rm -rf /var/lib/apt/lists/*

# Устанавливаем CMake версии 3.29
RUN wget https://github.com/Kitware/CMake/releases/download/v3.29.0/cmake-3.29.0.tar.gz && \
    tar -xzvf cmake-3.29.0.tar.gz && \
    cd cmake-3.29.0 && \
    ./bootstrap && \
    make && \
    make install && \
    cd .. && \
    rm -rf cmake-3.29.0 cmake-3.29.0.tar.gz

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта в контейнер
COPY . .

COPY data.csv ./data.csv

# Создаем каталог для сборки
RUN mkdir build

# Переходим в каталог сборки и выполняем CMake
WORKDIR /app/build
RUN cmake .. && make

# Указываем команду для запуска приложения
CMD ["./mltext"]