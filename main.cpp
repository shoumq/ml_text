#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <cmath>
#include <locale>
#include <algorithm>
#include <codecvt>

using namespace std;

class LogisticRegressor {
private:
    unordered_map<string, double> weights; // Веса для каждого слова
    double bias = 0.0; // Смещение
    double learning_rate = 0.1; // Скорость обучения
    int epochs = 1000; // Количество эпох

    string stem(string word) {
        // Убираем некоторые распространенные окончания
        if (word.length() > 5) {
            if (word.substr(word.length() - 2) == "ый") {
                return word.substr(0, word.length() - 2);
            }
            else if (word.substr(word.length() - 3) == "ая") {
                return word.substr(0, word.length() - 3);
            }
            else if (word.substr(word.length() - 3) == "ое") {
                return word.substr(0, word.length() - 3);
            }
        }
        return word; // Если не подошло, возвращаем слово как есть
    }

    // Нормализация строки (приводим к нижнему регистру)
    string normalize(string& word) {
        transform(word.begin(), word.end(), word.begin(), ::tolower); // Приводим к нижнему регистру
        return stem(word); // Применяем стемминг
    }

    // Сигмоидная функция
    double sigmoid(double z) {
        return 1.0 / (1.0 + exp(-z));
    }

public:
    // Обучение
    void fit(vector<string>& texts, vector<double>& labels) {
        size_t n = texts.size();

        // Инициализация весов
        for (const auto& text : texts) {
            stringstream ss(text);
            string word;
            while (ss >> word) {
                weights[normalize(word)] = 0.0; // Инициализируем вес слова
            }
        }

        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < n; ++i) {
                double prediction = bias;
                stringstream ss(texts[i]);
                string word;
                while (ss >> word) {
                    word = normalize(word);
                    prediction += weights.count(word) ? weights[word] : 0.0; // Добавляем вес слова
                }

                // Применяем сигмоидную функцию
                double prob = sigmoid(prediction);

                // Обновление весов и смещения
                double error = labels[i] - prob;
                bias += learning_rate * error;

                ss.clear();
                ss.str(texts[i]);
                while (ss >> word) {
                    word = normalize(word);
                    if (weights.count(word)) {
                        weights[word] += learning_rate * error; // Обновляем вес слова
                    }
                }
            }
        }
    }

    // Предсказание
    int predict(string& text) {
        double prediction = bias;
        stringstream ss(text);
        string word;
        while (ss >> word) {
            word = normalize(word);
            prediction += weights.count(word) ? weights[word] : 0.0; // Добавляем вес слова, если он есть
        }

        // Применяем сигмоидную функцию и округляем до 0 или 1
        return sigmoid(prediction) >= 0.5 ? 1 : 0;
    }
};

void readCSV(const string& filename, vector<string>& texts, vector<double>& labels) {
    ifstream file(filename);
    string line;

    if (!file.is_open()) {
        cerr << "Не удалось открыть файл: " << filename << endl;
        return;
    }

    getline(file, line); // Пропускаем заголовок

    while (getline(file, line)) {
        stringstream ss(line);
        string text;
        string labelStr;

        getline(ss, text, ','); // Читаем текст до запятой
        getline(ss, labelStr, ','); // Читаем метку после запятой

        cout << "Текст: " << text << ", Метка: " << labelStr << endl; // Выводим текст и метку для проверки

        texts.push_back(text);
        labels.push_back(stod(labelStr)); // Преобразуем строку в double
    }
}

int main(int argc, char* argv[]) {
    system("chcp 65001");
    setlocale(LC_CTYPE, "ru_RU.UTF-8");

    vector<string> texts;
    vector<double> labels;

    readCSV("./data.csv", texts, labels);

    LogisticRegressor model;
    model.fit(texts, labels);

    if (argc > 1) {
        string t = argv[1];
        cout << "Предсказанное значение для '" << t.c_str() << "': " << model.predict(t) << endl;
    } else {
        string t = "плохо";
        cout << "Предсказанное значение для 'плохо': " << model.predict(t) << endl;

        string t2 = "хорошо";
        cout << "Предсказанное значение для 'хорошо': " << model.predict(t2) << endl;
    }

    return 0;
}