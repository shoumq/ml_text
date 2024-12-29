//
// Created by Mac on 26.12.2024.
//

#ifndef LOGISTICREGRESSOR_H
#define LOGISTICREGRESSOR_H

#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <cmath>
#include <locale>
#include <algorithm>
#include <iostream>

using namespace std;

class LogisticRegressor {
private:
    unordered_map<string, double> weights; // Веса для каждого слова
    double bias = 0.0; // Смещение
    double learning_rate = 0.1; // Скорость обучения
    int epochs = 1000; // Количество эпох

    static string stem(string& word) {
        // Приводим слово к нижнему регистру
        string normalized = word;
        transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);

        vector<string> s1 = {"а", "о", "ы", "и"};
        vector<string> s2 = {"ая", "ой", "ий", "ого", "ую", "ем", "ет"};

        cout << (normalized.substr(normalized.length() - 3)) << endl;

        if (normalized.length() >= 4) {
            for (const auto & i : s1) {
                if (normalized.substr(normalized.length() - 2) == i) {
                    normalized = normalized.substr(0, normalized.length() - 2);
                }
            }

            for (const auto & i : s2) {
                if (normalized.substr(normalized.length() - 3) == i) {
                    normalized = normalized.substr(0, normalized.length() - 3);
                }
            }
        }

        return normalized;
    }


    static string normalize(string& word) {
        ranges::transform(word, word.begin(), ::tolower);
        return stem(word);
    }

    static double sigmoid(const double z) {
        return 1.0 / (1.0 + exp(-z));
    }

public:
    void fit(vector<string>& texts, vector<double>& labels) {
        size_t n = texts.size();

        // Инициализация весов. Временная слодожность O(n * m)
        // где n - количество текстов, m - количество слов в тексте
        for (const auto& text : texts) {
            stringstream ss(text);
            string word;
            while (ss >> word) {
                weights[normalize(word)] = 0.0;
            }
        }

        // Обучение. Временная сложность O(n * m * epochs)
        // где n - количество текстов, m - количество слов в тексте, epochs - количество эпох
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < n; ++i) {
                double prediction = bias;
                stringstream ss(texts[i]);
                string word;
                while (ss >> word) {
                    word = normalize(word);
                    prediction += weights.count(word) ? weights[word] : 0.0;
                }

                double prob = sigmoid(prediction);

                double error = labels[i] - prob;
                bias += learning_rate * error;

                ss.clear();
                ss.str(texts[i]);
                while (ss >> word) {
                    word = normalize(word);
                    if (weights.count(word)) {
                        weights[word] += learning_rate * error;
                    }
                }
            }
        }
    }

    // Предсказывание. Временная сложность O(m)
    // где m - количество слов в тексте
    int predict(string& text) {
        double prediction = bias;
        stringstream ss(text);
        string word;
        while (ss >> word) {
            word = normalize(word);
            prediction += weights.count(word) ? weights[word] : 0.0;
        }

        return sigmoid(prediction) >= 0.5 ? 1 : 0;
    }
};

#endif //LOGISTICREGRESSOR_H
