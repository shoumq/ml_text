//
// Created by Mac on 26.12.2024.
//

#include "LogisticRegression.h"
#include <algorithm>
#include <sstream>
#include <cmath>


string LogisticRegression::removeSuffixes(string &word, const vector<string> &suffixes) {
    for (const auto &suffix: suffixes) {
        if (word.length() >= suffix.length() &&
            word.compare(word.length() - suffix.length(), suffix.length(), suffix) == 0) {
            return word.substr(0, word.length() - suffix.length());
        }
    }
    return word;
}

string LogisticRegression::normalize(string &word) {
    string normalized = word;
    transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);

    vector<string> suffixes = {
        "а", "о", "ы", "и", "е", "у", "ё", "и",
        "ая", "ой", "ий", "ую", "ие", "ых", "ом", "его", "ему", "ою"
    };

    normalized = removeSuffixes(normalized, suffixes);

    return normalized.empty() ? "0" : normalized;
}

double LogisticRegression::sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

void LogisticRegression::fit(vector<string> &texts, vector<double> &labels) {
    size_t n = texts.size();

    // Инициализация весов
    for (auto &text: texts) {
        stringstream ss(text);
        string word;
        while (ss >> word) {
            weights[normalize(word)] = 0.0;
        }
    }

    // Обучение
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
                    weights[word] += learning_rate * (error + lambda * weights[word]);
                }
            }
        }
    }
}

int LogisticRegression::predict(const string &text) {
    double prediction = bias;
    stringstream ss(text);
    string word;
    while (ss >> word) {
        word = normalize(word);
        prediction += weights.count(word) ? weights[word] : 0.0;
    }

    return sigmoid(prediction) >= 0.5 ? 1 : 0;
}
