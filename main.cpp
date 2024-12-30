#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include "LogisticRegressor.h"
#include "GradientBoosting.h"

using namespace std;

void readCSV(const string& filename, vector<string>& texts, vector<double>& labels) {
    ifstream file(filename);
    string line;

    if (!file.is_open()) {
        cerr << "File error: " << filename << endl;
        return;
    }

    getline(file, line);

    while (getline(file, line)) {
        stringstream ss(line);
        string text;
        string labelStr;

        getline(ss, text, ',');
        getline(ss, labelStr, ',');

        texts.push_back(text);
        labels.push_back(stod(labelStr));
    }
}

int main(int argc, char* argv[]) {
    setlocale(LC_CTYPE, "ru_RU.UTF-8");

    vector<string> texts;
    vector<double> labels;

    readCSV("../data.csv", texts, labels);

    LogisticRegressor model;
    model.fit(texts, labels);

    GradientBoosting gb_model;

    // Преобразуем текстовые данные в векторные (в данном случае просто создадим векторы из весов)
    vector<vector<double>> feature_vectors(labels.size(), vector<double>(1));

    for (size_t i = 0; i < labels.size(); ++i) {
        feature_vectors[i][0] = model.predict(texts[i]); // Используем логистическую регрессию для получения признаков
    }

    gb_model.fit(feature_vectors, labels, 10); // Обучаем градиентный бустинг

    if (const char* varEnv = getenv("VAR_TEXT")) {
        string t = varEnv;
        vector<double> test_features(1);
        test_features[0] = model.predict(t); // Получаем вектор признаков для тестовой строки
        cout << gb_model.predict(test_features) << endl;
    } else {
        cout << "No predicted data" << endl;
    }

    return 0;
}