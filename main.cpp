#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <cmath>
#include <locale>
#include <algorithm>

using namespace std;

class LogisticRegressor {
private:
    unordered_map<string, double> weights; // Веса для каждого слова
    double bias = 0.0; // Смещение
    double learning_rate = 0.1; // Скорость обучения
    int epochs = 1000; // Количество эпох

    static string stem(string word) {
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
        return word;
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

        for (const auto& text : texts) {
            stringstream ss(text);
            string word;
            while (ss >> word) {
                weights[normalize(word)] = 0.0;
            }
        }

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

int main() {
    system("chcp 65001");
    setlocale(LC_CTYPE, "ru_RU.UTF-8");

    // vector<string> texts;
    // vector<double> labels;
    //
    // readCSV("./data.csv", texts, labels);
    //
    // LogisticRegressor model;
    // model.fit(texts, labels);
    //
    // if (argc > 1) {
    //     string t = argv[1];
    //     cout << argv[1] << endl;
    // } else {
    //     cout << "Arg is empty.";
    // }

    system("pause");
    return 0;
}