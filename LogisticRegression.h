//
// Created by Mac on 26.12.2024.
//

#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include <unordered_map>
#include <vector>
#include <string>
#include <locale>
#include <unordered_set>

using namespace std;

class LogisticRegression {
private:
    unordered_map<string, double> weights;
    double bias = 0.0;
    double learning_rate = 0.01;
    int epochs = 100;
    int lambda = 0.3;

    static string stem(const string &word);

    static string normalize(string &word);

    static double sigmoid(double z);

    static string removeSuffixes(string &word, const vector<string> &suffixes);

public:
    void fit(vector<string> &texts, vector<double> &labels);

    int predict(const string &text);
};

#endif //LOGISTICREGRESSION_H
