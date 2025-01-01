#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "LogisticRegression.h"
#include <iostream>
#include <chrono>
// #include "crow_all.h"

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

int main() {
    setlocale(LC_CTYPE, "ru_RU.UTF-8");

    auto start = std::chrono::high_resolution_clock::now();
    vector<string> texts;
    vector<double> labels;

    readCSV("../data.csv", texts, labels);

    LogisticRegression model;
    model.fit(texts, labels);

    // crow::SimpleApp app;
    //
    // CROW_ROUTE(app, "/predict/<string>")
    // ([&model](const crow::request& req, const string &text) {
    //     double prediction = model.predict(text);
    //     return crow::response{to_string(prediction)};
    // });
    //
    // app.port(18080).multithreaded().run();

    if (const char* varEnv = getenv("VAR_TEXT")) {
        string t = varEnv;
        cout << model.predict(t)<< endl;
    } else {
        cout << "No predicted data" << endl;
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "Время выполнения: " << duration.count() << " секунд" << std::endl;

    return 0;
}