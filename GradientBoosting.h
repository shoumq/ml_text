//
// Created by Mac on 26.12.2024.
//

#ifndef GRADIENTBOOSTING_H
#define GRADIENTBOOSTING_H

#include <vector>

using namespace std;

class DecisionTree {
public:
    double predict(vector<double>& features) {
        return features[0] > 0.5 ? 1.0 : 0.0;
    }
};

class GradientBoosting {
private:
    vector<DecisionTree> models;
    double learning_rate = 0.1;

public:
    void fit(vector<vector<double>>& X, const vector<double>& y, int n_estimators) {
        size_t n = y.size();
        vector<double> predictions(n, 0.0);

        for (int i = 0; i < n_estimators; ++i) {
            // Вычисляем остатки
            vector<double> residuals(n);
            for (size_t j = 0; j < n; ++j) {
                residuals[j] = y[j] - predictions[j];
            }

            // Обучаем новое дерево на остатках
            DecisionTree tree;
            models.push_back(tree);

            // Обновляем предсказания
            for (size_t j = 0; j < n; ++j) {
                predictions[j] += learning_rate * tree.predict(X[j]);
            }
        }
    }

    int predict(vector<double> &features) {
        double final_prediction = 0.0;
        for (auto& model : models) {
            final_prediction += model.predict(features);
        }
        return final_prediction >= (models.size() / 2.0) ? 1 : 0;
    }
};



#endif //GRADIENTBOOSTING_H
