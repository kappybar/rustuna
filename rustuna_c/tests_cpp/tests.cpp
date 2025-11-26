#include <cmath>
#include <iostream>

#include "../rustuna.hpp"

int main() {
  rustuna::Study study("test_study", {TunaDirectionMinimize});
  for (int i = 0; i < 100; i++) {
    rustuna::Trial trial = study.ask();
    double x = trial.suggest_float("x", -10.0, 10.0);
    int y = trial.suggest_int("y", -10, 10);
    std::string z = trial.suggest_categorical("z", {"foo", "bar", "baz"});

    double objective = pow(x - 3, 2) + pow(y + 5, 2);
    study.tell(trial, {objective});

    std::cout << i << ": " << objective << " (x=" << x << ", y=" << y
              << ", z=" << z << ")" << std::endl;
  }
  return 0;
}
