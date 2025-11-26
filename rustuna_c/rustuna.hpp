#pragma once

#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <ostream>
#include <string>
#include <vector>

extern "C" {
#include "rustuna.h"
}

namespace rustuna {
class Trial {
public:
  TunaTrial *trial;

  Trial(TunaTrial *trial) : trial(trial) {}

  double suggest_float(const std::string &name, const double &low,
                       const double &high) {
    double value;
    tuna_suggest_float(trial, name.c_str(), low, high, &value);
    return value;
  }

  int suggest_int(const std::string &name, const int &low, const int &high) {
    int value;
    tuna_suggest_int(trial, name.c_str(), low, high, &value);
    return value;
  }

  std::string suggest_categorical(const std::string &name,
                                  std::vector<std::string> choices) {
    u_int index;

    std::vector<const char *> cstrings;
    for (const auto &str : choices) {
      cstrings.push_back(str.c_str());
    }
    tuna_suggest_categorical(trial, name.c_str(), cstrings.data(),
                             choices.size(), &index);
    return choices[index];
  }
};

class Study {
  const std::string study_name;
  const std::vector<TunaDirection> directions;
  TunaStudy *study;

public:
  Study(const std::string &study_name,
        const std::vector<TunaDirection> directions = {TunaDirectionMinimize})
      : study_name(study_name), directions(directions) {
    // TODO(c-bata): Make it able to change the sampler.
    TunaSampler *sampler = tuna_new_tpe_sampler();
    study = tuna_create_study("test_study", *sampler, directions.data(), 1);
  }

  Trial ask() {
    TunaTrial *trial = tuna_ask(study);
    return Trial(trial);
  }

  void tell(const Trial &trial, std::vector<double> values) {
    tuna_tell(study, trial.trial->number, values.data(), values.size());
  }

  /*
    TODO(c-bata): Implement me.
    FrozenTrial best_trial() {
      return FrozenTrial(...);
    }

    std::vector<FrozenTrial> trials() {
    }*/
};

} // namespace rustuna
