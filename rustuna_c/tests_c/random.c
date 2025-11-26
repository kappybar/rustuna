#include <stdio.h>

#include "../rustuna.h"

int main(void) {
  TunaTrial *trial;
  uint32_t x;
  double values[1];
  const char *choices[] = {
      "foo",
      "bar",
  };

  TunaDirection direction[1] = {TunaDirectionMinimize};
  TunaSampler *sampler = tuna_new_tpe_sampler();
  TunaStudy *study = tuna_create_study("test_study", *sampler, direction, 1);
  for (int i = 0; i < 15; i++) {
    trial = tuna_ask(study);
    if (trial == NULL) {
      return 1;
    }
    tuna_suggest_categorical(trial, "x2", choices, 2, &x);
    if (x == 0) {
      values[0] = 0.0;
    } else {
      values[0] = 1.0;
    }
    tuna_tell(study, trial->number, values, 1);
    printf("  value=%f (x = %s)\n", values[0], choices[x]);
  }
  return 0;
}