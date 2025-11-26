#include <stdio.h>

#include "../rustuna.h"

int main(void) {
  TunaTrial *trial;
  double x;
  double values[1];

  TunaDirection direction[1] = {TunaDirectionMinimize};
  TunaSampler *sampler = tuna_new_tpe_sampler();
  TunaStudy *study = tuna_create_study("test_study", *sampler, direction, 1);

  for (int i = 0; i < 30; i++) {
    trial = tuna_ask(study);
    if (trial == NULL) {
      return 1;
    }
    tuna_suggest_float(trial, "x1", -10.0, 10.0, &x);
    values[0] = x * x;
    tuna_tell(study, trial->number, values, 1);
    printf("  value=%f (x1 = %f)\n", values[0], x);
  }
  return 0;
}