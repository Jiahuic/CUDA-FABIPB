#ifndef DIRECT_BACKEND_H
#define DIRECT_BACKEND_H

struct ssystem;

int cpuDirectApply(struct ssystem *sys, double alpha, double beta,
                   const double *sgm, double *pot);

#endif
