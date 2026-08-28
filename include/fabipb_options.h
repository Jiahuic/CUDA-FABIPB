#ifndef FABIPB_OPTIONS_H
#define FABIPB_OPTIONS_H

#include <stddef.h>

struct ssystem;

void fabipb_set_benchmark_thread_defaults(void);
int fabipb_missing_external_thread_env(void);
void fabipb_ensure_startup_thread_env(int nargs, char *argv[]);

unsigned long long fabipb_parse_unsigned_long_long_env(const char *name,
                                                       unsigned long long defaultValue);
unsigned long long fabipb_get_max_direct_rhs_pairs(void);
int fabipb_allow_large_direct_rhs(void);
unsigned long long fabipb_count_direct_rhs_pairs(int nPnls, int nChar);
unsigned long long fabipb_get_active_direct_rhs_pair_limit(int gpuMode);
int fabipb_should_use_tree_rhs(int nPnls, int nChar, int gpuMode);
void fabipb_report_direct_rhs_limit(int nPnls, int nChar, int gpuMode);

void fabipb_resolve_auto_solver_policy(struct ssystem *sys, int q2mExplicit,
                                       int precondExplicit, int sepExplicit,
                                       int nPnls);
const char *fabipb_auto_or_explicit_label(int explicitFlag, int value,
                                          char *buf, size_t bufSize);

#endif
