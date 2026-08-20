#ifndef FABIPB_SYSTEM_H
#define FABIPB_SYSTEM_H

#if defined(_WIN32)
#include <windows.h>
#else
#include <stddef.h>
#include <unistd.h>
#endif

#if defined(__APPLE__)
int sysctlbyname(const char *name, void *oldp, size_t *oldlenp,
                 void *newp, size_t newlen);
#endif

static inline int fabipb_online_cpu_count(void) {
#if defined(_WIN32)
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  return (info.dwNumberOfProcessors > 0) ? (int)info.dwNumberOfProcessors : 1;
#elif defined(_SC_NPROCESSORS_ONLN)
  long count = sysconf(_SC_NPROCESSORS_ONLN);
  return (count > 0) ? (int)count : 1;
#elif defined(__APPLE__)
  int count = 0;
  size_t size = sizeof(count);
  if (sysctlbyname("hw.logicalcpu", &count, &size, NULL, 0) == 0 && count > 0) {
    return count;
  }
  return 1;
#else
  return 1;
#endif
}

#endif
