#ifndef _PRECISION_H_
#define _PRECISION_H_

#include <stdint.h>

// Define DOUBLE_PRECISION to use double, otherwise use float
#ifdef DOUBLE_PRECISION
#define FLOAT double
#else
#define FLOAT float
#endif

#endif /* _PRECISION_H_ */