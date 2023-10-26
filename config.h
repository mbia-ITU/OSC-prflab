/*********************************************************
 * config.h - Configuration data for the driver.c program.
 *********************************************************/
#ifndef _CONFIG_H_
#define _CONFIG_H_

#if   defined(__O1__)
#define B64    14.5
#define B128   14.3
#define B256   14.2
#define B512   14.2
#define B1024  14.4
#define B2048  15.0
#define B4096  14.9
#define B8192  14.8
#define R64    3.0
#define R128   8.5
#define R256   11.0
#define R512   11.6
#define R1024  11.8
#define R2048  17.5
#define R4096  30.9
#define R8192  44.5
#define S64    128.9
#define S128   129.7
#define S256   130.0
#define S512   131.5
#define S1024  132.4
#define S2048  133.7
#define S4096  132.7
#define S8192  133.1

#elif defined(__O2__)
#define B64    14.5
#define B128   14.4
#define B256   14.3
#define B512   14.3
#define B1024  14.5
#define B2048  14.8
#define B4096  15.0
#define B8192  14.8
#define R64    3.0
#define R128   8.4
#define R256   11.1
#define R512   11.6
#define R1024  11.8
#define R2048  17.7
#define R4096  31.2
#define R8192  44.4
#define S64    68.2
#define S128   68.1
#define S256   68.2
#define S512   69.2
#define S1024  69.8
#define S2048  70.5
#define S4096  70.3
#define S8192  70.3

#elif defined(__O3__)
#define B64    14.6
#define B128   14.4
#define B256   14.3
#define B512   14.2
#define B1024  14.3
#define B2048  14.8
#define B4096  14.8
#define B8192  14.9
#define R64    3.2
#define R128   8.5
#define R256   10.8
#define R512   11.4
#define R1024  12.3
#define R2048  17.9
#define R4096  31.2
#define R8192  42.9
#define S64    86.5
#define S128   86.9
#define S256   87.6
#define S512   91.2
#define S1024  91.8
#define S2048  91.6
#define S4096  91.5
#define S8192  92.5

#else // O0 assumed
#define B64    33.0
#define B128   32.9
#define B256   32.7
#define B512   33.0
#define B1024  33.1
#define B2048  33.8
#define B4096  33.6
#define B8192  33.9
#define R64    9.4
#define R128   17.5
#define R256   46.1
#define R512   51.7
#define R1024  64.7
#define R2048  81.2
#define R4096  84.2
#define R8192  241.1
#define S64    313.2
#define S128   315.3
#define S256   318.5
#define S512   324.9
#define S1024  326.1
#define S2048  325.6
#define S4096  325.6
#define S8192  326.6

#endif /* __On__ */

#endif /* _CONFIG_H_ */
