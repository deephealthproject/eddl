/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: Università degli studi di Modena e Reggio Emilia (federico.bolelli@unimore.it)
* All rights reserved
*/

#ifndef EDDL_SYTEM_INFO_H_
#define EDDL_SYTEM_INFO_H_

#if _WIN32 || _WIN64 || WIN32 || __WIN32__ || __WINDOWS__ || __TOS_WIN__
#define EDDL_WINDOWS
#elif  __gnu_linux__ || __linux__
#define EDDL_LINUX
#elif  __unix || __unix__
#define EDDL_UNIX
#elif __APPLE__ || __MACH__ || macintosh || Macintosh || (__APPLE__ && __MACH__)
#define EDDL_APPLE
#endif

#endif // EDDL_SYTEM_INFO_H_
