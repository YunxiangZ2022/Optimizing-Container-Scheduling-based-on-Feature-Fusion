/*
 * Copyright 2014-2022 Shunji Tanaka.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   1. Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *   2. Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials
 *      provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *  $Id: timer.c,v 1.11 2022/11/17 00:54:54 tanaka Exp tanaka $
 *  $Revision: 1.11 $
 *  $Date: 2022/11/17 00:54:54 $
 *  $Author: tanaka $
 *
 */
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */
#ifndef _MSC_VER
#include <unistd.h>
#include <string.h>
#endif /* !_MSC_VER */

#include "define.h"
#include "timer.h"
#include "problem.h"

#ifdef USE_CLOCK
#include <time.h>
#else  /* !USE_CLOCK */
#include <sys/time.h>
#include <sys/resource.h>
#endif /* !USE_CLOCK */

#define INCLUDE_SYSTEM_TIME

void timer_start(problem_t *problem)
{
#ifdef _OPENMP
  problem->stime = (double) omp_get_wtime();
#ifndef _MSC_VER
#ifdef USE_CLOCK
  problem->total_stime = (double) clock()/(double) CLOCKS_PER_SEC;
#else  /* !USE_CLOCK */
  struct rusage ru;

  getrusage(RUSAGE_SELF, &ru);
  problem->total_stime = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec / 1000000.0;
#ifdef INCLUDE_SYSTEM_TIME
  problem->total_stime += ru.ru_stime.tv_sec + ru.ru_stime.tv_usec / 1000000.0;
#endif /* INCLUDE_SYSTEM_TIME */
#endif /* USE_CLOCK */
#endif /* _MSC_VER */
#else /* !_OPENMP */
#ifdef USE_CLOCK
  problem->stime = (double) clock()/(double) CLOCKS_PER_SEC;
#else  /* !USE_CLOCK */
  struct rusage ru;

  getrusage(RUSAGE_SELF, &ru);
  problem->stime = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec / 1000000.0;
#ifdef INCLUDE_SYSTEM_TIME
  problem->stime += ru.ru_stime.tv_sec + ru.ru_stime.tv_usec / 1000000.0;
#endif /* INCLUDE_SYSTEM_TIME */
#endif /* USE_CLOCK */
#endif /* !_OPENMP */
}

void timer_stop(problem_t *problem)
{
#ifdef _OPENMP
  problem->total_time = get_total_time(problem);
#endif /* _OPENMP */
  problem->time = get_time(problem);
}

double get_time(problem_t *problem)
{
  double cpu_time;

#ifdef _OPENMP
  cpu_time = (double) omp_get_wtime();
#else /* !_OPENMP */
#ifdef USE_CLOCK
  cpu_time = (double) clock()/(double) CLOCKS_PER_SEC;
#else  /* !USE_CLOCK */
  struct rusage ru;

  getrusage(RUSAGE_SELF, &ru);
  cpu_time = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec / 1000000.0;
#ifdef INCLUDE_SYSTEM_TIME
  cpu_time += ru.ru_stime.tv_sec + ru.ru_stime.tv_usec / 1000000.0;
#endif /* INCLUDE_SYSTEM_TIME */
#endif /* !USE_CLOCK */
#endif /* !_OPENMP */
  cpu_time -= problem->stime;

  if(cpu_time < 0.0) {
    cpu_time = 0.0;
  }

  return(cpu_time);
}

#ifdef _OPENMP
double get_total_time(problem_t *problem)
{
  double total_time;

#ifdef USE_CLOCK
  total_time = (double) clock()/(double) CLOCKS_PER_SEC;
#else  /* !USE_CLOCK */
  struct rusage ru;

  getrusage(RUSAGE_SELF, &ru);
  total_time = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec / 1000000.0;
#ifdef INCLUDE_SYSTEM_TIME
  total_time += ru.ru_stime.tv_sec + ru.ru_stime.tv_usec / 1000000.0;
#endif /* INCLUDE_SYSTEM_TIME */
#endif /* !USE_CLOCK */
  total_time -= problem->total_stime;

  if(total_time < 0.0) {
    total_time = 0.0;
  }

  return(total_time);
}
#endif /* _OPENMP */
