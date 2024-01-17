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
 *  $Id: problem.h,v 1.13 2022/11/17 00:54:54 tanaka Exp tanaka $
 *  $Revision: 1.13 $
 *  $Date: 2022/11/17 00:54:54 $
 *  $Author: tanaka $
 *
 */
#pragma once
#include "define.h"

typedef struct {
  int s;
  int t;
} coordinate_t;

typedef struct {
  int no;
  int priority;
} block_t;

typedef struct {
  int n_block;
  int n_stack;
  int s_height;
  int max_priority;
  coordinate_t *position;
  int *priority;
  int *n_tier;
  block_t **block;
  double time;
  double total_time;
#ifdef _OPENMP
  double stime;
  double total_stime;
#endif /* _OPENMP */
} problem_t;

extern uchar verbose;
extern uchar lower_bound_only;
extern uchar backtrack;
extern int tlimit;
extern int n_thread;

problem_t *create_problem(int, int, int);
void free_problem(problem_t *);

