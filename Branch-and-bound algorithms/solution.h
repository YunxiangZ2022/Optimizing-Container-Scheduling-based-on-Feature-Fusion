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
 *  $Id: solution.h,v 1.13 2022/11/17 00:54:54 tanaka Exp tanaka $
 *  $Revision: 1.13 $
 *  $Date: 2022/11/17 00:54:54 $
 *  $Author: tanaka $
 *
 */
#pragma once
#include "define.h"
#include "problem.h"

typedef struct {
  int src;
  int dst;
  block_t block;
} relocation_t;

typedef struct {
  int n_relocation;
  int n_block;
  relocation_t *relocation;
} solution_t;

typedef struct {
  block_t block;
  int n_stacked;
  int min_priority;
} bay_state_t;

typedef struct {
  int no;
  int n_tier;
  int n_stacked;
  int min_priority;
  int last_modified;
} stack_state_t;

typedef struct {
  int n_block;
  int n_bblock;
  int n_nonfull_stack;
  int *last_relocation;
  bay_state_t **bay;
  stack_state_t *sstate;
} state_t;

solution_t *create_solution(void);
void free_solution(solution_t *);
void copy_solution(solution_t *, solution_t *);
void add_relocation(solution_t *, int, int, block_t *);
state_t *create_state(problem_t *);
state_t *initialize_state(problem_t *, state_t *);
void copy_state(problem_t *, state_t *, state_t *);
state_t *duplicate_state(problem_t *, state_t *);
void free_state(state_t *);
uchar retrieve_all_blocks(problem_t *, state_t *, solution_t *);
int stack_state_comp(const void *, const void *);
