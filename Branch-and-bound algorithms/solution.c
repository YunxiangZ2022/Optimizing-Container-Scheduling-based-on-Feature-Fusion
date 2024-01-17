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
 *  $Id: solution.c,v 1.17 2022/11/17 00:54:54 tanaka Exp tanaka $
 *  $Revision: 1.17 $
 *  $Date: 2022/11/17 00:54:54 $
 *  $Author: tanaka $
 *
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "define.h"
#include "solution.h"

solution_t *create_solution(void)
{
  return((solution_t *) calloc((size_t) 1, sizeof(solution_t)));
}

void free_solution(solution_t *solution)
{
  if(solution != NULL) {
    free(solution->relocation);
    free(solution);
  }
}

void copy_solution(solution_t *dst, solution_t *src)
{
  if(dst != NULL && src != NULL) {
    if(dst->n_block < src->n_relocation) {
      dst->n_block = src->n_block;
      dst->relocation
	= (relocation_t *) realloc((void *) dst->relocation,
				   (size_t) dst->n_block
				   *sizeof(relocation_t));
    }
    dst->n_relocation = src->n_relocation;
    memcpy((void *) dst->relocation, (void *) src->relocation,
	   (size_t) src->n_relocation*sizeof(relocation_t));
  }
}

void add_relocation(solution_t *solution, int src, int dst, block_t *block)
{
  if(solution != NULL) {
    if(solution->n_block <= solution->n_relocation) {
      solution->n_block += 100;
      solution->relocation
	= (relocation_t *) realloc((void *) solution->relocation,
				   (size_t) solution->n_block
				   *sizeof(relocation_t));
    }
    solution->relocation[solution->n_relocation].src = src;
    solution->relocation[solution->n_relocation].dst = dst;
    solution->relocation[solution->n_relocation++].block = *block;
  }
}

state_t *create_state(problem_t *problem)
{
  int i;
  state_t *state = (state_t *) malloc(sizeof(state_t));
  
  state->n_block = 0;
  state->n_bblock = 0;
  state->n_nonfull_stack = problem->n_stack;
  state->last_relocation
    = (int *) malloc((size_t) problem->n_block*sizeof(int));

  state->sstate = (stack_state_t *) malloc((size_t) problem->n_stack
					   *sizeof(stack_state_t));
  state->bay = (bay_state_t **) malloc((size_t) problem->n_stack
				       *sizeof(bay_state_t *));
  state->bay[0] = (bay_state_t *) malloc((size_t) problem->n_stack
					 *(problem->s_height + 1)
					 *sizeof(bay_state_t));

  for(i = 1; i < problem->n_stack; ++i) {
    state->bay[i] = state->bay[i - 1] + (problem->s_height + 1);
  }

  return(state);
}

state_t *initialize_state(problem_t *problem, state_t *state)
{
  int i, j;
  state_t *nstate = (state == NULL)?create_state(problem):state;

  nstate->n_block = problem->n_block;
  nstate->n_bblock = 0;

  memset((void *) nstate->last_relocation, 0,
	 (size_t) problem->n_block*sizeof(int));
  memset((void *) nstate->bay[0], 0,
	 (size_t) problem->n_stack*(problem->s_height + 1)*sizeof(bay_state_t));

  nstate->n_nonfull_stack = 0;
  for(i = 0; i < problem->n_stack; ++i) {
    stack_state_t *csstate = nstate->sstate + i;
    bay_state_t *bay = nstate->bay[i];

    bay[0].min_priority = problem->max_priority + 1;
    bay[0].n_stacked = 0;
    for(j = 1; j <= problem->n_tier[i]; ++j) {
      bay[j].block = problem->block[i][j - 1];
      if(bay[j].block.priority > bay[j - 1].min_priority) {
	bay[j].min_priority = bay[j - 1].min_priority;
	bay[j].n_stacked = bay[j - 1].n_stacked + 1;
	++nstate->n_bblock;
      } else {
	bay[j].min_priority = bay[j].block.priority;
	bay[j].n_stacked = 0;
      }
    }

    csstate->no = i;
    csstate->n_tier = problem->n_tier[i];
    csstate->n_stacked = bay[csstate->n_tier].n_stacked;
    csstate->min_priority = bay[csstate->n_tier].min_priority;
    csstate->last_modified = 0;

    if(csstate->n_tier < problem->s_height) {
      ++nstate->n_nonfull_stack;
    }
  }

  qsort((void *) nstate->sstate, problem->n_stack, sizeof(stack_state_t),
	stack_state_comp);

  for(i = problem->n_stack - 1; i > 0; --i) {
    if(nstate->sstate[i].n_tier == problem->s_height) {
      stack_state_t sstate = nstate->sstate[i];

      for(j = i; j < problem->n_stack - 1
	    && nstate->sstate[j + 1].n_tier < problem->s_height; ++j) {
	nstate->sstate[j] = nstate->sstate[j + 1];
      }
      nstate->sstate[j] = sstate;
    }
  }

  if(nstate->sstate[0].n_tier == problem->s_height) {
    ++nstate->n_nonfull_stack;
  }

  return(nstate);
}

void copy_state(problem_t *problem, state_t *dst, state_t *src)
{
  dst->n_block = src->n_block;
  dst->n_nonfull_stack = src->n_nonfull_stack;
  dst->n_bblock = src->n_bblock;

  memcpy((void *) dst->last_relocation, (void *) src->last_relocation,
	 (size_t) problem->n_block*sizeof(int));
  memcpy((void *) dst->bay[0], (void *) src->bay[0],
	 (size_t) problem->n_stack*(problem->s_height + 1)
	 *sizeof(bay_state_t));
  memcpy((void *) dst->sstate, (void *) src->sstate,
	 (size_t) problem->n_stack*sizeof(stack_state_t));
}

state_t *duplicate_state(problem_t *problem, state_t *state)
{
  state_t *nstate = create_state(problem);
  copy_state(problem, nstate, state);
  return(nstate);
}

void free_state(state_t *state)
{
  if(state != NULL) {
    free(state->sstate);
    free(state->bay[0]);
    free(state->bay);
    free(state->last_relocation);
    free(state);
  }
}

uchar retrieve_all_blocks(problem_t *problem, state_t *state,
			  solution_t *solution)
{
  int i;
  stack_state_t *sstate = state->sstate, current;

  while(state->n_block > 0 && sstate[0].n_stacked == 0) {
    int src_stack = sstate[0].no;
    int n_tier = --sstate[0].n_tier;

    --state->n_block;
    sstate[0].min_priority = state->bay[src_stack][n_tier].min_priority;
    sstate[0].n_stacked = state->bay[src_stack][n_tier].n_stacked;

    if(solution != NULL) {
      int depth
	= state->last_relocation[state->bay[src_stack][n_tier + 1].block.no];

      if(depth > 0) {
	int j;

	for(j = state->n_nonfull_stack - 1; 
	    j > 0 && sstate[j].min_priority > sstate[0].min_priority; --j) {
	  if(sstate[j].n_tier <= sstate[0].n_tier
	     && sstate[j].last_modified < depth) {
	    return(True);
	  }
	}
	for(; j > 0; --j) {
	  if(sstate[j].n_tier < sstate[0].n_tier
	     && sstate[j].last_modified < depth) {
	    return(True);
	  }
	}
      }

      sstate[0].last_modified = solution->n_relocation;
    }

    current = sstate[0];
    for(i = 0; i < state->n_nonfull_stack - 1
	  && stack_state_comp((void *) &(sstate[i + 1]), (void *) &current) < 0;
	++i) {
      sstate[i] = sstate[i + 1];
    }
    sstate[i] = current;

    if(state->n_nonfull_stack < problem->n_stack) {
      if(sstate[0].min_priority > sstate[state->n_nonfull_stack].min_priority) {
	current = sstate[state->n_nonfull_stack];
	for(i = state->n_nonfull_stack; i > 0; --i) {
	  sstate[i] = sstate[i - 1];
	}
	sstate[0] = current;
	++state->n_nonfull_stack;
      }
    }
  }

  return(False);
}

int stack_state_comp(const void *a, const void *b)
{
  stack_state_t *x = (stack_state_t *) a;
  stack_state_t *y = (stack_state_t *) b;

  if(x->min_priority > y->min_priority) {
    return(1);
  } else if(x->min_priority < y->min_priority) {
    return(-1);
  }

  return(0);
}
