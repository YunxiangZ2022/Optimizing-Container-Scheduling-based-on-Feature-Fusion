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
 *  $Id: heuristics.c,v 1.17 2022/11/17 00:54:54 tanaka Exp tanaka $
 *  $Revision: 1.17 $
 *  $Date: 2022/11/17 00:54:54 $
 *  $Author: tanaka $
 *
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "define.h"
#include "heuristics.h"
#include "print.h"
#include "problem.h"
#include "solution.h"

solution_t *greedy_heuristic(problem_t *problem, state_t *state,
			     solution_t *solution, int ub)
{
  int i;
  int n_block, lb;
  bay_state_t **bay;
  stack_state_t *sstate;

#if 0
  printf("greedy_heuristic\n");
#endif

  bay = state->bay;
  sstate = state->sstate;

  if(solution == NULL) {
    solution = create_solution();
  }

  lb = state->n_bblock + solution->n_relocation;

  for(n_block = state->n_block; n_block > 0; --n_block) {
    int src_stack = sstate[0].no, dst_stack;
    stack_state_t current;
    block_t cblock;

    for(; sstate[0].n_stacked > 0; --sstate[0].n_stacked) {
#if 0
      print_state(problem, state, stdout);
      for(i = 0; i < problem->n_stack; ++i) {
	printf("[%d:%d:%d:%d]", sstate[i].no, sstate[i].n_tier,
	       sstate[i].n_stacked, sstate[i].min_priority);
      }
      printf("\n");
#endif
      cblock = bay[src_stack][sstate[0].n_tier--].block;

      i = state->n_nonfull_stack - 1;
      if(i <= 0){
	fprintf(stderr, "No stack found.\n");
	exit(1);
      }

      if(sstate[i].min_priority >= cblock.priority) {
	for(i = 1; i < state->n_nonfull_stack
	      && sstate[i].min_priority < cblock.priority; ++i);
	sstate[i].min_priority = cblock.priority;
	sstate[i].n_stacked = 0;
      } else if(++lb >= ub) {
	solution->n_relocation = ub;
	return(solution);
      } else {
	if(sstate[i].n_tier + 1 == problem->s_height && i > 1) {
	  --i;
	}
	++sstate[i].n_stacked;
      }

      dst_stack = sstate[i].no;
      add_relocation(solution, src_stack, dst_stack, &cblock);

      bay[dst_stack][++sstate[i].n_tier].block = cblock;
      bay[dst_stack][sstate[i].n_tier].min_priority = sstate[i].min_priority;
      bay[dst_stack][sstate[i].n_tier].n_stacked = sstate[i].n_stacked;

      current = sstate[i];
      if(current.n_tier == problem->s_height) {
	for(; i < state->n_nonfull_stack - 1; ++i) {
	  sstate[i] = sstate[i + 1];
	}
	for(; i < problem->n_stack - 1
	      && stack_state_comp((void *) &(sstate[i + 1]),
				  (void *) &current) < 0; ++i) {
	  sstate[i] = sstate[i + 1];
	}
	sstate[i] = current;
	--state->n_nonfull_stack;
      } else if(current.n_stacked == 0) {
	for(; i > 1 && stack_state_comp((void *) &(sstate[i - 1]),
					(void *) &current) > 0; --i) {
	  sstate[i] = sstate[i - 1];
	}
	sstate[i] = current;
      }
    }

    sstate[0].min_priority = bay[src_stack][--sstate[0].n_tier].min_priority;
    sstate[0].n_stacked = bay[src_stack][sstate[0].n_tier].n_stacked;

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

#if 0
    print_state(problem, state, stdout);
    for(i = 0; i < problem->n_stack; ++i) {
      printf("[%d:%d:%d:%d]", sstate[i].no, sstate[i].n_tier,
	     sstate[i].n_stacked, sstate[i].min_priority);
    }
    printf("\n");
#endif
  }

  return(solution);
}
