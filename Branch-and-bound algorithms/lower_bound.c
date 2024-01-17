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
 *  $Id: lower_bound.c,v 1.9 2022/11/17 00:54:54 tanaka Exp tanaka $
 *  $Revision: 1.9 $
 *  $Date: 2022/11/17 00:54:54 $
 *  $Author: tanaka $
 *
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "define.h"
#include "lower_bound.h"
#include "print.h"
#include "problem.h"
#include "solution.h"

#if LOWER_BOUND == 4
static void lb_sub(problem_t *, int, int, int *, int, int, stack_state_t *,
		   int, int *);
#endif /* LOWER_BOUND == 4 */

#if LOWER_BOUND == 2
int lower_bound2(problem_t *problem, state_t *state)
{
  int i;
  int lb = state->n_bblock;
  int src_stack = state->sstate[0].no;
  int max_priority = state->sstate[state->n_nonfull_stack - 1].min_priority;

  for(i = 0; i < state->sstate[0].n_stacked; ++i) {
    if(max_priority
       < state->bay[src_stack][state->sstate[0].n_tier - i].block.priority) {
      ++lb;
    }
  }

  return(lb);
}
#elif LOWER_BOUND == 3
/* Zhu, W., Qin, H., Lim, A., Zhang, H. (2012). */
int lower_bound3(problem_t *problem, state_t *state, stack_state_t *csstate,
		 int *lbcache, uchar check_type, int check_priority, int ub)
{
  int i;
  int lb = state->n_bblock;
  int n_block = state->n_block, n_bblock = state->n_bblock;
  int n_nonfull = state->n_nonfull_stack;
  stack_state_t *sstate = state->sstate;

  memcpy((void *) csstate, (void *) sstate,
	 (size_t) problem->n_stack*sizeof(stack_state_t));
  state->sstate = csstate;

  for(i = check_priority + 1; i < problem->max_priority; ++i) {
    lb += lbcache[i];
  }

  while(state->n_block > 0 && csstate[0].min_priority <= check_priority) {
    int n_stacked = csstate[0].n_stacked;

    if(check_type != 3 || csstate[0].min_priority == check_priority) {
      int src_stack = csstate[0].no;
      int max_priority = csstate[state->n_nonfull_stack - 1].min_priority;
      int lb_dbblock = 0;
      bay_state_t *src_stackp
	= state->bay[src_stack] + (csstate[0].n_tier - n_stacked + 1);

      for(i = n_stacked - 1; i >= 0; --i) {
	if(max_priority < src_stackp[i].block.priority)  {
	  ++lb_dbblock;
	}
      }

      lb += lb_dbblock;
      lbcache[csstate[0].min_priority] = lb_dbblock;
    } else {
      if(check_type == 2) {
	check_type = 3;
      }
      lb += lbcache[csstate[0].min_priority];
    }

#if 1    
    if(lb >= ub) {
      break;
    }
#endif

    n_bblock -= n_stacked;
    if(n_bblock == 0) {
      break;
    }

    state->n_block -= n_stacked;
    csstate[0].n_tier -= n_stacked;
    csstate[0].n_stacked = 0;

    retrieve_all_blocks(problem, state, NULL);
  }

  state->n_block = n_block;
  state->n_nonfull_stack = n_nonfull;
  state->sstate = sstate;

  return(lb);
}
#elif LOWER_BOUND == 4
/* Tanaka, S., Takii, K. (2016). */
int lower_bound4(problem_t *problem, state_t *state, stack_state_t *csstate,
		 int *lbcache, int *pr, uchar check_type, int check_priority,
		 int ub)
{
  int i;
  int lb = state->n_bblock;
  int n_block = state->n_block, n_bblock = state->n_bblock;
  int n_nonfull = state->n_nonfull_stack;
  stack_state_t *sstate = state->sstate;

  memcpy((void *) csstate, (void *) sstate,
	 (size_t) problem->n_stack*sizeof(stack_state_t));
  state->sstate = csstate;

  for(i = check_priority + 1; i < problem->max_priority; ++i) {
    lb += lbcache[i];
  }

  while(state->n_block > 0 && csstate[0].min_priority <= check_priority) {
    int n_stacked = csstate[0].n_stacked;

    if(check_type != 3 || csstate[0].min_priority == check_priority) {
      int n_pr = 0;
      int src_stack = csstate[0].no;
      int max_priority = csstate[state->n_nonfull_stack - 1].min_priority;
      int min_priority = problem->max_priority;
      int lb_dbblock = 0, lb_inc = 0;
      bay_state_t *src_stackp
	= state->bay[src_stack] + (csstate[0].n_tier - n_stacked + 1);

      for(i = n_stacked - 1; i >= 0; --i) {
	int priority = src_stackp[i].block.priority;

	if(max_priority < priority) {
	  ++lb_dbblock;
	} else {
	  pr[n_pr++] = priority;
	  min_priority = min(min_priority, priority);
	}
      }

      lb += lb_dbblock;
#if 1
      if(lb >= ub) {
	break;
      }
#endif
      if(n_pr > 1) {
	int j;
	lb_inc = n_pr - 1;

	for(j = 1; j < state->n_nonfull_stack
	      && csstate[j].min_priority < min_priority; ++j);
	if(state->n_nonfull_stack - j > 0) {
	  lb_sub(problem, 0, n_pr, pr, j - 1, state->n_nonfull_stack - 1,
		 csstate + 1, 0, &lb_inc);
	}
      }

      lb += lb_inc;
      lbcache[csstate[0].min_priority] = lb_inc + lb_dbblock;
    } else {
      if(check_type == 2) {
	check_type = 3;
      }
      lb += lbcache[csstate[0].min_priority];
    }
#if 1
    if(lb >= ub) {
      break;
    }
#endif

    n_bblock -= n_stacked;
    if(n_bblock == 0) {
      break;
    }

    state->n_block -= n_stacked;
    csstate[0].n_tier -= n_stacked;
    csstate[0].n_stacked = 0;

    retrieve_all_blocks(problem, state, NULL);
  }

  state->n_block = n_block;
  state->n_nonfull_stack = n_nonfull;
  state->sstate = sstate;

  return(lb);
}

void lb_sub(problem_t *problem, int depth, int n, int *pr, int s,
	    int w, stack_state_t *sstate, int f, int *ub)
{
  int i;

  if(depth == n) {
    *ub = f;
    return;
  }

  for(i = s; i < w && sstate[i].min_priority < pr[depth]; ++i);
  if(i < w) {
    int prev_priority = sstate[i].min_priority;

    sstate[i].min_priority = pr[depth];
    lb_sub(problem, depth + 1, n, pr, s, w, sstate, f, ub);
    sstate[i].min_priority = prev_priority;

    if(*ub == 0) {
      return;
    }
  }

  if(i > 0) {
    if(++f < *ub) {
      lb_sub(problem, depth + 1, n, pr, s, w, sstate, f, ub);
    }
  }
}
#elif LOWER_BOUND == 5
/* Quispe, K.E.Y., Lintzmayer, C.N., Xavier, E.C. (2018), */
/* Jin, B. (2020). */
int lower_bound5(problem_t *problem, state_t *state, stack_state_t *csstate,
		 int *lb_cache, int *lis, uchar check_type, int check_priority,
		 int ub)
{
  int i, j;
  int lb = state->n_bblock;
  int n_block = state->n_block, n_bblock = state->n_bblock;
  int n_nonfull = state->n_nonfull_stack;
  stack_state_t *sstate = state->sstate;

  memcpy((void *) csstate, (void *) sstate,
	 (size_t) problem->n_stack*sizeof(stack_state_t));
  state->sstate = csstate;

  for(i = check_priority + 1; i < problem->max_priority; ++i) {
    lb += lb_cache[i];
  }

  while(csstate[0].min_priority <= check_priority) {
    int n_stacked = csstate[0].n_stacked;

    if(check_type != 3 || csstate[0].min_priority == check_priority) {
      int l = 0;
      int src_stack = csstate[0].no;
      int max_priority = csstate[state->n_nonfull_stack - 1].min_priority;
      int lb_dbblock = 0, lb_inc = 0;
      bay_state_t *src_stackp
	= state->bay[src_stack] + (csstate[0].n_tier - n_stacked + 1);

      lis[0] = problem->max_priority + 1;
      for(i = 0; i < n_stacked; ++i) {
	int priority = src_stackp[i].block.priority;

	if(priority <= max_priority) {
	  if(lis[l] > priority) {
	    lis[++l] = priority;
	  } else {
#if 1
	    for(j = 1; j <= l; ++j) {
	      if(lis[j] <= priority) {
		lis[j] = priority;
		break;
	      }
	    }
#else
	    /* binary search */
	    if(lis[1] <= priority) {
	      lis[1] = priority;
	    } else {
	      int left = 1, right = l;

	      while(right - left > 1) {
		j = (left + right)/2;
		if(lis[j] <= priority) {
		  right = j;
		} else {
		  left = j;
		}
	      }
	      lis[right] = priority;
	    }
#endif
	  }
	} else {
	  ++lb_dbblock;
	}
      }

      lb += lb_dbblock;
      if(lb >= ub) {
	break;
      }
    
      lb_inc = 0;
      for(i = 1, j = state->n_nonfull_stack - 1; i <= l; ++i) {
	if(lis[i] > csstate[j].min_priority) {
	  ++lb_inc;
	} else {
	  --j;
	}
      }

      lb += lb_inc;
      lb_cache[csstate[0].min_priority] = lb_inc + lb_dbblock;
    } else {
      if(check_type == 2) {
	check_type = 3;
      }
      lb += lb_cache[csstate[0].min_priority];
    }

#if 1
    if(lb >= ub) {
      break;
    }
#endif   

    n_bblock -= n_stacked;
    if(n_bblock == 0) {
      break;
    }

    state->n_block -= n_stacked;
    csstate[0].n_tier -= n_stacked;
    csstate[0].n_stacked = 0;

    retrieve_all_blocks(problem, state, NULL);
  }

  state->n_block = n_block;
  state->n_nonfull_stack = n_nonfull;
  state->sstate = sstate;

  return(lb);
}
#elif LOWER_BOUND == 6
/* Bacci, T., Mattia, S. Ventura, P. (2019). */
int lower_bound6(problem_t *problem, state_t *state, stack_state_t *csstate,
		 int *demand, int ub)
{
  int i, j;
  int lb = state->n_bblock;
  int n_block = state->n_block;
  int n_nonfull = state->n_nonfull_stack;
  stack_state_t *sstate = state->sstate;

  memcpy((void *) csstate, (void *) sstate,
	 (size_t) problem->n_stack*sizeof(stack_state_t));
  state->sstate = csstate;

  while(state->n_block > 0) {
    int src_stack = csstate[0].no;
    int n_stacked = csstate[0].n_stacked;
    int max_priority = csstate[state->n_nonfull_stack - 1].min_priority;
    int cumulative_demand = 0, cumulative_supply = 0;
    int lb_add = 0;
    bay_state_t *src_stackp
      = state->bay[src_stack] + (csstate[0].n_tier - n_stacked + 1);

    memset((void *) demand, 0, (size_t) problem->n_stack*sizeof(int));

    csstate[0].n_tier -= n_stacked;

    for(i = 0; i < n_stacked; ++i) {
      int priority = src_stackp[i].block.priority;

      if(priority > max_priority) {
	++lb;
      } else {
	for(j = 1; j < state->n_nonfull_stack; ++j) {
	  if(csstate[j].min_priority >= priority) {
	    break;
	  }
	}
	++demand[j];
      }
    }

#if 1
    if(lb >= ub) {
      break;
    }
#endif   
    
    for(i = state->n_nonfull_stack - 1; i > 0; --i) {
      cumulative_demand += demand[i];
      cumulative_supply += problem->s_height - csstate[i].n_tier;
      lb_add = max(lb_add, cumulative_demand - cumulative_supply);
    }

    lb += lb_add;

#if 1
    if(lb >= ub) {
      break;
    }
#endif   

    state->n_block -= n_stacked;
    csstate[0].n_stacked = 0;

    retrieve_all_blocks(problem, state, NULL);
  }

  state->n_block = n_block;
  state->n_nonfull_stack = n_nonfull;
  state->sstate = sstate;

  return(lb);
}
#endif /* LOWER_BOUND == 6 */
