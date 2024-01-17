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
 *  $Id: solve.c,v 1.49 2022/11/17 00:54:54 tanaka Exp tanaka $
 *  $Revision: 1.49 $
 *  $Date: 2022/11/17 00:54:54 $
 *  $Author: tanaka $
 *
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#ifdef _MSC_VER
#include <windows.h>
#define usleep(x) Sleep((x)/1000)
#else /* !_MSC_VER */
#include <unistd.h>
#endif /* !_MSC_VER */
#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */
#include "define.h"
#include "heuristics.h"
#include "lower_bound.h"
#include "node_pool.h"
#include "print.h"
#include "problem.h"
#include "solution.h"
#include "solve.h"
#include "timer.h"

#define BB_HEURISTICS

/* NUMBER OF INITIAL NODES */
#define INITIAL_NODES (10*n_thread)

/* NUMBER OF NODES BEFORE BACKTRACK */
#define BACKTRACK_NODES (1ll<<16)
#define BACKTRACK_COUNT1 (1<<10)
#define BACKTRACK_COUNT2 (1<<16)

typedef struct {
  int index;
  int tier;
  bay_state_t dstb;
  int n_block;
  int n_bblock;
  int n_nonfull_stack;
#if LOWER_BOUND != 1
  int lb;
#endif /* LOWER_BOUND != 1 */
} child_node_t;

typedef struct {
  ulint n_node;
  int count;
  ulint n_backtrack_node;
  int backtrack_count;

  /* keep track of the bay configuration */
  stack_state_t ***stack_state;

  /* cache lower bound computation results at parent node */
  int ***lb_cache;

  /* branching */
  child_node_t **child_node;

#ifdef BB_HEURISTICS
  /* heuristic */
  state_t *hstate;
#endif /* BB_HEURISTICS */

  stack_state_t *lb_stack_state;
  int *lb_work;
} env_t;

static env_t *env = NULL;
static solution_t *solution = NULL;
static int initial_lb = 0;
static backtrack_node_pool_t *backtrack_node_pool;
#ifdef _OPENMP
static omp_lock_t solution_lock;
static omp_lock_t backtrack_lock;
static int n_active_thread = 0;
#endif /* _OPENMP */

static env_t *create_env(problem_t *, int);
static void free_env(env_t *);
static int preprocess(problem_t *, state_t *, solution_t *);

#ifdef _OPENMP
static uchar fill_node_pool(problem_t *, env_t *, int *, node_pool_t *);
#endif /* _OPENMP */

static uchar bb(problem_t *, env_t *, state_t *, int *, solution_t *, int *,
		int);
static uchar bb_sub(problem_t *, env_t *, state_t *, int *, solution_t *,
		    int *, int, int *);

env_t *create_env(problem_t *problem, int max_depth)
{
  int i, j;
  env_t *env = (env_t *) malloc(sizeof(env_t));

  env->n_node = env->n_backtrack_node = 0ll;
  env->count = env->backtrack_count = 0;

  env->stack_state = (stack_state_t ***) malloc((size_t) max_depth
						*sizeof(stack_state_t **));

  env->stack_state[0]
    = (stack_state_t **) malloc((size_t) (max_depth*problem->n_stack)
				*sizeof(stack_state_t *));
  env->stack_state[0][0]
    = (stack_state_t *) malloc((size_t) (max_depth*problem->n_stack
					 *problem->n_stack)
			       *sizeof(stack_state_t));

  env->lb_cache = (int ***) malloc((size_t) max_depth*sizeof(int **));
  env->lb_cache[0]
    = (int **) malloc((size_t) (max_depth*problem->n_stack)*sizeof(int *));
  env->lb_cache[0][0]
    = (int *) malloc((size_t) (max_depth*problem->n_stack*problem->n_block)
		     *sizeof(int));

  for(i = 0; i < max_depth; ++i) {
    if(i > 0) {
      env->stack_state[i] = env->stack_state[i - 1] + problem->n_stack;
      env->stack_state[i][0]
	= env->stack_state[i - 1][0] + problem->n_stack*problem->n_stack;

      env->lb_cache[i] = env->lb_cache[i - 1] + problem->n_stack;
      env->lb_cache[i][0]
	= env->lb_cache[i - 1][0] + problem->n_stack*problem->n_block;
    }
    for(j = 1; j < problem->n_stack; ++j) {
      env->stack_state[i][j]
	= env->stack_state[i][j - 1] + problem->n_stack;
      env->lb_cache[i][j] = env->lb_cache[i][j - 1] + problem->n_block;
    }
  }

  env->child_node
    = (child_node_t **) malloc((size_t) max_depth*sizeof(child_node_t *));
  env->child_node[0]
    = (child_node_t *) malloc((size_t) max_depth*problem->n_stack
			      *sizeof(child_node_t));

  for(i = 1; i < max_depth; ++i) {
    env->child_node[i] = env->child_node[i - 1] + problem->n_stack;
  }

#ifdef BB_HEURISTICS
  env->hstate = create_state(problem);
#endif /* BB_HEURISTICS */

  env->lb_stack_state = NULL;
  env->lb_work = NULL;
#if LOWER_BOUND >= 3
  env->lb_stack_state = (stack_state_t *) malloc((size_t) problem->n_stack
						     *sizeof(stack_state_t));
#if LOWER_BOUND == 6
  env->lb_work = (int *) malloc((size_t) problem->n_stack*sizeof(int));
#elif LOWER_BOUND == 5
  env->lb_work = (int *) malloc((size_t) problem->s_height*sizeof(int));
#elif LOWER_BOUND == 4
  env->lb_work = (int *) malloc((size_t) problem->s_height*sizeof(int));
#endif /* LOWER_BOUND == 4 */
#endif /* LOWER_BOUND >= 3 */

  return(env);
}

void free_env(env_t *env)
{
  if(env != NULL) {
    free(env->child_node[0]);
    free(env->child_node);
    free(env->lb_cache[0][0]);
    free(env->lb_cache[0]);
    free(env->lb_cache);
    free(env->stack_state[0][0]);
    free(env->stack_state[0]);
    free(env->stack_state);
    free_state(env->hstate);
#if LOWER_BOUND >= 3
    free(env->lb_work);
    free(env->lb_stack_state);
#endif /* LOWER_BOUND >= 3 */

    free(env);
    env = NULL;
  }
}

int preprocess(problem_t *problem, state_t *state, solution_t *sol)
{
  int i, k;
  bay_state_t **bay = state->bay;
  stack_state_t *sstate = state->sstate;

  retrieve_all_blocks(problem, state, NULL);

  while(state->n_block > 0 && state->n_nonfull_stack <= 2) {
    int src_stack = sstate[0].no, dst_stack = sstate[1].no;

    for(i = 0; i < sstate[0].n_stacked; ++i) {
      block_t cblock = bay[src_stack][sstate[0].n_tier--].block;
      bay[dst_stack][++sstate[1].n_tier].block = cblock;

      if(sstate[1].n_tier > problem->s_height) {
	fprintf(stderr, "Infeasible\n");
	exit(1);
      }

      if(sstate[1].min_priority < cblock.priority) {
	++sstate[1].n_stacked;
      } else {	    
	--state->n_bblock;
	sstate[1].min_priority = cblock.priority;
	sstate[1].n_stacked = 0;
      }

      bay[dst_stack][sstate[1].n_tier].min_priority = sstate[1].min_priority;
      bay[dst_stack][sstate[1].n_tier].n_stacked = sstate[1].n_stacked;

      add_relocation(sol, src_stack, dst_stack, &cblock);
      state->last_relocation[cblock.no] = sol->n_relocation;
    }

    sstate[0].last_modified = sol->n_relocation;

    if(sstate[1].n_tier == problem->s_height) {
      stack_state_t current = sstate[1];

      for(k = 1; k < state->n_nonfull_stack - 1; ++k) {
	sstate[k] = sstate[k + 1];
      }
      for(; k < problem->n_stack - 1
	    && stack_state_comp((void *) &(sstate[k + 1]),
				(void *) &current) < 0; ++k) {
	sstate[k] = sstate[k + 1];
      }
      sstate[k] = current;
      --state->n_nonfull_stack;
    }

    sstate[0].n_stacked = 0;
    retrieve_all_blocks(problem, state, sol);
  }

  return(sol->n_relocation);
}

uchar solve(problem_t *problem, solution_t *sol)
{
  int i;
  int max_depth = 0;
  uchar ret = False;
  ulint n_node = 1;
  state_t *state = initialize_state(problem, NULL);
  int *lb_cache = NULL;
  solution_t *fixed_solution = create_solution();

  preprocess(problem, state, fixed_solution);

#if 0
  fprintf(stderr, "fixed_relocation=%d\n", fixed_solution->n_relocation);
#endif

  solution = sol;
  copy_solution(solution, fixed_solution);

  if(state->n_block > 0) {
    state_t *hstate = duplicate_state(problem, state);
    greedy_heuristic(problem, hstate, solution, MAX_N_RELOCATION);
    free_state(hstate);
    max_depth = solution->n_relocation;
    env = create_env(problem, solution->n_relocation + 1);
#if LOWER_BOUND == 1
    initial_lb = state->n_bblock;
#else /* LOWER_BOUND != 1 */
    lb_cache = (int *) calloc((size_t) problem->n_block, sizeof(int));
    initial_lb = lower_bound(problem, state, env->lb_stack_state,
			     lb_cache, env->lb_work,
			     1, problem->max_priority,
			     solution->n_relocation
			     - fixed_solution->n_relocation);
#endif /* LOWER_BOUND != 1 */
    initial_lb += fixed_solution->n_relocation;

    fprintf(stderr, "initial lb=%d ub=%d\n", initial_lb,
	    solution->n_relocation);
  } else {
    fprintf(stderr, "initial lb=%d ub=%d\n", initial_lb,
	    solution->n_relocation);
    fprintf(stderr, "Trivial optimal solution (0 relocation).\n");
  }

  if(lower_bound_only == True) {
    free_env(env);
    free_solution(fixed_solution);
    free(lb_cache);
    free_state(state);
    return(False);
  }

  if(initial_lb == solution->n_relocation) {
    fprintf(stderr, "Initial upper bound is optimal.\n");
  } else {
    int ub;
    backtrack_node_pool = create_backtrack_node_pool(max_depth);

#ifdef _OPENMP
    if(n_thread > 1) {
      int thread;
      node_pool_t node_pool;
      node_t **current_node
	= (node_t **) malloc((size_t) n_thread*sizeof(node_t *));
      uchar *active_thread = (uchar *) malloc(n_thread);
      omp_lock_t node_pool_lock;

      omp_init_lock(&solution_lock);
      omp_init_lock(&backtrack_lock);
      omp_init_lock(&node_pool_lock);

      node_pool.node = node_pool.unused = NULL;
      node_pool.n = 0;

      for(ub = initial_lb; ub < solution->n_relocation; ++ub) {
	fprintf(stderr, "cub=%d nodes=%llu ", ub, n_node);
	print_current_time(problem, stderr);
	append_node_by_element(problem, &(node_pool.node), &(node_pool.unused),
			       state, lb_cache, fixed_solution);
	++node_pool.n;
	backtrack_node_pool->min_depth = fixed_solution->n_relocation + 1;
	while(node_pool.n > 0 && node_pool.n < INITIAL_NODES) {
	  ++backtrack_node_pool->min_depth;
	  if(fill_node_pool(problem, env, &ub, &node_pool) == True) {
	    ret = True;
	    break;
	  }
	}

	n_node += env->n_node;
	env->n_node = 0;

	/* an optimal solution is found */
	if(ret != False) {
	  break;
	}

	/* ub is proved to be not optimal */
	if(node_pool.n == 0) {
	  continue;
	}

	n_active_thread = n_thread;
	for(i = 0; i < n_thread; active_thread[i++] = True);

	current_node[0] = node_pool.node;
	for(i = 1; i < n_thread; ++i) {
	  current_node[i] = current_node[i - 1]->next;
	}

	/* loop the linked list */
	node_pool.node->prev->next = node_pool.node;

#pragma omp parallel for schedule(static)
	for(thread = 0; thread < n_thread; ++thread) {
	  uchar local_ret;
	  node_t *node = current_node[thread];
	  env_t *tenv = create_env(problem, solution->n_relocation + 1);
	  ulint n_tnode = 0ll;

	  while(1) {
	    if(ret != False) {
	      break;
	    }

	    omp_set_lock(&node_pool_lock);
	    if(node_pool.n == 0) {
	      omp_unset_lock(&node_pool_lock);
	      break;
	    }

	    for(; node->active == False; node = node->next);
	    node->active = False;
	    --node_pool.n;
#if 0
	    fprintf(stderr, "thread %d: pool=%d\n", thread, node_pool.n);
#endif
	    omp_unset_lock(&node_pool_lock);

	    tenv->n_backtrack_node = BACKTRACK_NODES;
	    tenv->backtrack_count = BACKTRACK_COUNT1;
	    local_ret = bb(problem, tenv, node->state, node->lb_cache,
			   node->partial_solution, &ub, node->depth + 1);

	    if(local_ret == BackTrack) {
#if 0
	      fprintf(stderr, "thread %d: backtrack pool=%d\n", thread,
		      backtrack_node_pool->total_n);
#endif
	      omp_unset_lock(&backtrack_lock);
	    }

	    n_tnode += tenv->n_node;
	    tenv->n_node = 0ll;

#pragma omp atomic
	    ret |= local_ret & 3;
	  }

	  while(1) {
	    int depth;

	    if(ret != False) {
	      break;
	    }

	    omp_set_lock(&backtrack_lock);
	    if(backtrack_node_pool->total_n == 0) {
	      if(active_thread[thread] == True) {
		--n_active_thread;
	      }
	      omp_unset_lock(&backtrack_lock);
	      active_thread[thread] = False;
	      if(n_active_thread == 0) {
		break;
	      } else {
		usleep(500);
		continue;
	      }
	    }

	    for(depth = backtrack_node_pool->min_depth;
		depth < ub && backtrack_node_pool->n[depth] == 0; ++depth);
	    if(active_thread[thread] == False) {
	      ++n_active_thread;
	    }
	    active_thread[thread] = True;

	    pop_node(backtrack_node_pool->node[depth], node);
	    --backtrack_node_pool->total_n;
	    --backtrack_node_pool->n[depth];
	    backtrack_node_pool->min_depth = depth;
#if 0
	    fprintf(stderr, "thread %d: backtrack pool[%d]=%d/%d\n", thread,
		    depth, backtrack_node_pool->n[depth],
		    backtrack_node_pool->total_n);
#endif
	    omp_unset_lock(&backtrack_lock);

	    tenv->n_backtrack_node = BACKTRACK_NODES;
	    tenv->backtrack_count = BACKTRACK_COUNT2;
	    local_ret = bb(problem, tenv, node->state,
			   node->lb_cache, node->partial_solution, &ub,
			   node->depth + 1);

	    if(local_ret == BackTrack) {
#if 0
	      fprintf(stderr, "thread %d: backtrack pool=%d\n", thread,
		      backtrack_node_pool->total_n);
#endif
	    } else {
	      omp_set_lock(&backtrack_lock);
	    }

	    push_node(backtrack_node_pool->unused, node);
	    omp_unset_lock(&backtrack_lock);

	    n_tnode += tenv->n_node;
	    tenv->n_node = 0ll;

#pragma omp atomic
	    ret |= local_ret & 3;
	  }

#pragma omp atomic
	  n_node += n_tnode;
	  
	  free_env(tenv);
	}

	/* cut the loop */
	node_pool.node->prev->next = NULL;

	if(ret != False) {
	  break;
	}

	node_pool.n = 0;
	concatenate_list(node_pool.unused, node_pool.node);
	node_pool.node = NULL;

	for(i = 0; i < ub; ++i) {
	  backtrack_node_pool->n[i] = 0;
	  concatenate_list(backtrack_node_pool->unused,
			   backtrack_node_pool->node[i]);
	  backtrack_node_pool->node[i] = NULL;
	}
	backtrack_node_pool->total_n = 0;
	initial_lb = 0;
      }

      omp_destroy_lock(&node_pool_lock);
      omp_destroy_lock(&solution_lock);
      omp_destroy_lock(&backtrack_lock);

      free(active_thread);
      free(current_node);
      free_list(node_pool.unused);
      free_list(node_pool.node);
    } else {
#endif /* _OPENMP */
      int n_fixed = fixed_solution->n_relocation;

      for(ub = initial_lb; ub < solution->n_relocation; ++ub) {
	fprintf(stderr, "cub=%d nodes=%llu ", ub, n_node);
	print_current_time(problem, stderr);
	env->n_backtrack_node = BACKTRACK_NODES;
	env->backtrack_count = BACKTRACK_COUNT1;
	fixed_solution->n_relocation = n_fixed;
	ret = bb(problem, env, state, lb_cache, fixed_solution, &ub,
		 n_fixed + 1);
	n_node += env->n_node;
	env->n_node = 0ll;

	if((ret & 3) != False) {
	  ret &= 3;
	  break;
	}

#if 0
	if(ret == BackTrack) {
	  fprintf(stderr, "backtrack pool=%d\n", backtrack_node_pool->total_n);
	}
#endif

	backtrack_node_pool->min_depth = n_fixed + 1;
	while(backtrack_node_pool->total_n > 0) {
	  int j;
	  node_t *node;

	  for(j = backtrack_node_pool->min_depth;
	      j < ub && backtrack_node_pool->n[j] == 0; ++j);

	  pop_node(backtrack_node_pool->node[j], node);
	  --backtrack_node_pool->n[j];
	  --backtrack_node_pool->total_n;
	  backtrack_node_pool->min_depth = j;
#if 0
	  fprintf(stderr, "backtrack pool[%d]=%d/%d\n",
		  j, backtrack_node_pool->n[j], backtrack_node_pool->total_n);
#endif
	  env->n_backtrack_node = BACKTRACK_NODES;
	  env->backtrack_count = BACKTRACK_COUNT2;
	  ret = bb(problem, env, node->state, node->lb_cache,
		   node->partial_solution, &ub, node->depth + 1);
	  n_node += env->n_node;
	  env->n_node = 0ll;

	  append_node(backtrack_node_pool->unused, node);

	  if((ret & 3) != False) {
	    ret &= 3;
	    break;
	  }

#if 0
	  if(ret == BackTrack) {
	    fprintf(stderr, "backtrack pool=%d\n",
		    backtrack_node_pool->total_n);
	  }
#endif
	}

	if(ret != False) {
	  break;
	}

	for(i = 0; i < ub; ++i) {
	  backtrack_node_pool->n[i] = 0;
	  concatenate_list(backtrack_node_pool->unused,
			   backtrack_node_pool->node[i]);
	  backtrack_node_pool->node[i] = NULL;
	}
	backtrack_node_pool->total_n = 0;

	initial_lb = 0;
      }
#ifdef _OPENMP
    }
#endif /* _OPENMP */

    free_backtrack_node_pool(backtrack_node_pool);
  }

  fprintf(stderr, "nodes=%llu\n", n_node);

  free_solution(fixed_solution);
  free(lb_cache);
  free_state(state);
  free_env(env);

  return((ret == TimeLimit)?False:True);
}

#ifdef _OPENMP
uchar fill_node_pool(problem_t *problem, env_t *env, int *ub,
		     node_pool_t *node_pool)
{
  int j;
  int n_child;
  int src_stack, dst_stack;
  uchar ret = False;
  child_node_t *cn;
  bay_state_t bay_state_backup;
  stack_state_t *sstate;
  block_t rblock;
  node_pool_t node_pool_new;
  node_t *node;

  node_pool_new.n = 0;
  node_pool_new.node = NULL;
  node_pool_new.unused = node_pool->unused;

  while(node_pool->node != NULL) {
    node_t *node2 = node_pool->node->next;

    pop_node(node_pool->node, node);
    --node_pool->n;

    if((ret = bb_sub(problem, env, node->state, node->lb_cache,
		     node->partial_solution, ub, node->depth + 1, &n_child))
       != False) {
      push_node(node_pool_new.unused, node);
      break;
    }

    sstate = node->state->sstate;
    src_stack = sstate[0].no;
    rblock = node->state->bay[src_stack][sstate[0].n_tier].block;
    node->state->last_relocation[rblock.no] = node->depth + 1;
    cn = env->child_node[node->depth + 1];

    for(j = 0; j < n_child; ++j) {
      /* bounding */
      if(cn[j].lb + node->depth + 1 > *ub) {
	continue;
      }

      dst_stack = sstate[cn[j].index].no;
      node->state->n_bblock = cn[j].n_bblock;
      node->state->n_block = cn[j].n_block;
      node->state->n_nonfull_stack = cn[j].n_nonfull_stack;
      node->state->sstate = env->stack_state[node->depth + 1][cn[j].index];
      bay_state_backup = node->state->bay[dst_stack][cn[j].tier];
      node->state->bay[dst_stack][cn[j].tier] = cn[j].dstb;

      node->partial_solution->n_relocation = node->depth;
      add_relocation(node->partial_solution, src_stack, dst_stack, &rblock);

      append_node_by_element(problem, &(node_pool_new.node),
			     &(node_pool_new.unused),
			     node->state,
			     env->lb_cache[node->depth + 1][cn[j].index],
			     node->partial_solution);
      ++node_pool_new.n;

      node->state->bay[dst_stack][cn[j].tier] = bay_state_backup;
    }

    node->state->sstate = sstate;

    push_node(node_pool_new.unused, node);
    node_pool->node = node2;
  }

  concatenate_list(node_pool_new.unused, node_pool->node);

  node_pool->n = node_pool_new.n;
  node_pool->node = node_pool_new.node;
  node_pool->unused = node_pool_new.unused;

  return(ret);
}
#endif /* _OPENMP */

uchar bb(problem_t *problem, env_t *env, state_t *state, int *lb_cache,
	 solution_t *partial_solution, int *ub, int depth)
{
  int j;
  int n_child;
  int src_stack = state->sstate[0].no, dst_stack;
  int n_block_backup = state->n_block, n_bblock_backup = state->n_bblock;
  int n_nonfull_backup = state->n_nonfull_stack;
  int last_relocation;
  uchar ret;
  child_node_t *cn = env->child_node[depth];
  bay_state_t **bay = state->bay, bay_state_backup;
  stack_state_t *sstate = state->sstate;
  stack_state_t sstate_backup = sstate[0];
  block_t rblock = bay[src_stack][sstate[0].n_tier].block;

  if(depth > *ub) {
    return(False);
  }
  if(solution->n_relocation <= *ub) {
    return(True);
  }
  if(tlimit > 0 && ++env->count == 200000ll) {
    env->count = 0ll;
    if(get_time(problem) >= (double) tlimit) {
      return(TimeLimit);
    }
  }

#if 0
  printf("depth=%d\n", depth);
  print_state(problem, state, stdout);
  for(j = 0; j < problem->n_stack; ++j) {
    printf("[%d:%d:%d:%d]", state->sstate[j].no, state->sstate[j].n_tier,
	   state->sstate[j].min_priority,
	   state->sstate[j].n_stacked);
  }
  printf("\n");
  print_solution_relocation(problem, partial_solution, stdout);
  fflush(stdout);
#endif

  /* generate child nodes */
  if((ret = bb_sub(problem, env, state, lb_cache, partial_solution, ub, depth,
		   &n_child)) != False) {
    return(ret);
  }

#ifdef _OPENMP
  if(env->n_node >= env->n_backtrack_node) {
    if(n_thread > 1) {
      if(n_active_thread < n_thread
	 || (backtrack == True && --env->backtrack_count == 0)) {
	omp_set_lock(&backtrack_lock);
	ret = BackTrack;
      } else {
	env->n_backtrack_node += BACKTRACK_NODES;
      }
    } else if(backtrack == True && --env->backtrack_count == 0) {
      ret = BackTrack;
    } else {
      env->n_backtrack_node += BACKTRACK_NODES;
    }
  }
#else /* !_OPENMP */
  if(backtrack == True) {
    if(env->n_node >= env->n_backtrack_node) {
      if(--env->backtrack_count == 0) {
	ret = BackTrack;
      } else {
	env->n_backtrack_node += BACKTRACK_NODES;
      }
    }      
  }
#endif /* !_OPENMP */

  last_relocation = state->last_relocation[rblock.no];
  state->last_relocation[rblock.no] = depth;

  /* branching */
  for(j = 0; j < n_child; ++j) {
    if(solution->n_relocation <= *ub) {
      ret = True;
#ifdef _OPENMP
      if(n_thread > 1) {
	omp_unset_lock(&backtrack_lock);
      }
#endif /* _OPENMP */
      break;
    }

    /* bounding */
    if(cn[j].lb + depth > *ub) {
      continue;
    }

    dst_stack = sstate[cn[j].index].no;
    state->n_bblock = cn[j].n_bblock;
    state->n_block = cn[j].n_block;
    state->n_nonfull_stack = cn[j].n_nonfull_stack;
    state->sstate = env->stack_state[depth][cn[j].index];

    bay_state_backup = bay[dst_stack][cn[j].tier];
    bay[dst_stack][cn[j].tier] = cn[j].dstb;

    partial_solution->n_relocation = depth - 1;
    add_relocation(partial_solution, src_stack, dst_stack, &rblock);

    if(ret == BackTrack) {
      append_node_by_element(problem, &(backtrack_node_pool->node[depth]),
			     &(backtrack_node_pool->unused), state,
			     env->lb_cache[depth][cn[j].index],
			     partial_solution);
      ++backtrack_node_pool->n[depth];
      ++backtrack_node_pool->total_n;
      backtrack_node_pool->min_depth
	= min(backtrack_node_pool->min_depth, depth);
    } else {
      ret = bb(problem, env, state, env->lb_cache[depth][cn[j].index],
	       partial_solution, ub, depth + 1);
      if((ret & 3) != False) {
	/* an optimal solution is found, or the time limit is reached */
	break;
      }
    }

    bay[dst_stack][cn[j].tier] = bay_state_backup;
  }

  state->n_block = n_block_backup;
  state->n_nonfull_stack = n_nonfull_backup;
  state->n_bblock = n_bblock_backup;
  sstate[0] = sstate_backup;
  state->sstate = sstate;
  state->last_relocation[rblock.no] = last_relocation;

  return(ret);
}

uchar bb_sub(problem_t *problem, env_t *env, state_t *state, int *lb_cache,
	     solution_t *partial_solution, int *ub, int depth, int *n_child)
{
  int j, k;
  int max_index, last_relocation;
  int src_stack = state->sstate[0].no, dst_stack, dst_stack_tier;
  int n_block_backup = state->n_block, n_bblock_backup = state->n_bblock;
  int n_nonfull_backup = state->n_nonfull_stack;
#if LOWER_BOUND != 1
  int lb;
#endif /* LOWER_BOUND != 1 */
  block_t rblock;
  stack_state_t *sstate = state->sstate, *nsstate;
  stack_state_t sstate_backup = sstate[0], current;
  bay_state_t **bay = state->bay, bay_state_backup;
  child_node_t *cn = env->child_node[depth];
  
  *n_child = 0;
  --sstate[0].n_stacked;
  --sstate[0].n_tier;
  sstate[0].last_modified = depth;

  rblock = bay[src_stack][sstate[0].n_tier + 1].block;
  last_relocation = state->last_relocation[rblock.no];
  state->last_relocation[rblock.no] = depth;

  if(sstate[n_nonfull_backup - 1].n_tier == 0) {
    for(max_index = n_nonfull_backup - 1;
	sstate[max_index].n_tier == 0; --max_index);	
    max_index += 2;
  } else {
    max_index = n_nonfull_backup;
  }

  for(j = 1; j < max_index; ++j) {
    if(sstate[j].last_modified < last_relocation) {
      continue;
    }

    ++(env->n_node);
#if 0
    if(n_node > 1U<<30) {
      print_current_time(problem, stderr);
      exit(1);
    }
#endif

    if(sstate[j].min_priority < rblock.priority
       && depth + state->n_bblock > *ub) {
      continue;
    }

    dst_stack = sstate[j].no;
    state->sstate = nsstate = env->stack_state[depth][j];
    memcpy((void *) nsstate, (void *) sstate,
	   (size_t) problem->n_stack*sizeof(stack_state_t));
    state->n_nonfull_stack = n_nonfull_backup;
    ++nsstate[j].n_tier;

    if(rblock.priority > nsstate[j].min_priority) {
      state->n_bblock = n_bblock_backup;
      ++nsstate[j].n_stacked;

      current = nsstate[j];
      if(current.n_tier == problem->s_height) {
	for(k = j; k < n_nonfull_backup - 1; ++k) {
	  nsstate[k] = nsstate[k + 1];
	}
	for(; k < problem->n_stack - 1
	      && stack_state_comp((void *) &(nsstate[k + 1]),
				  (void *) &current) < 0; ++k) {
	  nsstate[k] = nsstate[k + 1];
	}
	--state->n_nonfull_stack;
      } else {
	for(k = j; k < n_nonfull_backup - 1
	      && stack_state_comp((void *) &(nsstate[k + 1]),
				  (void *) &current) < 0; ++k) {
	  nsstate[k] = nsstate[k + 1];
	}
      }
      nsstate[k] = current;
    } else {
      state->n_bblock = n_bblock_backup - 1;
      nsstate[j].min_priority = rblock.priority;
      nsstate[j].n_stacked = 0;

      current = nsstate[j];
      if(current.n_tier == problem->s_height) {
	for(k = j; k < n_nonfull_backup - 1; ++k) {
	  nsstate[k] = nsstate[k + 1];
	}
	for(; k < problem->n_stack - 1
	      && stack_state_comp((void *) &(nsstate[k + 1]),
				  (void *) &current) < 0; ++k) {
	  nsstate[k] = nsstate[k + 1];
	}
	--state->n_nonfull_stack;
      } else {
	for(k = j; k > 1 && stack_state_comp((void *) &(nsstate[k - 1]),
					     (void *) &current) > 0; --k) {
	  nsstate[k] = nsstate[k - 1];
	}
      }
      nsstate[k] = current;
    }

    state->n_block = n_block_backup;
    dst_stack_tier = nsstate[k].n_tier;
    nsstate[k].last_modified = depth;

    bay_state_backup = bay[dst_stack][dst_stack_tier];
    bay[dst_stack][dst_stack_tier].block = rblock;
    bay[dst_stack][dst_stack_tier].min_priority = nsstate[k].min_priority;
    bay[dst_stack][dst_stack_tier].n_stacked = nsstate[k].n_stacked;

#if 0
    {
      int l;
      printf("===\n");
      print_state(problem, state, stdout);
      for(l = 0; l < problem->n_stack; ++l) {
	printf("[%d:%d:%d:%d]", nsstate[l].no, nsstate[l].n_tier,
	       nsstate[l].n_stacked, nsstate[l].min_priority);
      }
      printf("\n");
      printf("===\n");
    }
#endif

    partial_solution->n_relocation = depth - 1;
    add_relocation(partial_solution, src_stack, dst_stack, &rblock);

    if(sstate[0].n_stacked == 0) {
      if(retrieve_all_blocks(problem, state, partial_solution) == True) {
	bay[dst_stack][dst_stack_tier] = bay_state_backup;
	continue;
      }
      if(state->n_block == 0) {
#ifdef _OPENMP
	if(n_thread > 1) {
	  omp_set_lock(&solution_lock);
	  copy_solution(solution, partial_solution);
	  fprintf(stderr, "ub=%d ", solution->n_relocation);
	  print_current_time(problem, stderr);
	  omp_unset_lock(&solution_lock);
	} else {
#endif /* _OPENMP */
	  copy_solution(solution, partial_solution);
	  fprintf(stderr, "ub=%d ", solution->n_relocation);
	  print_current_time(problem, stderr);
#ifdef _OPENMP
	}	    
#endif /* _OPENMP */
	sstate[0] = sstate_backup;
	state->sstate = sstate;
	state->n_block = n_block_backup;
	state->n_nonfull_stack = n_nonfull_backup;
	state->n_bblock = n_bblock_backup;
	state->last_relocation[rblock.no] = last_relocation;
	bay[dst_stack][dst_stack_tier] = bay_state_backup;
	return(True);
      }
    }

#if LOWER_BOUND != 1
#if LOWER_BOUND >= 4
    memcpy((void *) env->lb_cache[depth][j], (void *) lb_cache,
	   (size_t) problem->n_block*sizeof(int));
    if(rblock.priority < nsstate[0].min_priority) {
      lb = state->n_bblock;
      for(k = nsstate[0].min_priority; k < problem->max_priority; ++k) {
	lb += lb_cache[k];
      }
    } else {
      uchar check_type = 3;
      int check_priority = bay[dst_stack][dst_stack_tier].min_priority;

      if(dst_stack_tier == problem->s_height
	 || bay[dst_stack][dst_stack_tier].n_stacked == 0) {
	check_type = 1;
      } else if(sstate[0].n_stacked > 0) {
	check_type = 2;
      }
#endif /* LOWER_BOUND >= 4 */

      lb = lower_bound(problem, state, env->lb_stack_state,
		       env->lb_cache[depth][j], env->lb_work,
		       check_type, check_priority, *ub - depth + 1);

      if(lb + depth > *ub) {
	bay[dst_stack][dst_stack_tier] = bay_state_backup;
	continue;
      }
#if LOWER_BOUND >= 4
    }
#endif /* LOWER_BOUND >= 4 */
#endif /* LOWER_BOUND != 1 */

#ifdef BB_HEURISTICS
#if LOWER_BOUND == 1
    if(state->n_bblock + depth == *ub - 1) {
#else /* LOWER_BOUND != 1 */
    if(lb + depth == *ub - 1 || state->n_bblock + depth < initial_lb) {
#endif /* LOWER_BOUND != 1 */
      copy_state(problem, env->hstate, state);
      greedy_heuristic(problem, env->hstate, partial_solution,
		       solution->n_relocation);

#ifdef _OPENMP
      if(n_thread > 1) {
	omp_set_lock(&solution_lock);
	if(partial_solution->n_relocation < solution->n_relocation) {
	  copy_solution(solution, partial_solution);
	  fprintf(stderr, "ub=%d ", solution->n_relocation);
	  print_current_time(problem, stderr);
	}
	omp_unset_lock(&solution_lock);
      } else {
#endif /* _OPENMP */
	if(partial_solution->n_relocation < solution->n_relocation) {
	  copy_solution(solution, partial_solution);
	  fprintf(stderr, "ub=%d ", solution->n_relocation);
	  print_current_time(problem, stderr);
	}
#ifdef _OPENMP
      }	    
#endif /* _OPENMP */

      if(solution->n_relocation <= *ub) {
	sstate[0] = sstate_backup;
	state->sstate = sstate;
	state->n_block = n_block_backup;
	state->n_nonfull_stack = n_nonfull_backup;
	state->n_bblock = n_bblock_backup;
	state->last_relocation[rblock.no] = last_relocation;
	bay[dst_stack][dst_stack_tier] = bay_state_backup;
	return(True);
      }
#if LOWER_BOUND == 1
    }
#else /* LOWER_BOUND != 1 */
    }
#endif /* LOWER_BOUND != 1 */
#endif /* BB_HEURISTICS */

    cn[problem->n_stack - 1].index = j;
    cn[problem->n_stack - 1].tier = dst_stack_tier;
    cn[problem->n_stack - 1].dstb = bay[dst_stack][dst_stack_tier];
    cn[problem->n_stack - 1].n_block = state->n_block;
    cn[problem->n_stack - 1].n_bblock = state->n_bblock;
    cn[problem->n_stack - 1].n_nonfull_stack = state->n_nonfull_stack;
#if LOWER_BOUND != 1
    cn[problem->n_stack - 1].lb = lb;
#endif /* LOWER_BOUND != 1 */

#if LOWER_BOUND == 1
    for(k = *n_child - 1;
	k >= 0 && (cn[k].n_bblock > state->n_bblock
		   || (cn[k].n_bblock == state->n_bblock
		       && sstate[cn[k].index].min_priority
		       < sstate[j].min_priority));
	--k) {
      cn[k + 1] = cn[k];
    }
#else /* LOWER_BOUND != 1 */
    for(k = *n_child - 1;
	k >= 0 && (cn[k].lb > lb
		   || (cn[k].lb == lb && sstate[cn[k].index].min_priority
		       < sstate[j].min_priority));
	--k) {
      cn[k + 1] = cn[k];
    }
#endif /* LOWER_BOUND != 1 */
    cn[k + 1] = cn[problem->n_stack - 1];
    ++(*n_child);

    bay[dst_stack][dst_stack_tier] = bay_state_backup;
  }

#if 0
  for(k = 0; k < *n_child; ++k) {
    if(cn[k].lb + depth <= *ub) {
      printf("[%d] lb=%d, dst_stack=%d\n", depth, cn[k].lb,
	     sstate[cn[k].index].no);
    }
  }
#endif

  state->n_block = n_block_backup;
  state->n_bblock = n_bblock_backup;
  state->n_nonfull_stack = n_nonfull_backup;
  sstate[0] = sstate_backup;
  state->sstate = sstate;
  state->last_relocation[rblock.no] = last_relocation;

  return(False);
}
