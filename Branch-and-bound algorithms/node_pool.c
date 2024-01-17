/*
 * Copyright 2018-2022 Shunji Tanaka.  All rights reserved.
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
 *  $Id: node_pool.c,v 1.4 2022/11/17 00:54:54 tanaka Exp tanaka $
 *  $Revision: 1.4 $
 *  $Date: 2022/11/17 00:54:54 $
 *  $Author: tanaka $
 *
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "define.h"
#include "problem.h"
#include "solution.h"
#include "node_pool.h"

node_t *create_node(problem_t *problem)
{
  node_t *node = malloc(sizeof(node_t));

  node->depth = 0;
  node->active = True;
  node->next = NULL;
  node->prev = node;
  node->state = create_state(problem);
  node->lb_cache = (int *) malloc((size_t) problem->n_block*sizeof(int));
  node->partial_solution = create_solution();

  return(node);
}

void free_node(node_t *node)
{
  if(node != NULL) {
    free_solution(node->partial_solution);
    free(node->lb_cache);
    free_state(node->state);
    free(node);
  }
}

backtrack_node_pool_t *create_backtrack_node_pool(int max_depth)
{
  backtrack_node_pool_t *backtrack_node_pool
    = (backtrack_node_pool_t *) malloc(sizeof(backtrack_node_pool_t));

  backtrack_node_pool->total_n = 0;
  backtrack_node_pool->min_depth = 0;
  backtrack_node_pool->max_depth = max_depth;
  backtrack_node_pool->unused = NULL;
  backtrack_node_pool->n = (int *) calloc((size_t) max_depth, sizeof(int));
  backtrack_node_pool->node
    = (node_t **) calloc((size_t) max_depth, sizeof(node_t *));

  return(backtrack_node_pool);
}

void free_backtrack_node_pool(backtrack_node_pool_t *backtrack_node_pool)
{
  if(backtrack_node_pool != NULL) {
    int i;

    free(backtrack_node_pool->n);

    for(i = 0; i < backtrack_node_pool->max_depth; ++i) {
      free_list(backtrack_node_pool->node[i]);
    }

    free_list(backtrack_node_pool->unused);

    free(backtrack_node_pool->node);
    free(backtrack_node_pool);
  }
}

node_pool_t *create_node_pool(void)
{
  node_pool_t *node_pool = (node_pool_t *) malloc(sizeof(node_pool_t));

  node_pool->n = 0;
  node_pool->node = node_pool->unused = NULL;

  return(node_pool);
}

void free_node_pool(node_pool_t *node_pool)
{
  if(node_pool != NULL) {
    free_list(node_pool->node);
    free_list(node_pool->unused);
    free(node_pool);
  }
}

void append_node_by_element(problem_t *problem, node_t **list, node_t **unused,
			    state_t *state, int *lb_cache,
			    solution_t *partial_solution)
{
  node_t *node;

  pop_node(*unused, node);
  append_node(*list, node);

  node->active = True;
  node->depth = partial_solution->n_relocation;
  copy_state(problem, node->state, state);
  memcpy((void *) node->lb_cache, (void *) lb_cache,
	 (size_t) problem->n_block*sizeof(int));
  copy_solution(node->partial_solution, partial_solution);
}
