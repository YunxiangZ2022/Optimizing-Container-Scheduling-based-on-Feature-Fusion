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
 *  $Id: main.c,v 1.30 2022/11/17 00:54:54 tanaka Exp tanaka $
 *  $Revision: 1.30 $
 *  $Date: 2022/11/17 00:54:54 $
 *  $Author: tanaka $
 *
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */
#include "define.h"
#include "description.h"
#include "print.h"
#include "problem.h"
#include "solution.h"
#include "solve.h"
#include "timer.h"

static problem_t *read_file(char *, int, int, int);
static void usage(char *);
static void remove_comments(char *);
static int blockdata_comp(const void *, const void *);

uchar verbose = False;
uchar lower_bound_only = False;
uchar backtrack = False;
int tlimit = -1;
int n_thread = 1;

int main(int argc, char **argv)
{
  char **agv;
  int n_stack = 0, s_height = 0, n_empty_tier = -1;
  uchar ret;
  problem_t *problem;
  solution_t *solution;

  for(agv = argv + 1, argc--; argc > 0 && agv[0][0] == '-'; --argc, ++agv) {
    switch(agv[0][1]) {
    default:
    case 'h':
      usage(argv[0]);
      return(1);
      break;
    case 'v':
      verbose = True;
      break;
    case 's':
      verbose = False;
      break;
    case 'l':
      lower_bound_only = True;
      break;
    case 'E':
      if(argc == 1) {
	usage(argv[0]);
	return(1);
      }
      n_empty_tier = (uint) atoi(agv[1]);
      ++agv;
      --argc;
      break;
    case 'S':
      if(argc == 1) {
	usage(argv[0]);
	return(1);
      }
      n_stack = (uint) atoi(agv[1]);
      ++agv;
      --argc;
      break;
    case 'T':
      if(argc == 1) {
	usage(argv[0]);
	return(1);
      }
      s_height = (uint) atoi(agv[1]);
      ++agv;
      --argc;
      break;
#ifdef _OPENMP
    case 'm':
      if(argc == 1) {
	usage(argv[0]);
	return(1);
      }
      n_thread = (int) atoi(agv[1]);
      ++agv;
      --argc;
      break;
#endif /* _OPENMP */
    case 'b':
      backtrack = True;
      break;
    case 't':
      if(argc == 1) {
	usage(argv[0]);
	return(1);
      }
      tlimit = (int) atoi(agv[1]);
      ++agv;
      --argc;
      break;
    }
  }

  problem = read_file((argc >= 1)?agv[0]:NULL, n_stack, s_height, n_empty_tier);

  if(problem == NULL) {
    return(0);
  }

#ifdef _OPENMP
  if(n_thread <= 0) {
    n_thread = omp_get_max_threads();
  } else {
    omp_set_num_threads(n_thread);
  }
  fprintf(stderr, "threads=%d\n", n_thread); 
#endif /* _OPENMP */

  if(verbose == True) {
    print_problem(problem, stderr);
  }

  solution = create_solution();

  timer_start(problem);

  ret = solve(problem, solution);

  timer_stop(problem);

#ifdef _OPENMP
  if(n_thread > 1) {
    print_total_time(problem, stderr);
  } else {
#endif /* _OPENMP */
    print_time(problem, stderr);
#ifdef _OPENMP
  }
#endif /* _OPENMP */

  if(ret == True) {
    fprintf(stderr, "opt=%d\n", solution->n_relocation);
  } else {
    fprintf(stderr, "best=%d\n", solution->n_relocation);
  }

  print_solution(problem, solution, stdout);

  free_solution(solution);

  free_problem(problem);

  return(0);
}

void usage(char *name)
{
#ifdef _OPENMP
  fprintf(stdout, "Usage: %s [-v|-s] [-E E] [-S S] [-T T] [-t L] [-m M] [-b] "
#else /* !_OPENMP */
  fprintf(stdout, "Usage: %s [-v|-s] [-E E] [-S S] [-T T] [-t L] [-b] "
#endif /* !_OPENMP */
	  "[input file]\n",
	  name);

  fprintf(stdout, DESCRIPTION_STRING "\n");
  fprintf(stdout, " -v|-s: verbose|silent\n");
  fprintf(stdout, " -E  E: additional empty tiers.\n");
  fprintf(stdout, " -S  S: number of stacks.\n");
  fprintf(stdout, " -T  T: stack height.\n");
  fprintf(stdout, " -t  L: time limit.\n");
#ifdef _OPENMP
  fprintf(stdout, " -m  M: number of threads. automatic when 0.\n");
#endif /* _OPENMP */
  fprintf(stdout, " -b   : backtrack and restart.\n");
  fprintf(stdout, "\n");
}

typedef struct {
  int s;
  int t;
  int priority;
} blockdata_t;

static struct {
  int type;
  char *key;
} key_list[] = {
  {0, "Tiers"},
  {0, "Height"},
  {1, "Width"},
  {1, "Stacks"},
  {2, "Containers"},
  {3, "Stack "},
  {-1, NULL}
};

problem_t *read_file(char *filename, int n_stack, int s_height,
		     int n_empty_tier)
{
  int i;
  int n_block;
  int current_block = 0, current_stack = 0, current_tier = 0;
  uchar file_type = 0;
  FILE *fp;
  problem_t *problem;
  char buf[MAXBUFLEN], *p;
  blockdata_t *blockdata;

  if(filename == NULL) {
    fp = stdin;
  } else {
#ifdef _MSC_VER
    if(fopen_s(&fp, filename, "r") != 0) {
#else /* !_MSC_VER */
    if((fp = fopen(filename, "r")) == NULL) {
#endif /* !_MSC_VER */
      fprintf(stderr, "Failed to open file: %s\n", filename);
      return(NULL);
    }
  }

  problem = NULL;
  while(fgets(buf, MAXBUFLEN, fp) != NULL) {
    int dn_stack;
    int ds_height;

    remove_comments(buf);
    for(p = buf; *p == ' ' || *p == '\t'; ++p);
    if(*p == '\0') {
      continue;
    }
    if(strchr(p, ':') != NULL) {
      int type = -1;
      int val;

      for(i = 0; key_list[i].type != -1; ++i) {
	if(strncmp(key_list[i].key, p, strlen(key_list[i].key)) == 0) {
	  type = key_list[i].type;
	  p += strlen(key_list[i].key);
	  break;
	}
      }

      if(type == -1) {
	continue;
      } else if(type == 3) {
	file_type = 1;
	break;
      }

      for(; *p != ':' && *p != '\0'; ++p);
      if(*p == ':') {
	++p;
      }
      val = (int) strtol(p, NULL, 10);

      switch(type) {
      default:
      case 0:
	if(n_empty_tier >= 0) {
	  s_height = val + n_empty_tier;
	} else {
	  s_height = val;
	}
	break;
	
      case 1:
	n_stack = max(n_stack, val);
	break;

      case 2:
	n_block = val;
	break;
      }
    } else {
      int c;

#ifdef _MSC_VER
      c = sscanf_s(buf, "%d %d %d", &dn_stack, &ds_height, &n_block);
#else /* !_MSC_VER */
      c = sscanf(buf, "%d %d %d", &dn_stack, &ds_height, &n_block);
#endif /* !_MSC_VER */
      if(c == 3) {
	n_stack = max(n_stack, dn_stack);
	if(s_height == 0) {
	  s_height = ds_height;
	}
      } else if(c == 2) {
	n_block = ds_height;
	n_stack = max(n_stack, dn_stack);
	if(s_height == 0 && n_empty_tier == -1) {
	  s_height = n_block;
	}
      }
      break;
    }
  }

  if(n_block == 0 || n_stack == 0) {
    problem = NULL;
    goto read_file_end;
  }

  problem = create_problem(n_stack, s_height, n_block);
  blockdata = calloc((size_t) n_block, sizeof(blockdata_t));

  if(file_type == 0) {
    int i;
    int n_tier = -1;

    while(current_stack < n_stack && current_block < n_block
	  && fgets(buf, MAXBUFLEN, fp) != NULL) {
      char *nptr, *endptr;

      remove_comments(buf);
      nptr = buf;

      while(1) {
	i = (int) strtol(nptr, &endptr, 10);
	if(nptr == endptr) {
	  break;
	} else {
	  nptr = endptr;
	}

	if(n_tier == -1) {
	  problem->n_tier[current_stack] = n_tier = i;
	  problem->s_height = max(problem->s_height, i);
	  if(n_tier == 0) {
	    if(++current_stack == n_stack) {
	      break;
	    }
	    n_tier = -1;
	  }
	  current_tier = 0;
	} else {
	  if(current_block == n_block) {
	    break;
	  }
	  blockdata[current_block].s = current_stack;
	  blockdata[current_block].t = current_tier;
	  blockdata[current_block++].priority = i;
	  if(++current_tier == n_tier) {
	    if(++current_stack == n_stack) {
	      break;
	    }
	    n_tier = -1;
	    current_tier = 0;
	  }
	}
      }
    }

    if(problem->s_height != n_block && n_empty_tier >= 0) {
      problem->s_height += n_empty_tier;
    }
  } else {
    do {
      char *endptr;

      for(p = buf; *p == ' ' || *p == '\t'; ++p);
      if(strncmp("Stack ", p, strlen("Stack ")) != 0) {
	continue;
      }

      for(p += strlen("Stack "); *p != ':' && *p != '\0'; ++p);
      if(*p == ':') {
	++p;
      }

      current_tier = 0;
      while(current_block < n_block) {
	int val = (int) strtol(p, &endptr, 10);

	if(p == endptr) {
	  break;
	} else {
	  p = endptr;
	}

	blockdata[current_block].s = current_stack;
	blockdata[current_block].t = current_tier++;
	blockdata[current_block++].priority = val;
      }

      problem->n_tier[current_stack++] = current_tier;
      s_height = max(s_height, current_tier);
    } while(fgets(buf, MAXBUFLEN, fp) != NULL);
  }

  if(n_block != current_block) {
    free_problem(problem);
    problem = NULL;
  } else {
    int min_priority, priority = -1, prev_priority = -1;

    qsort((void *) blockdata, n_block, sizeof(blockdata_t), blockdata_comp);

    problem->block[0]
      = (block_t *) calloc((size_t) problem->n_stack*problem->s_height,
			   sizeof(block_t));
    for(i = 1; i < problem->n_stack; ++i) {
      problem->block[i] = problem->block[i - 1] + problem->s_height;
    }

    min_priority = blockdata[0].priority;
    for(i = 0; i < n_block; ++i) {
      problem->priority[i] = blockdata[i].priority;
      blockdata[i].priority -= min_priority;
    }
    for(i = 0; i < n_block; ++i) {
      if(blockdata[i].priority > prev_priority) {
	++priority;
	prev_priority = blockdata[i].priority;
      }
      problem->position[i].s = blockdata[i].s;
      problem->position[i].t = blockdata[i].t;
      problem->block[blockdata[i].s][blockdata[i].t].no = i;
      problem->block[blockdata[i].s][blockdata[i].t].priority = priority;
    }

    problem->max_priority = priority;
  }

  free(blockdata);

  read_file_end:
  if(filename != NULL) {
    fclose(fp);
  }

  return(problem);
}

void remove_comments(char *c)
{
  char *a = c;

  for(; *c == ' ' || *c == '\t'; ++c);
  for(; *c != '\0' && *c != '#' && *c != '\n'; *a++ = *c, ++c);
  *a='\0';
}

int blockdata_comp(const void *a, const void *b)
{
  blockdata_t *x = (blockdata_t *) a;
  blockdata_t *y = (blockdata_t *) b;

  if(x->priority > y->priority) {
    return(1);
  } else if(x->priority < y->priority) {
    return(-1);
  } else if(x->t < y->t) {
    return(1);
  } else if(x->t > y->t) {
    return(-1);
  } else if(x->s > y->s) {
    return(1);
  } else if(x->s < y->s) {
    return(-1);
  }

  return(0);
}
