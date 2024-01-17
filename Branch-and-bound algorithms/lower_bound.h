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
 *  $Id: lower_bound.h,v 1.6 2022/11/17 00:54:54 tanaka Exp tanaka $
 *  $Revision: 1.6 $
 *  $Date: 2022/11/17 00:54:54 $
 *  $Author: tanaka $
 *
 */
#pragma once
#include "define.h"
#include "problem.h"
#include "solution.h"

#ifndef LOWER_BOUND
/* #define LOWER_BOUND (1) */
/* #define LOWER_BOUND (2) */
/* #define LOWER_BOUND (3) */
/* #define LOWER_BOUND (4) */
/* #define LOWER_BOUND (5) */
/* #define LOWER_BOUND (6) */
#define LOWER_BOUND (5)
#endif /* !LOWER_BOUND */

#if LOWER_BOUND == 2
#define lower_bound(x, y, z, u, v, w, p, q) lower_bound2(x, y)
int lower_bound2(problem_t *, state_t *);
#elif LOWER_BOUND == 3
#define lower_bound(x, y, z, u, v, w, p, q) lower_bound3(x, y, z, u, w, p, q)
int lower_bound3(problem_t *, state_t *, stack_state_t *, int *, uchar, int,
		 int);
#elif LOWER_BOUND == 4
#define lower_bound(x, y, z, u, v, w, p, q) lower_bound4(x, y, z, u, v, w, p, q)
int lower_bound4(problem_t *, state_t *, stack_state_t *, int *, int *, uchar,
		 int, int);
#elif LOWER_BOUND == 5
#define lower_bound(x, y, z, u, v, w, p, q) lower_bound5(x, y, z, u, v, w, p, q)
int lower_bound5(problem_t *, state_t *, stack_state_t *, int *, int *, uchar,
		 int, int);
#elif LOWER_BOUND == 6
#define lower_bound(x, y, z, p, q) lower_bound6(x, y, z, u, v, w, p, q)
int lower_bound6(problem_t *, state_t *, stack_state_t *, int *, int *, uchar,
		 int, int);
#endif /* LOWER_BOUND == 6 */
