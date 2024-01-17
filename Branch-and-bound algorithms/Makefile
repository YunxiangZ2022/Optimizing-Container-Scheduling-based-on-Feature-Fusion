#
#  $Id: Makefile,v 1.12 2021/05/17 12:54:14 tanaka Exp tanaka $
#  $Revision: 1.12 $
#  $Date: 2021/05/17 12:54:14 $
#  $Author: tanaka $
#
.PHONY: all strip clean depend

ARCH      := $(shell uname -m)
OBJS       = heuristics.o lower_bound.o main.o node_pool.o print.o problem.o \
	solution.o solve.o timer.o
SRCS      := $(OBJS:.o=.c)

TARGET     = rbrp_bb

CC         = gcc
MAKEDEP    = gcc -MM

CFLAGS     = -Wall -Wextra -pedantic -Ofast -fopenmp
#CFLAGS    += -march=corei7
#CFLAGS    += -march=znver2

LIBS       = 

override DEFS += -I. # -DDEBUG -DUSE_CLOCK

all:: $(TARGET)

.c.o:
	$(CC) $(CFLAGS) $(DEFS) -c $<

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(DEFS) -o $@ $(OBJS) $(LIBS)

strip:: $(TARGET)
	@strip $(TARGET)

clean:
	rm -f $(TARGET) $(OBJS) *~ *.bak #*

depend:
	@sed -i -e "/^# START/,/# END/d" Makefile
	@echo "# START" >> Makefile
	@$(MAKEDEP) $(DEFS) $(SRCS) >> Makefile
	@echo "# END" >> Makefile

# START
heuristics.o: heuristics.c define.h heuristics.h problem.h solution.h \
 print.h
lower_bound.o: lower_bound.c define.h lower_bound.h problem.h solution.h \
 print.h
main.o: main.c define.h description.h print.h problem.h solution.h \
 solve.h timer.h
node_pool.o: node_pool.c define.h problem.h solution.h node_pool.h
print.o: print.c define.h print.h problem.h solution.h timer.h
problem.o: problem.c define.h problem.h
solution.o: solution.c define.h solution.h problem.h
solve.o: solve.c define.h heuristics.h problem.h solution.h lower_bound.h \
 node_pool.h print.h solve.h timer.h
timer.o: timer.c define.h timer.h problem.h
# END
