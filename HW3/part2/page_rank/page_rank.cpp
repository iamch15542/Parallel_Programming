#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;

  #pragma omp parallel for
  for (int i = 0; i < numNodes; ++i)
  {
    solution[i] = equal_prob;
  }

  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
  bool converged = false;
  double* score_new = (double*)malloc(sizeof(double) * numNodes);
  double* tmp = (double*)malloc(sizeof(double) * numNodes);
  while(!converged) {

    // Variable
    double sum = 0.0, no_out = 0.0;

    // no outgoing edges
    #pragma omp parallel for reduction(+:no_out)
    for(int j = 0; j < numNodes; ++j) {
      int leave_size = outgoing_size(g, j);
      if(leave_size == 0) {
        no_out += (damping * solution[j] / numNodes);
      }
    }
    
    // Every node
    #pragma omp parallel for firstprivate(sum)
    for(int i = 0; i < numNodes; ++i) {

      // tmp variable
      sum = 0.0;
      
      // Get incoming edges
      const Vertex* start = incoming_begin(g, i);
      const Vertex* end = incoming_end(g, i);
      for(const Vertex* v = start; v != end; v++) {
        int leave_size = outgoing_size(g, *v);
        sum += (solution[*v] / leave_size);
      }
      
      score_new[i] = (damping * sum) + (1.0 - damping) / numNodes;

      // no out
      score_new[i] += no_out;
    }
    
    double global_diff = 0.0;
    #pragma omp parallel for
    for(int i = 0; i < numNodes; ++i) {
      tmp[i] = abs(score_new[i] - solution[i]);
    }

    #pragma omp parallel for reduction(+:global_diff)
    for(int i = 0; i < numNodes; ++i) {
      global_diff += tmp[i];
    }
    
    #pragma omp parallel for
    for(int i = 0; i < numNodes; ++i) {
      solution[i] = score_new[i];
    }

    // printf("%.10lf %.10lf\n", global_diff, convergence);
    converged = (global_diff < convergence);
  }
  free(score_new);
  free(tmp);
}
