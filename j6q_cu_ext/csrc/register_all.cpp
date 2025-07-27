#include <torch/library.h>

// Sort of a hack to allow the ops to be registered through FRAGMENTS.
TORCH_LIBRARY(j6q_cu_ext, m){}
