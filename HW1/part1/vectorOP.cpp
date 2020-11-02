#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //

  __pp_vec_float x;
  __pp_vec_int y; 
  
  __pp_vec_float result;
  
  __pp_vec_int zero = _pp_vset_int(0);
  __pp_vec_int one = _pp_vset_int(1);
  __pp_vec_float maximum = _pp_vset_float(9.999999f);
  
  __pp_mask maskVecLen, maskAll, maskIsEqual, maskIsNotEqual, maskbgt;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    // Because N is not sure can divided by VECTOR_WIDTH, so we cannot set the value be a VECTOR_WIDTH
    // We need to check which value is minimum
    maskVecLen = _pp_init_ones();
    maskAll = _pp_init_ones(min(N - i, VECTOR_WIDTH));

    // Because unknown which one is equal to zero, so we set everyone to be false
    maskIsEqual = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll);      // x = values[i];

    _pp_vset_int(y, 0, maskVecLen);
    _pp_vload_int(y, exponents + i, maskAll);     // y = exponents[i];

    // First, set the result to 1, and we can change the value when y != 0
    _pp_vset_float(result, 1, maskVecLen);

    // Record exp which one is equal to zero => true -> equal, false -> not equal
    _pp_veq_int(maskIsEqual, y, zero, maskVecLen);

    // To know which one is not equal to zero
    maskIsNotEqual = _pp_mask_not(maskIsEqual);

    // load value
    __pp_vec_int cnt;
    _pp_vset_int(cnt, 0, maskVecLen);      // To sure everyone is zero
    _pp_vmove_int(cnt, y, maskIsNotEqual); // count = y;

    // While loop
    while(_pp_cntbits(maskIsNotEqual)) {
      // Calculate
      _pp_vmult_float(result, result, x, maskIsNotEqual); // result *= x
      _pp_vsub_int(cnt, cnt, one, maskIsNotEqual); // cnt -= 1
      _pp_vgt_int(maskIsNotEqual, cnt, zero, maskVecLen); // Check which one is not be zero
    }

    // Check the result value whether bigger than the maximum is or not
    _pp_vgt_float(maskbgt, result, maximum, maskAll); // if (result > 9.999999f) {

    // If the result is bigger than the maximum then we set the result to maximum
    _pp_vmove_float(result, maximum, maskbgt); // result = 9.999999f;}

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll); // output[i] = result;
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  // Check everyone
  __pp_mask maskAll = _pp_init_ones();

  // Record the sum
  __pp_vec_float ans;
  _pp_vset_float(ans, 0, maskAll);

  //O(N / VECTOR_WIDTH)
  for (int i = 0; i < N; i += VECTOR_WIDTH) {
    __pp_vec_float tmp, tmp2;
    _pp_vload_float(tmp, values + i, maskAll);
    _pp_hadd_float(tmp2, tmp);
    _pp_vadd_float(ans, ans, tmp2, maskAll);
  }

  //O(log2(vector_width))
  int len = VECTOR_WIDTH / 2;
  while(len > 1) {
    __pp_vec_float tmp;
    _pp_interleave_float(tmp, ans);
    _pp_hadd_float(ans, tmp);
    len /= 2;
  }

  return ans.value[0];
}