#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

/* ---------- Types ---------- */

/* ---------- Bodies --------- */
int32_t _rite_body_0(size_t _0, const const uint8_t** _1) {
    int32_t _2;
    int32_t _3;

  basic_block0:
    _2 = (_rite_body_2)(2, 3); goto basic_block1;

  basic_block1:
    _3 = (_rite_body_1)(0, _2); goto basic_block2;

  basic_block2:
    return _3;
}

int32_t _rite_body_1(int32_t _0, int32_t _1) {

  basic_block0:
    return _0 + _1;
}

int32_t _rite_body_2(int32_t _0, int32_t _1) {

  basic_block0:
    return _0 * _1;
}


int main(int argc, char** argv) {
  return _rite_body_0((size_t) argc, (const uint8_t**) argv);
}
