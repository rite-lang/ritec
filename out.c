#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

/* ---------- Types ---------- */
size_t _rite_body_0(size_t _0, const const uint8_t** _1);
typedef struct {
    int32_t* _0;
    size_t _1;
    size_t _2;
} _rite_struct_0;

typedef struct {
} _rite_struct_1;

_rite_struct_0 _rite_body_1();
_rite_struct_1 _rite_body_2(_rite_struct_0* _0);
uint8_t _rite_body_3(size_t _0, size_t _1);
size_t _rite_body_4(size_t _0, size_t _1);
int32_t* _rite_body_5(size_t _0);
_rite_struct_1 _rite_body_6(int32_t* _0);

/* ---------- Bodies --------- */
size_t _rite_body_0(size_t _0, const const uint8_t** _1) {
    _rite_struct_0 _2;
    _rite_struct_0 _3;
    _rite_struct_0* _4;
    _rite_struct_1 _5;
    _rite_struct_0* _6;
    _rite_struct_1 _7;

  basic_block0:
    _3 = (_rite_body_1)(); goto basic_block1;

  basic_block1:
    _2 = _3;
    _4 = &_2;
    _5 = (_rite_body_2)(_4); goto basic_block2;

  basic_block2:
    _6 = &_2;
    _7 = (_rite_body_2)(_6); goto basic_block3;

  basic_block3:
    return (_2)._2;
}

_rite_struct_0 _rite_body_1() {
    _rite_struct_0 _0;

  basic_block0:
    _0 = (_rite_struct_0) { (void*) 0, 0, 0 };
    return _0;
}

_rite_struct_1 _rite_body_2(_rite_struct_0* _0) {
    size_t _1;
    size_t _2;
    int32_t* _3;
    uint8_t _4;
    size_t _5;
    size_t _6;
    int32_t* _7;
    uint8_t _8;
    _rite_struct_1 _9;
    _rite_struct_1 _10;

  basic_block0:
    _1 = 0;
    _4 = (_rite_body_3)((*_0)._2, _1); goto basic_block1;

  basic_block1:
    switch (_4) {
      case 0: goto basic_block3;
      default: goto basic_block2;
    }

  basic_block2:
    _6 = 1;
    goto basic_block5;

  basic_block3:
    _5 = (_rite_body_4)((*_0)._2, (*_0)._2); goto basic_block4;

  basic_block4:
    _6 = _5;
    goto basic_block5;

  basic_block5:
    _2 = _6;
    _7 = (_rite_body_5)(_2); goto basic_block6;

  basic_block6:
    _3 = _7;
    _8 = (_rite_body_3)((*_0)._2, _1); goto basic_block7;

  basic_block7:
    switch (_8) {
      case 0: goto basic_block9;
      default: goto basic_block8;
    }

  basic_block8:
    (*_0)._0 = _3;
    (*_0)._2 = _2;
    _10 = (_rite_struct_1) {  };
    goto basic_block11;

  basic_block9:
    _9 = (_rite_body_6)((*_0)._0); goto basic_block10;

  basic_block10:
    (*_0)._0 = _3;
    (*_0)._2 = _2;
    _10 = (_rite_struct_1) {  };
    goto basic_block11;

  basic_block11:
    return _10;
}

uint8_t _rite_body_3(size_t _0, size_t _1) {

  basic_block0:
    return _0 == _1;
}

size_t _rite_body_4(size_t _0, size_t _1) {

  basic_block0:
    return _0 + _1;
}

int32_t* _rite_body_5(size_t _0) {

  basic_block0:
    return malloc(_0);
}

_rite_struct_1 _rite_body_6(int32_t* _0) {

  basic_block0:
    free(_0);
    return (_rite_struct_1) {  };
}


int main(int argc, char** argv) {
  return _rite_body_0((size_t) argc, (const uint8_t**) argv);
}
