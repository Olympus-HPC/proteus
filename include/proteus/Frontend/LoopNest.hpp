#ifndef PROTEUS_FRONTEND_LOOP_NEST_HPP
#define PROTEUS_FRONTEND_LOOP_NEST_HPP

#include <functional>
#include <optional>
#include <vector>

#include "proteus/Frontend/Func.hpp"
#include "proteus/Frontend/Var.hpp"

namespace proteus {

class LoopBoundsDescription {
public:
  Var &IterVar;
  Var &Init;
  Var &UpperBound;
  Var &Inc;

  LoopBoundsDescription(Var &IterVar, Var &Init, Var &UpperBound, Var &Inc);
};

class ForLoopBuilder {
public:
  LoopBoundsDescription Bounds;
  std::optional<int> TileSize;
  std::optional<std::function<void()>> Body;

  explicit ForLoopBuilder(LoopBoundsDescription Bounds);

  ForLoopBuilder(LoopBoundsDescription Bounds, std::function<void()> Body);

  ForLoopBuilder &tile(int Tile);
};

class LoopNestBuilder {
private:
  std::vector<ForLoopBuilder> Loops;
  FuncBase &Fn;

public:
  LoopNestBuilder(FuncBase &Fn, std::vector<ForLoopBuilder> Loops);

  static LoopNestBuilder create(FuncBase &Fn,
                                std::vector<ForLoopBuilder> Loops);

  static LoopNestBuilder create(FuncBase &Fn,
                                std::initializer_list<ForLoopBuilder> Loops);

  void emit();
};
} // namespace proteus

#endif