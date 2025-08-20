#include "proteus/Frontend/LoopNest.hpp"

namespace proteus {

LoopBoundsDescription::LoopBoundsDescription(Var &IterVar, Var &Init,
                                             Var &UpperBound, Var &Inc)
    : IterVar(IterVar), Init(Init), UpperBound(UpperBound), Inc(Inc) {}

ForLoopBuilder::ForLoopBuilder(LoopBoundsDescription Bounds)
    : Bounds(std::move(Bounds)) {}

ForLoopBuilder::ForLoopBuilder(LoopBoundsDescription Bounds,
                               std::function<void()> Body)
    : Bounds(std::move(Bounds)), Body(std::move(Body)) {}

ForLoopBuilder &ForLoopBuilder::tile(int Tile) {
  TileSize = Tile;
  return *this;
}

LoopNestBuilder::LoopNestBuilder(FuncBase &Fn,
                                 std::vector<ForLoopBuilder> Loops)
    : Loops(std::move(Loops)), Fn(Fn) {}

LoopNestBuilder LoopNestBuilder::create(FuncBase &Fn,
                                        std::vector<ForLoopBuilder> Loops) {
  return LoopNestBuilder(Fn, std::move(Loops));
}

LoopNestBuilder
LoopNestBuilder::create(FuncBase &Fn,
                        std::initializer_list<ForLoopBuilder> Loops) {
  return create(Fn, std::vector<ForLoopBuilder>(Loops));
}

void LoopNestBuilder::emit() { emitLoopAtDimension(0); }

void LoopNestBuilder::emitLoopAtDimension(std::size_t Dim) {
  if (Dim >= Loops.size()) {
    return;
  }

  auto &CurLoop = Loops[Dim];

  const bool UseTiling = CurLoop.TileSize.has_value();
  if (UseTiling) {
    auto &TileIter = Fn.declVar<int>("tile_iter_" + std::to_string(Dim));
    auto &TileEnd = Fn.declVar<int>("tile_end_" + std::to_string(Dim));
    auto &TileStep = Fn.declVar<int>("tile_step_" + std::to_string(Dim));
    TileStep = CurLoop.TileSize.value();

    Fn.beginFor(TileIter, CurLoop.Bounds.Init, CurLoop.Bounds.UpperBound,
                TileStep);
    {
      // Inner loop over the tile range
      TileEnd = TileIter + TileStep;
      Fn.beginFor(CurLoop.Bounds.IterVar, TileIter, TileEnd,
                  CurLoop.Bounds.Inc);
      {
        if (CurLoop.Body.has_value()) {
          CurLoop.Body.value()();
        }
        emitLoopAtDimension(Dim + 1);
      }
      Fn.endFor();
    }
    Fn.endFor();
    return;
  }

  // Non-tiled simple loop
  Fn.beginFor(CurLoop.Bounds.IterVar, CurLoop.Bounds.Init,
              CurLoop.Bounds.UpperBound, CurLoop.Bounds.Inc);
  {
    if (CurLoop.Body.has_value()) {
      CurLoop.Body.value()();
    }
    emitLoopAtDimension(Dim + 1);
  }
  Fn.endFor();
}

} // namespace proteus