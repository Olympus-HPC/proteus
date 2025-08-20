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

void LoopNestBuilder::emit() {
  const std::size_t NumDims = Loops.size();
  std::vector<Var *> TileIter(NumDims, nullptr);
  std::vector<Var *> TileEnd(NumDims, nullptr);
  std::vector<Var *> TileStep(NumDims, nullptr);

  std::function<void(std::size_t)> EmitFrom;
  EmitFrom = [&](std::size_t Dim) {
    if (Dim >= NumDims) {
      return;
    }

    auto &CurLoop = Loops[Dim];
    const bool IsTiled = CurLoop.TileSize.has_value();

    if (IsTiled) {
      // Determine contiguous group of tiled dimensions starting at 'dim'
      std::size_t GroupEnd = Dim;
      while (GroupEnd < NumDims && Loops[GroupEnd].TileSize.has_value()) {
        ++GroupEnd;
      }

      // Declare tile vars for the group
      for (std::size_t GroupIdx = Dim; GroupIdx < GroupEnd; ++GroupIdx) {
        TileIter[GroupIdx] = &Fn.declVar<int>("tile_iter_" + std::to_string(GroupIdx));
        TileEnd[GroupIdx] = &Fn.declVar<int>("tile_end_" + std::to_string(GroupIdx));
        TileStep[GroupIdx] = &Fn.declVar<int>("tile_step_" + std::to_string(GroupIdx));
        (*TileStep[GroupIdx]) = Loops[GroupIdx].TileSize.value();
      }

      // Open nested tile loops for the group in order
      std::function<void(std::size_t)> OpenTiles;
      OpenTiles = [&](std::size_t GroupIdx) {
        if (GroupIdx >= GroupEnd) {
          // Inside all tile loops: emit element loops for the group
          std::function<void(std::size_t)> EmitGroupElems;
          EmitGroupElems = [&](std::size_t ElemIdx) {
            if (ElemIdx >= GroupEnd) {
              // Continue with remaining dimensions after the group
              EmitFrom(GroupEnd);
              return;
            }
            auto &LoopE = Loops[ElemIdx];
            (*TileEnd[ElemIdx]) = (*TileIter[ElemIdx]) + (*TileStep[ElemIdx]);
            Fn.beginFor(LoopE.Bounds.IterVar, *TileIter[ElemIdx], *TileEnd[ElemIdx],
                        LoopE.Bounds.Inc);
            {
              if (LoopE.Body.has_value()) {
                LoopE.Body.value()();
              }
              EmitGroupElems(ElemIdx + 1);
            }
            Fn.endFor();
          };

          EmitGroupElems(Dim);
          return;
        }

        auto &LoopG = Loops[GroupIdx];
        Fn.beginFor(*TileIter[GroupIdx], LoopG.Bounds.Init, LoopG.Bounds.UpperBound,
                    *TileStep[GroupIdx]);
        {
          OpenTiles(GroupIdx + 1);
        }
        Fn.endFor();
      };

      OpenTiles(Dim);
      return;
    }

    // Non-tiled: emit this dimension's element loop immediately
    Fn.beginFor(CurLoop.Bounds.IterVar, CurLoop.Bounds.Init, CurLoop.Bounds.UpperBound, CurLoop.Bounds.Inc);
    {
      if (CurLoop.Body.has_value()) {
        CurLoop.Body.value()();
      }
      EmitFrom(Dim + 1);
    }
    Fn.endFor();
  };

  EmitFrom(0);
}

} // namespace proteus