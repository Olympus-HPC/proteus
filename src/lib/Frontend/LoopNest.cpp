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

  emitDimension(0, TileIter, TileEnd, TileStep);
}

void LoopNestBuilder::emitDimension(std::size_t Dim,
                                    std::vector<Var *> &TileIter,
                                    std::vector<Var *> &TileEnd,
                                    std::vector<Var *> &TileStep) {
  const std::size_t NumDims = Loops.size();

  if (Dim >= NumDims) {
    return;
  }

  auto &CurLoop = Loops[Dim];
  const bool IsTiled = CurLoop.TileSize.has_value();

  if (IsTiled) {
    // Determine contiguous group of tiled dimensions starting at 'dim'
    size_t GroupEnd = Dim;
    while (GroupEnd < NumDims && Loops[GroupEnd].TileSize.has_value()) {
      ++GroupEnd;
    }

    // Declare tile vars for the group
    for (std::size_t GroupIdx = Dim; GroupIdx < GroupEnd; ++GroupIdx) {
      auto &LB = Loops[GroupIdx].Bounds;

      auto *IndTy = LB.IterVar.getPointerElemType();
      if (!IndTy || !IndTy->isIntegerTy()) {
        PROTEUS_FATAL_ERROR("ForLoop tiling requires integral induction type");
      }

      TileIter[GroupIdx] = &Fn.declVarInternal(
          "tile_iter_" + std::to_string(GroupIdx), IndTy, IndTy);
      TileEnd[GroupIdx] = &Fn.declVarInternal(
          "tile_end_" + std::to_string(GroupIdx), IndTy, IndTy);
      TileStep[GroupIdx] = &Fn.declVarInternal(
          "tile_step_" + std::to_string(GroupIdx), IndTy, IndTy);
      (*TileStep[GroupIdx]) = Loops[GroupIdx].TileSize.value();
    }

    emitTileLoops(Dim, GroupEnd, Dim, TileIter, TileEnd, TileStep);
    return;
  }

  // Non-tiled: emit this dimension's element loop immediately
  Fn.beginFor(CurLoop.Bounds.IterVar, CurLoop.Bounds.Init,
              CurLoop.Bounds.UpperBound, CurLoop.Bounds.Inc);
  {
    if (CurLoop.Body.has_value()) {
      CurLoop.Body.value()();
    }
    emitDimension(Dim + 1, TileIter, TileEnd, TileStep);
  }
  Fn.endFor();
}

void LoopNestBuilder::emitTileLoops(std::size_t GroupIdx, std::size_t GroupEnd,
                                    std::size_t Dim,
                                    std::vector<Var *> &TileIter,
                                    std::vector<Var *> &TileEnd,
                                    std::vector<Var *> &TileStep) {
  if (GroupIdx >= GroupEnd) {
    // Inside all tile loops: emit element loops for the group
    emitInnerLoops(Dim, GroupEnd, Dim, TileIter, TileEnd, TileStep);
    return;
  }

  auto &LoopG = Loops[GroupIdx];
  Fn.beginFor(*TileIter[GroupIdx], LoopG.Bounds.Init, LoopG.Bounds.UpperBound,
              *TileStep[GroupIdx]);
  {
    emitTileLoops(GroupIdx + 1, GroupEnd, Dim, TileIter, TileEnd, TileStep);
  }
  Fn.endFor();
}

void LoopNestBuilder::emitInnerLoops(std::size_t ElemIdx, std::size_t GroupEnd,
                                     std::size_t Dim,
                                     std::vector<Var *> &TileIter,
                                     std::vector<Var *> &TileEnd,
                                     std::vector<Var *> &TileStep) {
  if (ElemIdx >= GroupEnd) {
    // Continue with remaining dimensions after the group
    emitDimension(GroupEnd, TileIter, TileEnd, TileStep);
    return;
  }

  auto &LoopE = Loops[ElemIdx];
  // Clamp the tile end to the loop upper bound to handle partial tiles
  auto &EndCandidate = (*TileIter[ElemIdx]) + (*TileStep[ElemIdx]);
  auto &ClampedEnd = min(EndCandidate, LoopE.Bounds.UpperBound);
  (*TileEnd[ElemIdx]) = ClampedEnd;
  Fn.beginFor(LoopE.Bounds.IterVar, *TileIter[ElemIdx], *TileEnd[ElemIdx],
              LoopE.Bounds.Inc);
  {
    if (LoopE.Body.has_value()) {
      LoopE.Body.value()();
    }
    emitInnerLoops(ElemIdx + 1, GroupEnd, Dim, TileIter, TileEnd, TileStep);
  }
  Fn.endFor();
}

} // namespace proteus