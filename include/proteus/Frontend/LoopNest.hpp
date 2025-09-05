#ifndef PROTEUS_FRONTEND_LOOP_NEST_HPP
#define PROTEUS_FRONTEND_LOOP_NEST_HPP

#include <memory>
#include <optional>

#include "proteus/Frontend/Func.hpp"
#include "proteus/Frontend/Var.hpp"

namespace proteus {

class LoopBoundInfo {
public:
  Var &IterVar;
  Var &Init;
  Var &UpperBound;
  Var &Inc;

  LoopBoundInfo(Var &IterVar, Var &Init, Var &UpperBound, Var &Inc);
};

template <typename BodyLambda> class ForLoopBuilder {
public:
  LoopBoundInfo Bounds;
  std::optional<int> TileSize;
  BodyLambda Body;
  FuncBase &Fn;

  ForLoopBuilder(const LoopBoundInfo &Bounds, FuncBase &Fn, BodyLambda &&Body)
      : Bounds(Bounds), Body(std::move(Body)), Fn(Fn) {}
  ForLoopBuilder &tile(int Tile) {
    TileSize = Tile;
    return *this;
  }

  void emit() {
    Fn.beginFor(Bounds.IterVar, Bounds.Init, Bounds.UpperBound, Bounds.Inc);
    Body();
    Fn.endFor();
  }
};

template <typename... LoopBuilders> class LoopNestBuilder {
private:
  std::tuple<LoopBuilders...> Loops;
  std::array<std::unique_ptr<LoopBoundInfo>, std::tuple_size_v<decltype(Loops)>>
      TiledLoopBounds;
  FuncBase &Fn;

  template <std::size_t... Is>
  void setupTiledLoops(std::index_sequence<Is...>) {
    (
        // Declare the tile iter and step variables for each tiled loop,
        // storing them in the TiledLoopBounds array.
        [&]() {
          auto &Loop = std::get<Is>(Loops);
          if (Loop.TileSize.has_value()) {
            auto &Bounds = std::get<Is>(Loops).Bounds;

            auto &TileIter =
                Fn.declVarInternal("tile_iter_" + std::to_string(Is),
                                   Bounds.IterVar.getValueType());
            auto &TileStep =
                Fn.declVarInternal("tile_step_" + std::to_string(Is),
                                   Bounds.IterVar.getValueType());

            TileStep = Loop.TileSize.value();
            TiledLoopBounds[Is] = std::make_unique<LoopBoundInfo>(
                TileIter, Bounds.Init, Bounds.UpperBound, TileStep);
          }
        }(),
        ...);
  }

  template <std::size_t... Is>
  void beginTiledLoops(std::index_sequence<Is...>) {
    (
        // Begin the tiled loops, using the computed tile bounds.
        [&]() {
          auto &Loop = std::get<Is>(Loops);
          if (Loop.TileSize.has_value()) {
            auto &Bounds = *TiledLoopBounds[Is];
            Fn.beginFor(Bounds.IterVar, Bounds.Init, Bounds.UpperBound,
                        Bounds.Inc);
          }
        }(),
        ...);
  }

  template <std::size_t... Is> void emitInnerLoops(std::index_sequence<Is...>) {
    (
        // Emit the inner loops, using this tile's iter var as init.
        [&]() {
          auto &Loop = std::get<Is>(Loops);
          if (Loop.TileSize.has_value()) {
            auto &TiledBounds = *TiledLoopBounds[Is];
            auto &EndCandidate = TiledBounds.IterVar + TiledBounds.Inc;
            // Clamp to handle partial tiles.
            EndCandidate = min(EndCandidate, TiledBounds.UpperBound);
            Fn.beginFor(Loop.Bounds.IterVar, TiledBounds.IterVar, EndCandidate,
                        Loop.Bounds.Inc);
          } else {
            Fn.beginFor(Loop.Bounds.IterVar, Loop.Bounds.Init,
                        Loop.Bounds.UpperBound, Loop.Bounds.Inc);
          }
          Loop.Body();
        }(),
        ...);
    (
        [&]() {
          // Force unpacking so we emit enough endFors.
          (void)std::get<Is>(Loops);
          Fn.endFor();
        }(),
        ...);
  }

  template <std::size_t... Is> void endTiledLoops(std::index_sequence<Is...>) {
    (
        [&]() {
          // Close tiled loops in reverse order to properly handle nesting.
          auto &Loop = std::get<sizeof...(Is) - 1U - Is>(Loops);
          if (Loop.TileSize.has_value()) {
            Fn.endFor();
          }
        }(),
        ...);
  }

  template <std::size_t... Is>
  void tileImpl(int Tile, std::index_sequence<Is...>) {
    (std::get<Is>(Loops).tile(Tile), ...);
  }

public:
  LoopNestBuilder(FuncBase &Fn, LoopBuilders... Loops)
      : Loops(std::move(Loops)...), Fn(Fn) {}

  LoopNestBuilder &tile(int Tile) {
    auto IdxSeq = std::index_sequence_for<LoopBuilders...>{};
    tileImpl(Tile, IdxSeq);
    return *this;
  }
  void emit() {
    auto IdxSeq = std::index_sequence_for<LoopBuilders...>{};
    setupTiledLoops(IdxSeq);
    beginTiledLoops(IdxSeq);
    emitInnerLoops(IdxSeq);
    endTiledLoops(IdxSeq);
  }
};
} // namespace proteus

#endif
