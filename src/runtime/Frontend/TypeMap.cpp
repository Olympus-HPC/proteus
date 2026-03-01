#include "proteus/Frontend/TypeMap.h"

#include "proteus/Frontend/IRType.h"

#include <cstddef>
#include <optional>

namespace proteus {

// Definitions for specializations.

IRType TypeMap<void>::get(std::size_t) { return {IRTypeKind::Void}; }
std::optional<IRType> TypeMap<void>::getPointerElemType() {
  return std::nullopt;
}
bool TypeMap<void>::isSigned() { return false; }

IRType TypeMap<float>::get(std::size_t) { return {IRTypeKind::Float}; }
std::optional<IRType> TypeMap<float>::getPointerElemType() {
  return std::nullopt;
}
bool TypeMap<float>::isSigned() { return false; }

IRType TypeMap<float[]>::get(std::size_t NElem) {
  return {IRTypeKind::Array, false, NElem, IRTypeKind::Float};
}
std::optional<IRType> TypeMap<float[]>::getPointerElemType() {
  return std::nullopt;
}
bool TypeMap<float[]>::isSigned() { return false; }

IRType TypeMap<double>::get(std::size_t) { return {IRTypeKind::Double}; }
std::optional<IRType> TypeMap<double>::getPointerElemType() {
  return std::nullopt;
}
bool TypeMap<double>::isSigned() { return false; }

IRType TypeMap<double[]>::get(std::size_t NElem) {
  return {IRTypeKind::Array, false, NElem, IRTypeKind::Double};
}
std::optional<IRType> TypeMap<double[]>::getPointerElemType() {
  return std::nullopt;
}
bool TypeMap<double[]>::isSigned() { return false; }

IRType TypeMap<size_t>::get(std::size_t) { return {IRTypeKind::Int64}; }
std::optional<IRType> TypeMap<size_t>::getPointerElemType() {
  return std::nullopt;
}
bool TypeMap<size_t>::isSigned() { return false; }

IRType TypeMap<size_t[]>::get(std::size_t NElem) {
  return {IRTypeKind::Array, false, NElem, IRTypeKind::Int64};
}
std::optional<IRType> TypeMap<size_t[]>::getPointerElemType() {
  return std::nullopt;
}
bool TypeMap<size_t[]>::isSigned() { return false; }

IRType TypeMap<int>::get(std::size_t) {
  return {IRTypeKind::Int32, /*Signed=*/true};
}
std::optional<IRType> TypeMap<int>::getPointerElemType() {
  return std::nullopt;
}
bool TypeMap<int>::isSigned() { return true; }

IRType TypeMap<int[]>::get(std::size_t NElem) {
  return {IRTypeKind::Array, /*Signed=*/true, NElem, IRTypeKind::Int32};
}
std::optional<IRType> TypeMap<int[]>::getPointerElemType() {
  return std::nullopt;
}
bool TypeMap<int[]>::isSigned() { return true; }

IRType TypeMap<unsigned int>::get(std::size_t) { return {IRTypeKind::Int32}; }
std::optional<IRType> TypeMap<unsigned int>::getPointerElemType() {
  return std::nullopt;
}
bool TypeMap<unsigned int>::isSigned() { return false; }

IRType TypeMap<unsigned int[]>::get(std::size_t NElem) {
  return {IRTypeKind::Array, false, NElem, IRTypeKind::Int32};
}
std::optional<IRType> TypeMap<unsigned int[]>::getPointerElemType() {
  return std::nullopt;
}
bool TypeMap<unsigned int[]>::isSigned() { return false; }

IRType TypeMap<int *>::get(std::size_t) {
  return {IRTypeKind::Pointer, /*Signed=*/true, 0, IRTypeKind::Int32};
}
std::optional<IRType> TypeMap<int *>::getPointerElemType() {
  return IRType{IRTypeKind::Int32, /*Signed=*/true};
}
bool TypeMap<int *>::isSigned() { return true; }

IRType TypeMap<unsigned int *>::get(std::size_t) {
  return {IRTypeKind::Pointer, false, 0, IRTypeKind::Int32};
}
std::optional<IRType> TypeMap<unsigned int *>::getPointerElemType() {
  return IRType{IRTypeKind::Int32};
}
bool TypeMap<unsigned int *>::isSigned() { return false; }

IRType TypeMap<bool>::get(std::size_t) { return {IRTypeKind::Int1}; }
std::optional<IRType> TypeMap<bool>::getPointerElemType() {
  return std::nullopt;
}
bool TypeMap<bool>::isSigned() { return false; }

IRType TypeMap<bool[]>::get(std::size_t NElem) {
  return {IRTypeKind::Array, false, NElem, IRTypeKind::Int1};
}
std::optional<IRType> TypeMap<bool[]>::getPointerElemType() {
  return std::nullopt;
}
bool TypeMap<bool[]>::isSigned() { return false; }

IRType TypeMap<double *>::get(std::size_t) {
  return {IRTypeKind::Pointer, false, 0, IRTypeKind::Double};
}
std::optional<IRType> TypeMap<double *>::getPointerElemType() {
  return IRType{IRTypeKind::Double};
}
bool TypeMap<double *>::isSigned() { return false; }

IRType TypeMap<float *>::get(std::size_t) {
  return {IRTypeKind::Pointer, false, 0, IRTypeKind::Float};
}
std::optional<IRType> TypeMap<float *>::getPointerElemType() {
  return IRType{IRTypeKind::Float};
}
bool TypeMap<float *>::isSigned() { return false; }

} // namespace proteus
