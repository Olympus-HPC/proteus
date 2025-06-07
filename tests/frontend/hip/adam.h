typedef enum {
  ADAM_MODE_0 = 0, // eps under square root
  ADAM_MODE_1 = 1  // eps outside square root
} adamMode_t;

template <typename T, typename G>
void reference(int repeat, T *p, T *m, T *v, const G *g, const float b1,
               const float b2, const float eps, const float grad_scale,
               const float step_size, const int time_step,
               const size_t vector_size, adamMode_t mode, const float decay) {
  for (int i = 0; i < repeat; i++) {
    for (size_t j = 0; j < vector_size; j++) {
      for (int t = 0; t < time_step; t++) {
        T scaled_grad = g[j] / grad_scale;
        m[j] = b1 * m[j] + (1.f - b1) * scaled_grad;
        v[j] = b2 * v[j] + (1.f - b2) * scaled_grad * scaled_grad;
        float m_corrected = m[j] / (1.f - powf(b1, t));
        float v_corrected = v[j] / (1.f - powf(b2, t));
        float denom;
        if (mode == ADAM_MODE_0)
          denom = sqrtf(v_corrected + eps);
        else // Mode 1
          denom = sqrtf(v_corrected) + eps;
        float update = (m_corrected / denom) + (decay * p[j]);
        p[j] -= (step_size * update);
      }
    }
  }
}
