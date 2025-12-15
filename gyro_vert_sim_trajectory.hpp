#include <cmath>
inline double get_V(const double t) noexcept {
  if (t >= 0.0 && t < 27.75)
    return 0.8 * t;
  else
    return 22.2;
}
inline double get_dtheta_p(const double t) noexcept {
  if (t >= 0.0 && t < 15.0)
    return 0.0;
  else if (t >= 15.0 && t < 15.0 + 3.4)
    return 7.4e-3;
  else
    return 0.0;
}
inline double get_dpsi_p(const double t) noexcept {
  if (t >= 0.0 && t < 25.0)
    return 0.0;
  else if (t >= 25.0 && t < 25.0 + 1.8)
    return 1.8e-2 * (t - 25.0);
  else if (t >= 25.0 + 1.8 && t < 25.0 + 1.8 + 49.5)
    return 3.2e-2;
  else if (t >= 25.0 + 1.8 + 49.5 && t < 25.0 + 49.5 + 1.8 * 2)
    return 3.2e-2 - 1.8e-2 * (t - (25.0 + 1.8 + 49.5));
  else
    return 0.0;
}
inline double get_dgam_p(const double t) noexcept {
  if (t >= 0.0 && t < 22.0)
    return 0.0;
  else if (t >= 22.0 && t < 27.0)
    return 1.5e-2;
  else if (t >= 27.0 && t < 77.0)
    return 0.0;
  else if (t >= 77.0 && t < 82.0)
    return -1.5e-2;
  else
    return 0.0;
}
inline double get_dgam_k(const double t) noexcept {
  constexpr double A = 0.2 / 57.3;
  constexpr double w = 2 * 3.14 * 1.8;
  constexpr double dt = 20;
  if (t >= 0 && t < dt)
    return (A * w) * t / dt * cos(w * t + 3.14 / 2);
  else
    return A * w * cos(w * t + 3.14 / 2);
}
inline double get_dtheta_k(const double t) noexcept {
  constexpr double A = 0.1 / 57.3;
  constexpr double w = 2 * 3.14 * 3.7;
  constexpr double dt = 20;
  if (t >= 0 && t < dt)
    return (A * w) * t / dt * cos(w * t + 3.14 / 3);
  else
    return A * w * cos(w * t + 3.14 / 3);
}
inline double get_dpsi_k(const double t) noexcept {
  constexpr double A = 2e-3;
  constexpr double w = 2 * 3.14 * 1.5;
  constexpr double dt = 20;
  if (t >= 0 && t < dt)
    return (A * w) * t / dt * cos(w * t);
  else
    return A * w * cos(w * t);
}
inline double get_a_lm(const double t) noexcept {
  constexpr double A = 0.3;
  constexpr double w = 2 * 3.14 * 1.2;
  constexpr double dt = 20;
  if (t >= 0 && t < dt)
    return A / dt * t * cos(w * t + 3.14 / 4);
  else
    return A * cos(w * t + 3.14 / 4);
}
