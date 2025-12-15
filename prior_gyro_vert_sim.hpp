#include <array>
#include <boost/numeric/odeint.hpp>
#include <cmath>
#include <cstdio>
#include <string>

// Tools for auto differentiation
template <std::size_t N> struct AutoDiffDualVar final {
  double val = 0;
  std::array<double, N> grad{};
  AutoDiffDualVar() = default;
  AutoDiffDualVar(double v_) : val(v_) {};
  AutoDiffDualVar(double v_, double grad_, std::size_t index) : val(v_) {
    grad.at(index) = grad_;
  };
  double get_full_diff() const {
    double res = 0;
    for (const auto &gr : grad)
      res += gr;
    return res;
  };
};
template <std::size_t N>
AutoDiffDualVar<N> operator+(const AutoDiffDualVar<N> &a,
                             const AutoDiffDualVar<N> &b) {
  AutoDiffDualVar<N> result;
  result.val = a.val + b.val;
  for (size_t i = 0; i < N; ++i) {
    result.grad[i] = a.grad[i] + b.grad[i];
  }
  return result;
}
template <std::size_t N>
AutoDiffDualVar<N> operator-(const AutoDiffDualVar<N> &a,
                             const AutoDiffDualVar<N> &b) {
  AutoDiffDualVar<N> result;
  result.val = a.val - b.val;
  for (size_t i = 0; i < N; ++i) {
    result.grad[i] = a.grad[i] - b.grad[i];
  }
  return result;
}
template <size_t N>
AutoDiffDualVar<N> operator*(const AutoDiffDualVar<N> &a,
                             const AutoDiffDualVar<N> &b) {
  AutoDiffDualVar<N> result;
  result.val = a.val * b.val;
  for (size_t i = 0; i < N; ++i)
    result.grad[i] = a.grad[i] * b.val + a.val * b.grad[i];
  return result;
}
template <std::size_t N> AutoDiffDualVar<N> sin(const AutoDiffDualVar<N> &x) {
  AutoDiffDualVar<N> result;
  result.val = std::sin(x.val);
  for (std::size_t i = 0; i < N; ++i)
    result.grad[i] = std::cos(x.val) * x.grad[i];
  return result;
}
template <std::size_t N> AutoDiffDualVar<N> cos(const AutoDiffDualVar<N> &x) {
  AutoDiffDualVar<N> result;
  result.val = std::cos(x.val);
  for (std::size_t i = 0; i < N; ++i)
    result.grad[i] = -std::sin(x.val) * x.grad[i];
  return result;
}
template <std::size_t N>
AutoDiffDualVar<N> operator+(const double cnst, const AutoDiffDualVar<N> &var) {
  return AutoDiffDualVar<N>(cnst) + var;
}
template <std::size_t N>
AutoDiffDualVar<N> operator+(const AutoDiffDualVar<N> &var, const double cnst) {
  return AutoDiffDualVar<N>(cnst) + var;
}
template <std::size_t N>
AutoDiffDualVar<N> operator-(const double cnst, const AutoDiffDualVar<N> &var) {
  return AutoDiffDualVar<N>(cnst) - var;
}
template <std::size_t N>
AutoDiffDualVar<N> operator-(const AutoDiffDualVar<N> &var, const double cnst) {
  return var - AutoDiffDualVar<N>(cnst);
}
template <std::size_t N>
AutoDiffDualVar<N> operator*(const double cnst, const AutoDiffDualVar<N> &var) {
  return AutoDiffDualVar<N>(cnst) * var;
}
template <std::size_t N>
AutoDiffDualVar<N> operator*(const AutoDiffDualVar<N> &var, const double cnst) {
  return AutoDiffDualVar<N>(cnst) * var;
}
//==============================================================================
// Dynamic subsystems
template <std::size_t N> struct IIR2 final {
  // Controllable canonical form
  // W(s) = d + (b1 * s + b0) / (s**2 + a1 * s + a0)
  const std::size_t a_x0;
  const std::size_t a_x1;
  static constexpr std::size_t dim = 2;
  double b1{0.0};
  double b0{0.0};
  double a1{0.0};
  double a0{0.0};
  double d{0.0};
  double in{0.0};
  double out{0.0};
  IIR2(std::size_t &ss_p_)
      : a_x0([](std::size_t &ss_p) { return ss_p++; }(ss_p_)),
        a_x1([](std::size_t &ss_p) { return ss_p++; }(ss_p_)) {};
  void operator()(const std::array<double, N> &x, std::array<double, N> &dx,
                  const double t) {
    dx[a_x0] = x[a_x1];
    dx[a_x1] = in - a0 * x[a_x0] - a1 * x[a_x1];
    out = d * in + b0 * x[a_x0] + b1 * x[a_x1];
  };
  void write_state_names(std::array<std::string, N> &name_arr,
                         const std::string &prefix) {
    name_arr[a_x0] = prefix + "_" + "x0";
    name_arr[a_x1] = prefix + "_" + "x1";
  };
};
//=========================================================================
// Model of the gyro vertical (single-axis vertical gyro stabilized platform)
struct GyroVertical final {
  static constexpr std::size_t ss_dim = IIR2<0>::dim * 3 + 12;
  using state_vector_t = std::array<double, ss_dim>;
  std::array<std::string, ss_dim> state_names;
  std::size_t state_addr_init;
  // Dynamic subsystems
  IIR2<ss_dim> diff_psi;
  IIR2<ss_dim> diff_theta;
  IIR2<ss_dim> diff_gam;
  // Addresses of the state variables
  const std::size_t a_psi;
  const std::size_t a_theta;
  const std::size_t a_gam;
  const std::size_t a_dpsi;
  const std::size_t a_dtheta;
  const std::size_t a_dgam;
  const std::size_t a_alpha;
  const std::size_t a_dalpha;
  const std::size_t a_beta;
  const std::size_t a_dbeta;
  const std::size_t a_alpha_m;
  const std::size_t a_dalpha_m;
  // Trajectory function pointers
  double (*p_getV)(const double t);
  double (*p_get_dtheta_p)(const double t);
  double (*p_get_dpsi_p)(const double t);
  double (*p_get_dgam_p)(const double t);
  double (*p_get_dtheta_k)(const double t);
  double (*p_get_dgam_k)(const double t);
  double (*p_get_dpsi_k)(const double t);
  double (*p_get_a_lm)(const double t);
  double get_V(const double t) const noexcept {
    return (p_getV) ? p_getV(t) : 0.0;
  };
  double get_dtheta_p(const double t) const noexcept {
    return (p_get_dtheta_p) ? p_get_dtheta_p(t) : 0.0;
  };
  double get_dpsi_p(const double t) const noexcept {
    return (p_get_dpsi_p) ? p_get_dpsi_p(t) : 0.0;
  };
  double get_dgam_p(const double t) const noexcept {
    return (p_get_dgam_p) ? p_get_dgam_p(t) : 0.0;
  };
  double get_dgam_k(const double t) const noexcept {
    return (p_get_dgam_k) ? p_get_dgam_k(t) : 0.0;
  };
  double get_dtheta_k(const double t) const noexcept {
    return (p_get_dtheta_k) ? p_get_dtheta_k(t) : 0.0;
  };
  double get_dpsi_k(const double t) const noexcept {
    return (p_get_dpsi_k) ? p_get_dpsi_k(t) : 0.0;
  };
  double get_a_lm(const double t) const noexcept {
    return (p_get_a_lm) ? p_get_a_lm(t) : 0.0;
  };
  // Constants
  double Ue{7.29e-5};
  double g{9.81};
  double Re{6371e3};
  // Parameters
  bool to_turn;
  bool to_pitch;
  bool is_tilt_corr_on;
  double phi;
  double J1;
  double Jr_s;
  double Jr_k;
  double J2;
  double H;
  double k_m_s;
  double k_m_k;
  double k_d_s;
  double k_d_k;
  double q_s;
  double q_k;
  double k_s;
  double k_k;
  double Mfx_max;
  double Mfz_max;
  double Mdx;
  double Mdz;
  double u_cor_max;
  double u_stab_max;
  double t_m;
  double t_md;
  // Logger
  std::shared_ptr<std::array<double, ss_dim + 4>> log_arr;
  std::shared_ptr<std::string> log_header_str;
  // Functions of the rate projections onto the b-frame
  template <typename T>
  T get_w_x(const T &psi, const T &theta, const T &gam, const T &d_psi,
            const T &d_theta, const T &d_gam, const double &V) {
    return d_gam - d_psi * sin(theta) +
           Ue * (sin(phi) * sin(theta) + cos(phi) * cos(psi) * cos(theta)) +
           (V / Re) * cos(theta) * tan(phi) * sin(psi) * sin(theta);
  };
  template <typename T>
  T get_w_y(const T &psi, const T &theta, const T &gam, const T &d_psi,
            const T &d_theta, const T &d_gam, const double &V) {
    return d_theta * sin(gam) - d_psi * cos(theta) * cos(gam) +
           Ue * (sin(phi) * cos(gam) * cos(theta) -
                 cos(phi) *
                     (sin(gam) * sin(psi) + cos(gam) * cos(psi) * sin(theta))) +
           (V / Re) * cos(theta) *
               (tan(phi) * sin(psi) * cos(theta) * cos(gam) - sin(gam));
  };
  template <typename T>
  T get_w_z(const T &psi, const T &theta, const T &gam, const T &d_psi,
            const T &d_theta, const T &d_gam, const double &V) {
    return d_theta * cos(gam) + d_psi * cos(theta) * sin(gam) +
           Ue * (cos(phi) *
                     (sin(theta) * sin(gam) * cos(psi) - cos(gam) * sin(psi)) -
                 sin(phi) * sin(gam) * cos(theta)) -
           (V / Re) * cos(theta) *
               (tan(phi) * sin(psi) * cos(theta) * sin(gam) + cos(gam));
  };
  GyroVertical()
      : state_addr_init(0), diff_psi(state_addr_init),
        diff_theta(state_addr_init), diff_gam(state_addr_init),
        a_psi([](std::size_t &ss_p) { return ss_p++; }(state_addr_init)),
        a_theta([](std::size_t &ss_p) { return ss_p++; }(state_addr_init)),
        a_gam([](std::size_t &ss_p) { return ss_p++; }(state_addr_init)),
        a_dpsi([](std::size_t &ss_p) { return ss_p++; }(state_addr_init)),
        a_dtheta([](std::size_t &ss_p) { return ss_p++; }(state_addr_init)),
        a_dgam([](std::size_t &ss_p) { return ss_p++; }(state_addr_init)),
        a_alpha([](std::size_t &ss_p) { return ss_p++; }(state_addr_init)),
        a_dalpha([](std::size_t &ss_p) { return ss_p++; }(state_addr_init)),
        a_beta([](std::size_t &ss_p) { return ss_p++; }(state_addr_init)),
        a_dbeta([](std::size_t &ss_p) { return ss_p++; }(state_addr_init)),
        a_alpha_m([](std::size_t &ss_p) { return ss_p++; }(state_addr_init)),
        a_dalpha_m([](std::size_t &ss_p) { return ss_p++; }(state_addr_init)) {
    // Set names for state variables
    diff_psi.write_state_names(state_names, "diff_x");
    diff_theta.write_state_names(state_names, "diff_y");
    diff_gam.write_state_names(state_names, "diff_z");
    state_names[a_psi] = "psi";
    state_names[a_theta] = "theta";
    state_names[a_gam] = "gam";
    state_names[a_alpha] = "alpha";
    state_names[a_dpsi] = "dpsi";
    state_names[a_dtheta] = "dtheta";
    state_names[a_dgam] = "dgam";
    state_names[a_dalpha] = "dalpha";
    state_names[a_beta] = "beta";
    state_names[a_dbeta] = "dbeta";
    state_names[a_alpha_m] = "alpha_m";
    state_names[a_dalpha_m] = "dalpha_m";
    // Set logger
    log_arr = std::make_shared<std::array<double, ss_dim + 4>>();
    log_header_str = std::make_shared<std::string>();
    *log_header_str += "time,V,u_k,u_s";
    for (const auto &name : state_names)
      *log_header_str += std::string(",") + std::string(name);
    *log_header_str += "\n";
  };
  void operator()(const state_vector_t &x, state_vector_t &dx, const double t) {
    // Trajectory operator-rameters
    const double V = get_V(t);
    const double a_lm = get_a_lm(t);
    diff_psi.in = get_dpsi_p(t) * to_turn + get_dpsi_k(t);
    diff_psi(x, dx, t);
    diff_theta.in = get_dtheta_p(t) * to_pitch + get_dtheta_k(t);
    diff_theta(x, dx, t);
    diff_gam.in = get_dgam_p(t) * to_turn + get_dgam_k(t);
    diff_gam(x, dx, t);
    dx[a_dpsi] = diff_psi.out;
    dx[a_dtheta] = diff_theta.out;
    dx[a_dgam] = diff_gam.out;
    dx[a_psi] = x[a_dpsi];
    dx[a_theta] = x[a_dtheta];
    dx[a_gam] = x[a_dgam];
    // Angular rate projections onto the b-frame and their derivaties
    AutoDiffDualVar<6> psi_(x[a_psi], x[a_dpsi], 0);
    AutoDiffDualVar<6> theta_(x[a_theta], x[a_dtheta], 1);
    AutoDiffDualVar<6> gam_(x[a_gam], x[a_dgam], 2);
    AutoDiffDualVar<6> dpsi_(x[a_dpsi], dx[a_dpsi], 3);
    AutoDiffDualVar<6> dtheta_(x[a_dtheta], dx[a_dtheta], 4);
    AutoDiffDualVar<6> dgam_(x[a_dgam], dx[a_dgam], 5);
    auto w_x_ = get_w_x(psi_, theta_, gam_, dpsi_, dtheta_, dgam_, V);
    auto w_y_ = get_w_y(psi_, theta_, gam_, dpsi_, dtheta_, dgam_, V);
    auto w_z_ = get_w_z(psi_, theta_, gam_, dpsi_, dtheta_, dgam_, V);
    const double w_x = w_x_.val;
    const double w_y = w_y_.val;
    const double w_z = w_z_.val;
    const double dw_x = w_x_.get_full_diff();
    const double dw_y = w_y_.get_full_diff();
    const double dw_z = w_z_.get_full_diff();
    // Dynamics equations
    dx[a_alpha] = x[a_dalpha];
    dx[a_beta] = x[a_dbeta];
    dx[a_alpha_m] = x[a_dalpha_m];
    // Stablilization loop
    const double tilting_coeff = k_s * k_k * k_m_k * q_k / g / H;
    double u_s = k_s * x[a_beta] + is_tilt_corr_on * tilting_coeff * V;
    u_s = std::abs(u_s) > u_stab_max ? std::copysign(u_stab_max, u_s) : u_s;
    // Friction model of the X axis
    const double sum_trq_x =
        -J1 * dw_x + Jr_s * q_s * q_s * dw_x +
        H * (x[a_dbeta] - w_y * sin(x[a_alpha]) + w_z * cos(x[a_alpha])) *
            cos(x[a_beta]) +
        u_s * k_m_s * q_s + Mdx;
    double Mfx = 0.0;
    if (std::abs(x[a_dalpha]) < 1e-10)
      Mfx = std::abs(sum_trq_x) > Mfx_max ? std::copysign(Mfx_max, sum_trq_x)
                                          : sum_trq_x;
    else
      Mfx = 0.9 * std::copysign(Mfx_max, x[a_dalpha]);
    dx[a_dalpha] = (sum_trq_x - Mfx) / J1;
    // Pendulum sensitive element
    // Transverse lateral acceleration
    double a_l =
        V * cos(x[a_theta]) *
            (x[a_dpsi] - V / Re * tan(phi) * sin(x[a_psi]) * cos(x[a_theta])) -
        2 * Ue * V *
            (sin(phi) * cos(x[a_theta]) -
             cos(phi) * sin(x[a_theta]) * cos(x[a_psi])) +
        a_lm;
    // Transverse vertical acceleration
    const double a_v = g * cos(x[a_theta]) +
                       V * (x[a_dtheta] - V * cos(x[a_theta]) / Re) -
                       2 * Ue * V * cos(phi) * sin(x[a_psi]);
    dx[a_dalpha_m] = -dw_x * t_m * t_m - (x[a_dalpha_m] - x[a_dalpha]) * t_md +
                     1 / g *
                         (a_l * cos(x[a_gam] + x[a_alpha_m]) -
                          a_v * sin(x[a_gam] + x[a_alpha_m]));
    dx[a_dalpha_m] *= 1 / t_m / t_m;
    // Correction loop
    double u_k = k_k * (x[a_alpha_m] - x[a_alpha]);
    u_k = std::abs(u_k) > u_cor_max ? std::copysign(u_cor_max, u_k) : u_k;
    // Friction model of the Z axis
    const double sum_trq_z =
        -J2 * (dw_z * cos(x[a_alpha]) - dw_y * sin(x[a_alpha])) +
        Jr_k * q_k * q_k * (dw_z * cos(x[a_alpha]) - dw_y * sin(x[a_alpha])) -
        H * (cos(x[a_beta]) * (x[a_dalpha] + w_x) +
             sin(x[a_beta]) * (w_y * cos(x[a_alpha]) + w_z * sin(x[a_alpha]))) +
        u_k * k_m_k * q_k - k_d_k * q_k * q_k * x[a_dbeta] + Mdz;
    double Mfz = 0.0;
    if (std::abs(x[a_dbeta]) < 1e-10)
      Mfz = std::abs(sum_trq_z) > Mfz_max ? std::copysign(Mfz_max, sum_trq_z)
                                          : sum_trq_z;
    else
      Mfz = 0.9 * std::copysign(Mfz_max, x[a_dbeta]);
    dx[a_dbeta] = (sum_trq_z - Mfz) / J2;
    // Logging
    (*log_arr)[0] = t;
    (*log_arr)[1] = V;
    (*log_arr)[2] = u_k;
    (*log_arr)[3] = u_s;
    memcpy(&(*log_arr)[4], &x[0], sizeof(double) * ss_dim);
  };
};
