#include <array>
#include <boost/numeric/odeint.hpp>
#include <cmath>
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
template <std::size_t N> struct PI final {
  const std::size_t a_integral;
  static constexpr std::size_t dim = 1;
  double k_p{0.0};
  double inv_T_i{0.0};
  double k_aw{0.0};
  double u_max{0.0};
  double u_min{0.0};
  double in{0.0};
  double out{0.0};
  double u_add{0.0};
  PI(std::size_t &ss_p_)
      : a_integral([](std::size_t &ss_p) { return ss_p++; }(ss_p_)) {};
  void operator()(const std::array<double, N> &x, std::array<double, N> &dx,
                  const double t) {
    dx[a_integral] = in;
    out = k_p * in + k_p * inv_T_i * x[a_integral] + u_add;
    if (out > u_max) {
      dx[a_integral] += -k_aw * (out - u_max);
      out = u_max;
    } else if (out < u_min) {
      dx[a_integral] += -k_aw * (out - u_min);
      out = u_min;
    }
  };
  void write_state_names(std::array<std::string, N> &name_arr,
                         const std::string prefix) {
    name_arr[a_integral] = prefix + "_" + "integral";
  };
};
template <std::size_t N> struct DC_MOTOR final {
  const std::size_t a_current;
  static constexpr std::size_t dim = 1;
  double in_v{0.0};
  double c_m{0.0};
  double c_e{0.0};
  double r{0.0};
  double l{0.0};
  double omega{0.0};
  double out_trq{0.0};
  DC_MOTOR(std::size_t &ss_p_)
      : a_current([](std::size_t &ss_p) { return ss_p++; }(ss_p_)) {};
  void operator()(const std::array<double, N> &x, std::array<double, N> &dx,
                  const double t) {
    dx[a_current] = (in_v - r * x[a_current] - c_e * omega) / l;
    out_trq = c_m * x[a_current];
  };
  void write_state_names(std::array<std::string, N> &name_arr,
                         const std::string prefix) {
    name_arr[a_current] = prefix + "_" + "current";
  };
};
//=========================================================================
// Model of the gyro vertical (single-axis vertical gyro stabilized platform)
struct GyroVertical final {
  static constexpr std::size_t ss_dim =
      IIR2<0>::dim * 6 + PI<0>::dim * 2 + DC_MOTOR<0>::dim * 2 + 13;
  using state_vector_t = std::array<double, ss_dim>;
  std::array<std::string, ss_dim> state_names;
  std::size_t state_addr_init;
  // Dynamic subsystems
  IIR2<ss_dim> diff_psi;
  IIR2<ss_dim> diff_theta;
  IIR2<ss_dim> diff_gam;
  IIR2<ss_dim> stab_loop_filter;
  IIR2<ss_dim> gd_feedback_filter;
  IIR2<ss_dim> pitch_filter;
  PI<ss_dim> stab_loop_pi;
  PI<ss_dim> gd_pi;
  DC_MOTOR<ss_dim> stab_motor;
  DC_MOTOR<ss_dim> gd_motor;
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
  const std::size_t a_omega_gd;
  // Constants
  double Ue{7.29e-5};
  double g{9.81};
  double Re{6371e3};
  // Parameters
  double phi;
  double J1;
  double J2;
  double Jpu;
  double H;
  double Mfx_max;
  double Mfz_max;
  double ml;
  double b_m;
  double Jrgd;
  double Mfm_max;
  double Mf_gd_max;
  double k_dc_k;
  double c_m_k;
  double Mdx;
  double Mdz;
  double u_cor_max;
  bool is_gyn_tun_on = true;
  bool is_earth_rate_corr_on = true;
  bool is_pitch_corr_on = true;
  bool is_vel_error_on = false;
  // Logger
  std::shared_ptr<std::array<double, ss_dim + 3>> log_arr;
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
  // Trajectory planning
  double get_V(const double t) const noexcept;
  double get_dtheta_p(const double t) const noexcept;
  double get_dpsi_p(const double t) const noexcept;
  double get_dgam_p(const double t) const noexcept;
  double get_dgam_k(const double t) const noexcept;
  double get_dtheta_k(const double t) const noexcept;
  double get_dpsi_k(const double t) const noexcept;
  double get_a_lm(const double t) const noexcept;
  GyroVertical()
      : state_addr_init(0), diff_psi(state_addr_init),
        diff_theta(state_addr_init), diff_gam(state_addr_init),
        stab_loop_filter(state_addr_init), gd_feedback_filter(state_addr_init),
        pitch_filter(state_addr_init), stab_loop_pi(state_addr_init),
        gd_pi(state_addr_init), stab_motor(state_addr_init),
        gd_motor(state_addr_init),
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
        a_dalpha_m([](std::size_t &ss_p) { return ss_p++; }(state_addr_init)),
        a_omega_gd([](std::size_t &ss_p) { return ss_p++; }(state_addr_init)) {
    // Set names for state variables
    diff_psi.write_state_names(state_names, "diff_x");
    diff_theta.write_state_names(state_names, "diff_y");
    diff_gam.write_state_names(state_names, "diff_z");
    stab_loop_filter.write_state_names(state_names, "stab_loop_flt");
    stab_motor.write_state_names(state_names, "stab_motor");
    gd_feedback_filter.write_state_names(state_names, "gd_fb_flt");
    pitch_filter.write_state_names(state_names, "flt_pitch");
    stab_loop_pi.write_state_names(state_names, "stab_pi");
    gd_pi.write_state_names(state_names, "gd_pi");
    gd_motor.write_state_names(state_names, "gd_motor");
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
    state_names[a_dalpha_m] = "d_alpha_m";
    state_names[a_omega_gd] = "omega_gd";
    // Set logger
    log_arr = std::make_shared<std::array<double, ss_dim + 3>>();
    log_header_str = std::make_shared<std::string>();
    *log_header_str += "time,V,u_k";
    for (const auto &name : state_names)
      *log_header_str += std::string(",") + std::string(name);
    *log_header_str += "\n";
  };
  void operator()(const state_vector_t &x, state_vector_t &dx, const double t) {
    // Trajectory operator-rameters
    const double V = get_V(t);
    const double a_lm = get_a_lm(t);
    diff_psi.in = get_dpsi_p(t) + get_dpsi_k(t);
    diff_psi(x, dx, t);
    diff_theta.in = get_dtheta_p(t) + get_dtheta_k(t);
    diff_theta(x, dx, t);
    diff_gam.in = get_dgam_p(t) + get_dgam_k(t);
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
    pitch_filter.in = x[a_theta];
    pitch_filter(x, dx, t);
    stab_loop_filter.in = x[a_beta];
    if (is_pitch_corr_on)
      stab_loop_filter.in += pitch_filter.out;
    stab_loop_filter(x, dx, t);
    stab_loop_pi.in = stab_loop_filter.out;
    stab_loop_pi(x, dx, t);
    stab_motor.in_v = stab_loop_pi.out;
    stab_motor.omega = x[a_dalpha];
    stab_motor(x, dx, t);
    // Friction model of the X axis
    const double sum_trq_x =
        -J1 * dw_x +
        H * (x[a_dbeta] - w_y * sin(x[a_alpha]) + w_z * cos(x[a_alpha])) *
            cos(x[a_beta]) +
        stab_motor.out_trq + Mdx;
    double Mfx = 0.0;
    if (std::abs(x[a_dalpha]) < 1e-10)
      Mfx = std::abs(sum_trq_x) > Mfx_max ? std::copysign(Mfx_max, sum_trq_x)
                                          : sum_trq_x;
    else
      Mfx = 0.9 * std::copysign(Mfx_max, x[a_dalpha]);
    dx[a_dalpha] = (sum_trq_x - Mfx) / J1;
    // Correction loop
    double u_k = k_dc_k * (x[a_alpha_m] - x[a_alpha]);
    if (is_earth_rate_corr_on)
      u_k += H * Ue * cos(phi) * cos(x[a_psi]) / c_m_k;
    u_k = std::abs(u_k) > u_cor_max ? std::copysign(u_cor_max, u_k) : u_k;
    // Friction model of the Z axis
    const double sum_trq_z =
        -J2 * dw_z * cos(x[a_alpha]) + J2 * dw_y * sin(x[a_alpha]) -
        H * (cos(x[a_beta]) * (x[a_dalpha] + w_x) +
             sin(x[a_beta]) * (w_y * cos(x[a_alpha]) + w_z * sin(x[a_alpha]))) +
        u_k * c_m_k + Mdz;
    double Mfz = 0.0;
    if (std::abs(x[a_dbeta]) < 1e-10)
      Mfz = std::abs(sum_trq_z) > Mfz_max ? std::copysign(Mfz_max, sum_trq_z)
                                          : sum_trq_z;
    else
      Mfz = 0.9 * std::copysign(Mfz_max, x[a_dbeta]);
    dx[a_dbeta] = (sum_trq_z - Mfz) / J2;
    // Gyro pendulum
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
    const double H_m = x[a_omega_gd] * Jrgd;
    // Friction model of the gyro pendulum suspension axis
    const double sum_trq_m =
        -Jpu * dw_x -
        H_m * (w_y * cos(x[a_alpha_m]) + w_z * sin(x[a_alpha_m])) -
        b_m * (x[a_dalpha_m] - x[a_dalpha]) +
        ml * (a_l * cos(x[a_gam] + x[a_alpha_m]) -
              a_v * sin(x[a_gam] + x[a_alpha_m]));
    double Mfm = 0.0;
    if (std::abs(x[a_dalpha_m] - x[a_dalpha]) < 1e-10)
      Mfm = std::abs(sum_trq_m) > Mfm_max ? std::copysign(Mfm_max, sum_trq_m)
                                          : sum_trq_m;
    else
      Mfm = 0.9 * std::copysign(Mfm_max, x[a_dalpha_m] - x[a_dalpha]);
    dx[a_dalpha_m] = (sum_trq_m - Mfm) / Jpu;
    // Gyro pendulum's dynamic tuning
    gd_feedback_filter.in = x[a_omega_gd];
    gd_feedback_filter(x, dx, t);
    double e_gd = 0.0;
    if (is_gyn_tun_on) {
      if (is_vel_error_on) {
        e_gd = -ml * V * 1.002 / Jrgd - gd_feedback_filter.out;
      } else {
        e_gd = -ml * V / Jrgd - gd_feedback_filter.out;
      }
    }
    gd_pi.in = e_gd;
    gd_pi.u_add = gd_motor.c_e * gd_feedback_filter.out;
    gd_pi(x, dx, t);
    gd_motor.in_v = gd_pi.out;
    gd_motor.omega = x[a_omega_gd];
    gd_motor(x, dx, t);
    // Friction model of the gyropendulum's rotor suspension axis
    const double sum_trq_gd = gd_motor.out_trq;
    double Mf_gd = 0.0;
    if (std::abs(x[a_omega_gd]) < 1e-10)
      Mf_gd = std::abs(sum_trq_gd) > Mf_gd_max
                  ? std::copysign(Mf_gd_max, sum_trq_gd)
                  : sum_trq_gd;
    else
      Mf_gd = 0.9 * std::copysign(Mf_gd_max, x[a_omega_gd]);
    dx[a_omega_gd] = (sum_trq_gd - Mf_gd) / Jrgd;
    // Logging
    (*log_arr)[0] = t;
    (*log_arr)[1] = V;
    (*log_arr)[2] = u_k;
    memcpy(&(*log_arr)[3], &x[0], sizeof(double) * ss_dim);
  };
};
