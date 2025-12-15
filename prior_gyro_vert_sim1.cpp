#include "prior_gyro_vert_sim.hpp"
#include <algorithm>
#include <boost/numeric/odeint.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <ostream>
#include <string>
#include <vector>

double GyroVertical::get_V(const double t) const noexcept {
  if (t >= 0.0 && t < 27.75)
    return 0.8 * t;
  else
    return 22.2;
}
double GyroVertical::get_dtheta_p(const double t) const noexcept {
  if (t >= 0.0 && t < 15.0)
    return 0.0;
  else if (t >= 15.0 && t < 15.0 + 3.4)
    return 7.4e-3;
  else
    return 0.0;
}
double GyroVertical::get_dpsi_p(const double t) const noexcept {
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
double GyroVertical::get_dgam_p(const double t) const noexcept {
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
double GyroVertical::get_dgam_k(const double t) const noexcept {
  constexpr double A = 0.2 / 57.3;
  constexpr double w = 2 * 3.14 * 1.5;
  constexpr double dt = 20;
  if (t >= 0 && t < dt)
    return (A * w) * t / dt * cos(w * t + 3.14 / 2);
  else
    return A * w * cos(w * t + 3.14 / 2);
}
double GyroVertical::get_dtheta_k(const double t) const noexcept {
  constexpr double A = 0.02 / 57.3;
  constexpr double w = 3 * 3.14 * 2;
  constexpr double dt = 20;
  if (t >= 0 && t < dt)
    return (A * w) * t / dt * cos(w * t - 3.14 / 3);
  else
    return A * w * cos(w * t - 3.14 / 3);
}
double GyroVertical::get_dpsi_k(const double t) const noexcept {
  constexpr double A = 2e-3;
  constexpr double w = 2 * 3.14 * 2.5;
  constexpr double dt = 20;
  if (t >= 0 && t < dt)
    return (A * w) * t / dt * cos(w * t + 3.14 / 6);
  else
    return A * w * cos(w * t + 3.14 / 6);
}
double GyroVertical::get_a_lm(const double t) const noexcept {
  constexpr double A = 0.3;
  constexpr double w = 1.0 * 2 * 3.14;
  constexpr double dt = 20;
  if (t >= 0 && t < dt)
    return A / dt * t * cos(w * t + 3.14 / 4);
  else
    return A * cos(w * t + 3.14 / 4);
}

typedef boost::numeric::odeint::runge_kutta_dopri5<GyroVertical::state_vector_t>
    error_stepper_type;

int main(const int argc, const char *argv[]) {
  // Default parameters
  bool to_turn = true;
  bool to_pitch = true;
  bool is_corr_on = true;
  // Read input flags
  std::vector<std::string> args(argv, argv + argc);
  bool is_help =
      (std::find(args.begin(), args.end(), "--help") != args.end()) ||
      (std::find(args.begin(), args.end(), "-h") != args.end()) || (argc == 1);
  if (is_help) {
    std::cout << argc << "\n";
    std::cout << "Available options" << "\n";
    std::cout << "-h, --help : to this message" << "\n";
    std::cout << "-t, --turn [ON / OFF] : modlling a turn"
              << "\n";
    std::cout << "-p, --pitch [ON / OFF] : modlling an inclined path"
              << "\n";
    std::cout << "-c, --correction [ON / OFF] : turning error correction"
              << "\n";
    std::cout << "-o, --output [file_name] : specifies the name of the output "
                 "file of the simulation"
              << "\n";
    return 0;
  }
  auto it_turning = std::find(args.begin(), args.end(), "--turn");
  if (it_turning == args.end())
    it_turning = std::find(args.begin(), args.end(), "-t");
  if (it_turning != args.end()) {
    if (std::next(it_turning) == args.end()) {
      std::cout << "A required argument was not provided";
      return 1;
    } else if (*std::next(it_turning) == "ON")
      to_turn = true;
    else if (*std::next(it_turning) == "OFF")
      to_turn = false;
    else {
      std::cout << "An unknown argument was provided";
      return 1;
    }
  }
  auto it_pitch = std::find(args.begin(), args.end(), "--pitch");
  if (it_pitch == args.end())
    it_pitch = std::find(args.begin(), args.end(), "-p");
  if (it_pitch != args.end()) {
    if (std::next(it_pitch) == args.end()) {
      std::cout << "A required argument was not provided";
      return 1;
    } else if (*std::next(it_pitch) == "ON")
      to_pitch = true;
    else if (*std::next(it_pitch) == "OFF")
      to_pitch = false;
    else {
      std::cout << "An unknown argument was provided";
      return 1;
    }
  }
  auto it_corr = std::find(args.begin(), args.end(), "--correction");
  if (it_corr == args.end())
    it_corr = std::find(args.begin(), args.end(), "-c");
  if (it_corr != args.end()) {
    if (std::next(it_corr) == args.end()) {
      std::cout << "A required argument was not provided";
      return 1;
    } else if (*std::next(it_corr) == "ON")
      is_corr_on = true;
    else if (*std::next(it_corr) == "OFF")
      is_corr_on = false;
    else {
      std::cout << "An unknown argument was provided";
      return 1;
    }
  }
  std::string file_name("sim_gv.csv");
  auto it_output = std::find(args.begin(), args.end(), "--output");
  if (it_output == args.end())
    it_output = std::find(args.begin(), args.end(), "-o");
  if (it_output != args.end()) {
    if (std::next(it_output) == args.end()) {
      std::cout << "A required argument was not provided";
      return 1;
    } else {
      file_name = *std::next(it_output);
    }
  }
  // Initialize system
  GyroVertical gyro_vert;
  // Parameters settings
  gyro_vert.to_turn = to_turn;
  gyro_vert.to_pitch = to_pitch;
  gyro_vert.is_tilt_corr_on = is_corr_on;
  const double T_diff = 1e-2;
  const double ksi_diff = 0.707;
  gyro_vert.diff_psi.a0 = 1 / (T_diff * T_diff);
  gyro_vert.diff_psi.a1 = 2 / T_diff * ksi_diff;
  gyro_vert.diff_psi.b1 = gyro_vert.diff_psi.a0;
  gyro_vert.diff_theta.a0 = 1 / (T_diff * T_diff);
  gyro_vert.diff_theta.a1 = 2 / T_diff * ksi_diff;
  gyro_vert.diff_theta.b1 = gyro_vert.diff_theta.a0;
  gyro_vert.diff_gam.a0 = 1 / (T_diff * T_diff);
  gyro_vert.diff_gam.a1 = 2 / T_diff * ksi_diff;
  gyro_vert.diff_gam.b1 = gyro_vert.diff_gam.a0;
  gyro_vert.phi = 55.75 / 57.3;
  // Gyro vertival paramters
  gyro_vert.J1 = 1e-2;
  gyro_vert.J2 = 1.46e-3;
  gyro_vert.H = 2;
  gyro_vert.k_m_s = 2.6e-4;
  gyro_vert.k_d_s = 9.6e-6;
  gyro_vert.k_m_k = 5e-5;
  gyro_vert.k_d_s = 7.96e-7;
  gyro_vert.q_s = 200.0;
  gyro_vert.q_k = 20.0;
  gyro_vert.k_s = 100.0;
  gyro_vert.k_k = 200.0;
  gyro_vert.Mfx_max = 1.0;
  gyro_vert.Mfz_max = 1.3e-4;
  gyro_vert.Mdx = 0.02;
  gyro_vert.Mdz = 2e-5;
  gyro_vert.u_cor_max = 30.0;
  gyro_vert.u_stab_max = 30.0;
  gyro_vert.t_m = 1.0 / 25.0;
  gyro_vert.t_md = 1.0;
  // Initial value of a state space vector
  GyroVertical::state_vector_t x0{};
  const double t_start = 0.0;
  const double t_end = 100.0;
  double t_trig = 0;
  // Set logging
  std::ofstream log_file(file_name);
  log_file << *gyro_vert.log_header_str;
  // Integrate
  auto perf_time_start = std::chrono::steady_clock::now();
  size_t steps = boost::numeric::odeint::integrate_const(
      boost::numeric::odeint::make_dense_output<error_stepper_type>(5e-5, 5e-5),
      gyro_vert, x0, t_start, t_end, 0.001,
      [&gyro_vert, &log_file, &t_trig,
       t_end](const GyroVertical::state_vector_t &x, double t) {
        bool is_first = true;
        for (const auto &val : *(gyro_vert.log_arr)) {
          if (is_first) {
            is_first = false;
            log_file << val;
          } else {
            log_file << "," << val;
          }
        }
        log_file << "\n";
        if (t > t_trig) {
          std::cout << "Progress: " << int(t / t_end * 100) << "%" << "\r";
          std::cout.flush();
          t_trig += 1.0;
        }
      });
  auto perf_time_end = std::chrono::steady_clock::now();
  std::cout << "Number of steps: " << steps << "\n";
  std::cout << "Elapsed time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   perf_time_end - perf_time_start)
                   .count()
            << "ms" << std::endl;
  if (argc == 1) {
    std::cout << "Press any key to exit...";
    std::cin.get();
  }
  return 0;
}
