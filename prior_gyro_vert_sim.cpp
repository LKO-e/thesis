#include "prior_gyro_vert_sim.hpp"
#include "gyro_vert_sim_trajectory.hpp"
#include <algorithm>
#include <boost/numeric/odeint.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <ostream>
#include <string>
#include <vector>

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
  // Trajectory
  gyro_vert.p_get_dpsi_p = &get_dpsi_p;
  gyro_vert.p_get_dtheta_p = &get_dtheta_p;
  gyro_vert.p_get_dgam_p = &get_dgam_p;
  gyro_vert.p_get_dpsi_k = &get_dpsi_k;
  gyro_vert.p_get_dtheta_k = &get_dtheta_k;
  gyro_vert.p_get_dgam_k = &get_dgam_k;
  gyro_vert.p_getV = &get_V;
  gyro_vert.p_get_a_lm = &get_a_lm;
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
  gyro_vert.Jr_s = 2.5e-7;
  gyro_vert.Jr_k = 6.7e-8;
  gyro_vert.q_s = 200.0;
  gyro_vert.q_k = 20.0;
  gyro_vert.J1 = 1e-2 + gyro_vert.Jr_s * gyro_vert.q_s * gyro_vert.q_s;
  gyro_vert.J2 = 1.46e-3 + gyro_vert.Jr_k * gyro_vert.q_k * gyro_vert.q_k;
  gyro_vert.H = 2;
  gyro_vert.k_m_s = 2.7e-4;
  gyro_vert.k_d_s = 9.6e-6;
  gyro_vert.k_m_k = 5e-5;
  gyro_vert.k_d_k = 7.96e-7;
  gyro_vert.k_s = 83.4;
  gyro_vert.k_k = 214.0;
  gyro_vert.Mfx_max = 1.0;
  gyro_vert.Mfz_max = 5e-4;
  gyro_vert.Mdx = 0.01;
  gyro_vert.Mdz = -1.5e-4;
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
