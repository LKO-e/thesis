#include "gyro_vert_sim.hpp"
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
  bool is_dyn_tun_on = true;
  bool is_psi_error = false;
  bool is_pitch_corr_on = true;
  bool is_vel_meas_error_on = false;
  // Read input flags
  std::vector<std::string> args(argv, argv + argc);
  bool is_help =
      (std::find(args.begin(), args.end(), "--help") != args.end()) ||
      (std::find(args.begin(), args.end(), "-h") != args.end()) || (argc == 1);
  if (is_help) {
    std::cout << argc << "\n";
    std::cout << "Available options" << "\n";
    std::cout << "-h, --help : to this message" << "\n";
    std::cout << "-d, --dynamic_tuning [ON / OFF] : turns on and off the "
                 "dynamic tuning system of the gyro pendulum"
              << "\n";
    std::cout << "-e, --earth_rate_correction [ON / OFF] : turns on and off "
                 "the Earth's angular velocity correction loop"
              << "\n";
    std::cout << "-p, --pitch_correction [ON / OFF] : turns on and off the "
                 "inclination system for the main gyro"
              << "\n";
    std::cout << "-v, --vel_meas_error [ON / OFF] : turns on and off velocity "
                 "measurement error of 0.2%"
              << "\n";
    std::cout << "-o, --output [file_name] : specifies the name of the output "
                 "file of the simulation"
              << "\n";
    return 0;
  }
  auto it_dyn_tun = std::find(args.begin(), args.end(), "--dynamic_tuning");
  if (it_dyn_tun == args.end())
    it_dyn_tun = std::find(args.begin(), args.end(), "-d");
  if (it_dyn_tun != args.end()) {
    if (std::next(it_dyn_tun) == args.end()) {
      std::cout << "A required argument was not provided";
      return 1;
    } else if (*std::next(it_dyn_tun) == "ON")
      is_dyn_tun_on = true;
    else if (*std::next(it_dyn_tun) == "OFF")
      is_dyn_tun_on = false;
    else {
      std::cout << "An unknown argument was provided";
      return 1;
    }
  }
  auto it_psi_error = std::find(args.begin(), args.end(), "--yaw_error");
  if (it_psi_error == args.end())
    it_psi_error = std::find(args.begin(), args.end(), "-y");
  if (it_psi_error != args.end()) {
    if (std::next(it_psi_error) == args.end()) {
      std::cout << "A required argument was not provided";
      return 1;
    } else if (*std::next(it_psi_error) == "ON")
      is_psi_error = true;
    else if (*std::next(it_psi_error) == "OFF")
      is_psi_error = false;
    else {
      std::cout << "An unknown argument was provided";
      return 1;
    }
  }
  auto it_pitch_corr =
      std::find(args.begin(), args.end(), "--pitch_correction");
  if (it_pitch_corr == args.end())
    it_pitch_corr = std::find(args.begin(), args.end(), "-p");
  if (it_pitch_corr != args.end()) {
    if (std::next(it_pitch_corr) == args.end()) {
      std::cout << "A required argument was not provided";
      return 1;
    } else if (*std::next(it_pitch_corr) == "ON")
      is_pitch_corr_on = true;
    else if (*std::next(it_pitch_corr) == "OFF")
      is_pitch_corr_on = false;
    else {
      std::cout << "An unknown argument was provided";
      return 1;
    }
  }
  auto it_vel_error = std::find(args.begin(), args.end(), "--vel_meas_error");
  if (it_vel_error == args.end())
    it_vel_error = std::find(args.begin(), args.end(), "-v");
  if (it_vel_error != args.end()) {
    if (std::next(it_vel_error) == args.end()) {
      std::cout << "A required argument was not provided";
      return 1;
    } else if (*std::next(it_vel_error) == "ON")
      is_vel_meas_error_on = true;
    else if (*std::next(it_vel_error) == "OFF")
      is_vel_meas_error_on = false;
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
  gyro_vert.is_psi_error = is_psi_error;
  gyro_vert.is_pitch_corr_on = is_pitch_corr_on;
  gyro_vert.is_gyn_tun_on = is_dyn_tun_on;
  gyro_vert.is_vel_error_on = is_vel_meas_error_on;
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
  gyro_vert.stab_loop_filter.a0 = 219.0 * 219.0;
  gyro_vert.stab_loop_filter.a1 = 2.0 * 219.0 * 0.58;
  gyro_vert.stab_loop_filter.b0 = gyro_vert.stab_loop_filter.a0;
  gyro_vert.pitch_filter.a0 = 31.4 * 31.4;
  gyro_vert.pitch_filter.a1 = 2 * 31.4 * 0.707;
  gyro_vert.pitch_filter.b0 = gyro_vert.pitch_filter.a0;
  gyro_vert.stab_motor.c_e = 0.11;
  gyro_vert.stab_motor.c_m = 0.11;
  gyro_vert.stab_motor.l = 9.0e-3;
  gyro_vert.stab_motor.r = 18.0;
  gyro_vert.stab_loop_pi.k_aw = 1.0 / gyro_vert.stab_loop_pi.k_p;
  gyro_vert.stab_loop_pi.inv_T_i = 1.0 / 0.3;
  gyro_vert.stab_loop_pi.k_p = 2.6e3;
  gyro_vert.stab_loop_pi.u_max = 24.0;
  gyro_vert.stab_loop_pi.u_min = -24.0;
  gyro_vert.k_dc_k = 0.607e3;
  gyro_vert.c_m_k = 3.83e-5;
  gyro_vert.u_cor_max = 36.0;
  gyro_vert.gd_pi.k_aw = 1.0 / gyro_vert.gd_pi.k_p;
  gyro_vert.gd_pi.inv_T_i = 1.0 / 0.3;
  gyro_vert.gd_pi.k_p = 9.73;
  gyro_vert.gd_pi.u_max = 36.0;
  gyro_vert.gd_pi.u_min = -36.0;
  gyro_vert.gd_feedback_filter.a0 = 1.0 / (4e-3 * 4e-3);
  gyro_vert.gd_feedback_filter.a1 = 2.0 / 4e-3 * 0.707;
  gyro_vert.gd_feedback_filter.b0 = gyro_vert.gd_feedback_filter.a0;
  gyro_vert.gd_motor.c_e = 6e-3;
  gyro_vert.gd_motor.c_m = 6e-3;
  gyro_vert.gd_motor.l = 25e-3;
  gyro_vert.gd_motor.r = 62.0;
  gyro_vert.Mf_gd_max = 3.39e-4;
  gyro_vert.phi = 55.75 / 57.3;
  gyro_vert.J1 = 1.1e-3;
  gyro_vert.J2 = 2.54e-4;
  gyro_vert.Jpu = 3.0e-5;
  gyro_vert.H = 0.2;
  gyro_vert.Mfx_max = 200e-4;
  gyro_vert.Mfz_max = 5.0e-5;
  gyro_vert.ml = 9e-4;
  gyro_vert.b_m = 2.0e-3;
  gyro_vert.Jrgd = 1.5e-5;
  gyro_vert.Mfm_max = 2.6e-6;
  gyro_vert.Mdx = 20e-4;
  gyro_vert.Mdz = -1.35e-5;
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
      boost::numeric::odeint::make_dense_output<error_stepper_type>(1e-5, 1e-5),
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
