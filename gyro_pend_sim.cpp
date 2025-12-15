#include <algorithm>
#include <array>
#include <boost/numeric/odeint.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
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

struct GyroMotor {
  static constexpr std::size_t ss_dim =
      IIR2<0>::dim + PI<0>::dim + DC_MOTOR<0>::dim + 1;
  using state_vector_t = std::array<double, ss_dim>;
  std::array<std::string, ss_dim> state_names;
  std::size_t state_addr_init;
  // Dynamic subsystems
  IIR2<ss_dim> feedback_flt;
  PI<ss_dim> pi_reg;
  DC_MOTOR<ss_dim> motor;
  // Addresses of the state variables
  const std::size_t a_omega;
  // Parameters
  double J{0.0};
  double Mf_max{0.0};
  double w_req{0.0};
  GyroMotor()
      : state_addr_init(0), feedback_flt(state_addr_init),
        pi_reg(state_addr_init), motor(state_addr_init),
        a_omega([](std::size_t &ss_p) { return ss_p++; }(state_addr_init)) {
    feedback_flt.write_state_names(state_names, "flt");
    pi_reg.write_state_names(state_names, "pi");
    motor.write_state_names(state_names, "motor");
    state_names[a_omega] = "omega";
  }

public:
  void operator()(const GyroMotor::state_vector_t &x,
                  GyroMotor::state_vector_t &dx, const double t) {
    feedback_flt.in = x[a_omega];
    feedback_flt(x, dx, t);
    pi_reg.in = w_req - feedback_flt.out;
    pi_reg.u_add = feedback_flt.out * motor.c_e;
    pi_reg(x, dx, t);
    motor.in_v = pi_reg.out;
    motor.omega = x[a_omega];
    motor(x, dx, t);
    const double sum_trq = motor.out_trq;
    double Mf = 0.0;
    if (std::abs(x[a_omega]) < 1e-10)
      Mf =
          std::abs(sum_trq) > Mf_max ? std::copysign(Mf_max, sum_trq) : sum_trq;
    else
      Mf = 0.9 * std::copysign(Mf_max, x[a_omega]);
    dx[a_omega] = (sum_trq - Mf) / J;
  };
};

typedef boost::numeric::odeint::runge_kutta_dopri5<GyroMotor::state_vector_t>
    error_stepper_type;
int main(int argc, char *argv[]) {
  std::vector<std::string> args(argv, argv + argc);
  std::string file_name("sim_gp.csv");
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
  // Initialization
  GyroMotor gyro_motor;
  gyro_motor.pi_reg.k_p = 9.73;
  gyro_motor.pi_reg.inv_T_i = 1.0 / 0.3;
  gyro_motor.pi_reg.k_aw = 1.0 / gyro_motor.pi_reg.k_p;
  gyro_motor.pi_reg.u_max = 30.0;
  gyro_motor.pi_reg.u_min = -30.0;
  gyro_motor.motor.l = 25e-3;
  gyro_motor.motor.r = 62.0;
  gyro_motor.motor.c_e = 6e-3;
  gyro_motor.motor.c_m = 6e-3;
  gyro_motor.Mf_max = 3.39e-4;
  gyro_motor.w_req = 1e3;
  gyro_motor.J = 1.5e-5;
  gyro_motor.feedback_flt.a0 = 1.0 / (4e-3 * 4e-3);
  gyro_motor.feedback_flt.a1 = 2.0 / 4e-3 * 0.707;
  gyro_motor.feedback_flt.b0 = gyro_motor.feedback_flt.a0;
  // Output file
  std::ofstream log_file(file_name);
  log_file << "time";
  for (const auto &name : gyro_motor.state_names)
    log_file << "," << name;
  log_file << "\n";
  // Initial conditions
  GyroMotor::state_vector_t x0 = {};
  // Integration
  size_t steps = boost::numeric::odeint::integrate_const(
      boost::numeric::odeint::make_dense_output<error_stepper_type>(1e-6, 1e-6),
      gyro_motor, x0, 0.0, 10.0, 0.001,
      [&log_file](const GyroMotor::state_vector_t &x, double t) {
        log_file << t;
        for (auto &&s : x)
          log_file << ',' << s;
        log_file << '\n';
      });
  std::cout << "Number of steps: " << steps << std::endl;
  log_file.close();
  return 0;
}
