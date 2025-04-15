#include <pybind11/pybind11.h>
#include <vector>
#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <cmath>

struct Pose
{
    double x, y, theta;

    Pose(double x, double y, double theta)
        : x(x), y(y), theta(theta)
    {
    }

    Eigen::Vector3d getVector()
    {
        Eigen::Vector3d pose;
        pose << x, y, theta;
        return pose;
    }
};

struct ControlCommand
{
    double v, omega;

    ControlCommand(double v, double omega)
        : v(v), omega(omega)
    {
    }

    Eigen::Vector2d getVector()
    {
        Eigen::Vector2d control_command;
        control_command << v, omega;
        return control_command;
    }
};

struct Noise
{
    double translational_noise, rotational_noise;

    Noise(double translational_noise, double rotational_noise)
        : translational_noise(translational_noise), rotational_noise(rotational_noise)
    {
    }
};

struct VelocityMotionModel
{
    const Noise velocity_noise, angular_noise, rotational_noise;
    const double del_time = 0.1;
    std::random_device rd;
    std::mt19937 gen;

    VelocityMotionModel(const Noise &v_noise, const Noise &a_noise, const Noise &r_noise, const double &del_time)
        : velocity_noise(v_noise), angular_noise(a_noise), rotational_noise(r_noise), del_time(del_time)
    {
        gen.seed(rd());
    }

    double calc_pdf(double a, double b)
    {
        if (b == 0)
            return 0.0;

        double variance = b * b;
        double normalization = 1.0 / sqrt(2.0 * EIGEN_PI * variance);
        double exponent = -(a * a) / (2.0 * variance);

        // std::cout << "Calc " << normalization << " " << exponent << " " << a << " " << b << std::endl;

        return normalization * exp(exponent);
    }

    double get_posterior_probability(const Pose &prev_pose, const Pose &curr_pose, const ControlCommand &control_command)
    {
        double mu = 0.5 * (((prev_pose.x - curr_pose.x) * cos(prev_pose.theta) + (prev_pose.y - curr_pose.y) * sin(prev_pose.theta)) / ((prev_pose.y - curr_pose.y) * cos(prev_pose.theta) - (prev_pose.x - curr_pose.x) * sin(prev_pose.theta)));
        double x_center = 0.5 * (prev_pose.x + curr_pose.x) + mu * (prev_pose.y - curr_pose.y);
        double y_center = 0.5 * (prev_pose.y + curr_pose.y) + mu * (curr_pose.x - prev_pose.x);
        double radius = sqrt(pow(prev_pose.x - x_center, 2) + pow(prev_pose.y - y_center, 2));
        double del_theta = atan2(curr_pose.y - y_center, curr_pose.x - x_center) - atan2(prev_pose.y - y_center, prev_pose.x - x_center);

        double v_hat = (del_theta / del_time) * radius;
        double omega_hat = del_theta / del_time;
        double gamma_hat = ((curr_pose.theta - prev_pose.theta) / del_time) - omega_hat;

        // std::cout << "Posterior : " << del_theta << " " << v_hat << " " << omega_hat << " " << gamma_hat << std::endl;

        std::normal_distribution velocity_distribution(control_command.v - v_hat, velocity_noise.translational_noise * pow(control_command.v, 2) + velocity_noise.rotational_noise * pow(control_command.omega, 2));
        std::normal_distribution angular_distribution(control_command.omega - omega_hat, angular_noise.translational_noise * pow(control_command.v, 2) + angular_noise.rotational_noise * pow(control_command.omega, 2));
        std::normal_distribution gamma_distribution(gamma_hat, rotational_noise.translational_noise * pow(control_command.v, 2) + rotational_noise.rotational_noise * pow(control_command.omega, 2));

        double velocity_distribution_val = calc_pdf(control_command.v - v_hat, velocity_noise.translational_noise * pow(control_command.v, 2) + velocity_noise.rotational_noise * pow(control_command.omega, 2));
        double angular_distribution_val = calc_pdf(control_command.omega - omega_hat, angular_noise.translational_noise * pow(control_command.v, 2) + angular_noise.rotational_noise * pow(control_command.omega, 2));
        double gamma_distribution_val = calc_pdf(gamma_hat, rotational_noise.translational_noise * pow(control_command.v, 2) + rotational_noise.rotational_noise * pow(control_command.omega, 2));

        // std::cout << velocity_distribution_val << " " << angular_distribution_val << " " << gamma_distribution_val << std::endl;

        return (velocity_distribution_val * angular_distribution_val * gamma_distribution_val);
    }

    Pose sample_motion(const Pose &prev_pose, const ControlCommand &control_command)
    {
        std::normal_distribution velocity_distribution(0.0, velocity_noise.translational_noise * pow(control_command.v, 2) + velocity_noise.rotational_noise * pow(control_command.omega, 2));
        std::normal_distribution angular_distribution(0.0, angular_noise.translational_noise * pow(control_command.v, 2) + angular_noise.rotational_noise * pow(control_command.omega, 2));
        std::normal_distribution gamma_distribution(0.0, rotational_noise.translational_noise * pow(control_command.v, 2) + rotational_noise.rotational_noise * pow(control_command.omega, 2));

        double v_hat = control_command.v + velocity_distribution(gen);
        double omega_hat = control_command.omega + angular_distribution(gen);
        double gamma_hat = gamma_distribution(gen);

        // std::cout << v_hat << " " << velocity_distribution.stddev() << " " << velocity_noise.translational_noise * pow(control_command.v, 2) + velocity_noise.rotational_noise * pow(control_command.omega, 2) << std::endl;

        double x_predicted = prev_pose.x - ((v_hat / omega_hat) * sin(prev_pose.theta)) + ((v_hat / omega_hat) * sin(prev_pose.theta + (omega_hat * del_time)));
        double y_predicted = prev_pose.y + ((v_hat / omega_hat) * cos(prev_pose.theta)) - ((v_hat / omega_hat) * cos(prev_pose.theta + (omega_hat * del_time)));
        double theta_predicted = prev_pose.theta + (omega_hat * del_time) + (gamma_hat * del_time);

        return Pose(x_predicted, y_predicted, theta_predicted);
    }
};

PYBIND11_MODULE(velocity_motion_model, m)
{
    m.doc() = "Velocity Motion Model";

    pybind11::class_<Pose>(m, "Pose")
        .def(pybind11::init<double, double, double>())
        .def_readwrite("x", &Pose::x)
        .def_readwrite("y", &Pose::y)
        .def_readwrite("theta", &Pose::theta);

    pybind11::class_<ControlCommand>(m, "ControlCommand")
        .def(pybind11::init<double, double>())
        .def_readwrite("v", &ControlCommand::v)
        .def_readwrite("omega", &ControlCommand::omega);

    pybind11::class_<Noise>(m, "Noise")
        .def(pybind11::init<double, double>())
        .def_readwrite("translational_noise", &Noise::translational_noise)
        .def_readwrite("rotational_noise", &Noise::rotational_noise);

    pybind11::class_<VelocityMotionModel>(m, "VelocityMotionModel")
        .def(pybind11::init<const Noise &, const Noise &, const Noise &, const double &>())
        .def("get_posterior_probability", &VelocityMotionModel::get_posterior_probability)
        .def("sample_motion", &VelocityMotionModel::sample_motion);
}