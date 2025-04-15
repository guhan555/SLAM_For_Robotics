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
    double rot_1, trans, rot_2;

    ControlCommand(double rot_1, double trans, double rot_2)
        : rot_1(rot_1), trans(trans), rot_2(rot_2)
    {
    }

    Eigen::Vector3d getVector()
    {
        Eigen::Vector3d control_command;
        control_command << rot_1, trans, rot_2;
        return control_command;
    }
};

struct Noise
{
    double translational_noise, rotational_noise;

    Noise(double translational_noise, double rot_noise)
        : translational_noise(translational_noise), rotational_noise(rot_noise)
    {
    }
};

struct OdometryMotionModel
{
    const Noise trans_noise, rot_noise, odometry_noise;
    const double del_time = 0.1;
    std::random_device rd;
    std::mt19937 gen;

    OdometryMotionModel(const Noise &v_noise, const Noise &a_noise, const Noise &o_noise, const double &del_time)
        : trans_noise(v_noise), rot_noise(a_noise), odometry_noise(o_noise), del_time(del_time)
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

    Pose get_odometry_pose(const Pose &pose)
    {
        return pose;
        // TODO: Add noise
    }

    Pose get_odometry_pose(const Pose &prev_pose, const ControlCommand &control_command)
    {
        double odometry_theta = prev_pose.theta + control_command.rot_1;
        double odometry_x = prev_pose.x + control_command.trans * cos(odometry_theta);
        double odometry_y = prev_pose.y + control_command.trans * sin(odometry_theta);

        odometry_theta += control_command.rot_2;

        return Pose(odometry_x, odometry_y, odometry_theta);

        // TODO: Add noise
    }

    double get_posterior_probability(const Pose &prev_pose, const Pose &curr_pose, const ControlCommand &control_command)
    {
        Pose odometry_prev_pose = get_odometry_pose(prev_pose);
        Pose odometry_curr_pose = get_odometry_pose(prev_pose, control_command);

        double odometry_rot_1 = atan2(odometry_curr_pose.y - odometry_prev_pose.y, odometry_curr_pose.x - odometry_prev_pose.x) - odometry_curr_pose.theta;
        double odometry_trans = sqrt(pow(odometry_curr_pose.x - odometry_prev_pose.x, 2) + pow(odometry_curr_pose.y - odometry_prev_pose.y, 2));
        double odometry_rot_2 = odometry_curr_pose.theta - odometry_prev_pose.theta - odometry_rot_1;

        double rot_1_hat = atan2(curr_pose.y - prev_pose.y, curr_pose.x - prev_pose.x) - curr_pose.theta;
        double trans_hat = sqrt(pow(curr_pose.x - prev_pose.x, 2) + pow(curr_pose.y - prev_pose.y, 2));
        double rot_2_hat = curr_pose.theta - prev_pose.theta - rot_1_hat;

        double p1 = calc_pdf(odometry_rot_1 - rot_1_hat, trans_noise.translational_noise * trans_hat + rot_noise.rotational_noise * trans_hat);
        double p2 = calc_pdf(odometry_trans - trans_hat, rot_noise.translational_noise * trans_hat + rot_noise.rotational_noise * (rot_1_hat + rot_2_hat));
        double p3 = calc_pdf(odometry_rot_2 - rot_2_hat, trans_noise.translational_noise * trans_hat + rot_noise.rotational_noise * rot_2_hat);

        // std::cout << velocity_distribution_val << " " << angular_distribution_val << " " << gamma_distribution_val << std::endl;

        return (p1 * p2 * p3);
    }

    Pose sample_motion(const Pose &prev_pose, const ControlCommand &control_command)
    {
        std::normal_distribution rot_1_distribution(0.0, rot_noise.translational_noise * control_command.trans + rot_noise.rotational_noise * abs(control_command.rot_1));
        std::normal_distribution trans_distribution(0.0, trans_noise.translational_noise * control_command.trans + trans_noise.rotational_noise * (abs(control_command.rot_1) + abs(control_command.rot_2)));
        std::normal_distribution rot_2_distribution(0.0, rot_noise.translational_noise * control_command.trans + rot_noise.rotational_noise * abs(control_command.rot_2));

        double rot_1_hat = control_command.rot_1 + rot_1_distribution(gen);
        double trans_hat = control_command.trans + trans_distribution(gen);
        double rot_2_hat = control_command.rot_2 + rot_2_distribution(gen);

        // std::cout << v_hat << " " << velocity_distribution.stddev() << " " << trans_noise.translational_noise * pow(control_command.v, 2) + trans_noise.rot_2_noise * pow(control_command.omega, 2) << std::endl;

        double x_predicted = prev_pose.x + trans_hat * cos(prev_pose.theta + rot_1_hat);
        double y_predicted = prev_pose.y + trans_hat * sin(prev_pose.theta + rot_1_hat);
        double theta_predicted = prev_pose.theta + rot_1_hat + rot_2_hat;

        return Pose(x_predicted, y_predicted, theta_predicted);
    }
};

PYBIND11_MODULE(odometry_motion_model, m)
{
    m.doc() = "Odometry Motion Model";

    pybind11::class_<Pose>(m, "Pose")
        .def(pybind11::init<double, double, double>())
        .def_readwrite("x", &Pose::x)
        .def_readwrite("y", &Pose::y)
        .def_readwrite("theta", &Pose::theta);

    pybind11::class_<ControlCommand>(m, "ControlCommand")
        .def(pybind11::init<double, double, double>())
        .def_readwrite("rot_1", &ControlCommand::rot_1)
        .def_readwrite("trans", &ControlCommand::trans)
        .def_readwrite("rot_2", &ControlCommand::rot_2);

    pybind11::class_<Noise>(m, "Noise")
        .def(pybind11::init<double, double>())
        .def_readwrite("translational_noise", &Noise::translational_noise)
        .def_readwrite("rotational_noise", &Noise::rotational_noise);

    pybind11::class_<OdometryMotionModel>(m, "OdometryMotionModel")
        .def(pybind11::init<const Noise &, const Noise &, const Noise &, const double &>())
        .def("get_posterior_probability", &OdometryMotionModel::get_posterior_probability)
        .def("sample_motion", &OdometryMotionModel::sample_motion);
}