#include <iostream>
#include <cmath>
#include <matplot/matplot.h>
#include <Eigen/Dense>

struct Point
{
    double x;
    double y;
    double z;
};

struct Line
{
    Point p1;
    Point p2;
};

struct Square
{
    Point p1;
    Point p2;
    Point p3;
    Point p4;
};

struct transforms
{
    double x, y, z;
    double rot_x, rot_y, rot_z;
    double scale;
    double rot_scale;
};

class HomogeneousTransform
{
private:
    Eigen::MatrixXd rotate_x(double rad)
    {
        Eigen::MatrixXd R_x(3, 3);
        R_x << 1, 0, 0,
            0, cos(rad), -sin(rad),
            0, sin(rad), cos(rad);
        return R_x;
    }

    Eigen::MatrixXd rotate_y(double rad)
    {
        Eigen::MatrixXd R_y(3, 3);
        R_y << cos(rad), 0, sin(rad),
            0, 1, 0,
            -sin(rad), 0, cos(rad);
        return R_y;
    }

    Eigen::MatrixXd rotate_z(double rad)
    {
        Eigen::MatrixXd R_z(3, 3);
        R_z << cos(rad), -sin(rad), 0,
            sin(rad), cos(rad), 0,
            0, 0, 1;
        return R_z;
    }

public:
    Eigen::Vector4d transform_point(Point pt, transforms t)
    {
        Eigen::Vector4d pt_transformed;
        Eigen::MatrixXd rotation_matrix = rotate_x(t.rot_x) * rotate_y(t.rot_y) * rotate_z(t.rot_z);
        rotation_matrix << t.rot_scale * rotation_matrix;

        Eigen::MatrixXd transform_matrix(4, 4);
        transform_matrix.topLeftCorner(3, 3) = rotation_matrix;
        transform_matrix.rightCols(1) << t.x, t.y, t.z, 1;
        transform_matrix.bottomLeftCorner(1, 3) << 0, 0, 0;

        pt_transformed = t.scale * transform_matrix * Eigen::Vector4d(pt.x, pt.y, pt.z, 1);

        std::cout << pt_transformed[0] << " " << pt_transformed[1] << " " << pt_transformed[2] << std::endl;

        return pt_transformed;
    }

    std::vector<Eigen::Vector4d> transform_line(Line line, transforms t)
    {
        Eigen::Vector4d pt_transformed_1, pt_transformed_2;
        std::vector<Eigen::Vector4d> line_transformed;

        pt_transformed_1 = transform_point(line.p1, t);
        pt_transformed_2 = transform_point(line.p2, t);

        line_transformed.push_back(pt_transformed_1);
        line_transformed.push_back(pt_transformed_2);

        return line_transformed;
    }

    std::vector<Eigen::Vector4d> transform_square(Square square, transforms t)
    {
        Eigen::Vector4d pt_transformed_1, pt_transformed_2, pt_transformed_3, pt_transformed_4;
        std::vector<Eigen::Vector4d> square_transformed;

        pt_transformed_1 = transform_point(square.p1, t);
        pt_transformed_2 = transform_point(square.p2, t);
        pt_transformed_3 = transform_point(square.p3, t);
        pt_transformed_4 = transform_point(square.p4, t);

        square_transformed.push_back(pt_transformed_1);
        square_transformed.push_back(pt_transformed_2);
        square_transformed.push_back(pt_transformed_3);
        square_transformed.push_back(pt_transformed_4);

        return square_transformed;
    }
};

void show_transformed_point(Point point, Eigen::Vector4d transformed_point)
{
    std::vector<double> x = {point.x};
    std::vector<double> y = {point.y};
    matplot::scatter(x, y);
    matplot::hold(true);
    x = {transformed_point[0]};
    y = {transformed_point[1]};
    matplot::scatter(x, y);
    matplot::show();
}

void show_transformed_line(Line line, std::vector<Eigen::Vector4d> transformed_point)
{
    std::vector<double> x = {line.p1.x, line.p2.x};
    std::vector<double> y = {line.p1.y, line.p2.y};
    std::vector<double> z = {line.p1.z, line.p2.z};

    matplot::plot3(x, y, z);
    matplot::hold(true);

    x.clear();
    y.clear();
    z.clear();

    for (unsigned int i = 0; i < transformed_point.size(); i++)
    {
        x.push_back(transformed_point[i][0]);
        y.push_back(transformed_point[i][1]);
        z.push_back(transformed_point[i][2]);
    }
    matplot::plot3(x, y, z);
    matplot::xlabel("X-Axis");
    matplot::ylabel("Y-Axis");
    matplot::zlabel("Z-Axis");
    matplot::show();
}

void show_transformed_square(Square square, std::vector<Eigen::Vector4d> transformed_square)
{
    std::vector<double> x = {square.p1.x, square.p2.x, square.p3.x, square.p4.x};
    std::vector<double> y = {square.p1.y, square.p2.y, square.p3.y, square.p4.y};
    std::vector<double> z = {square.p1.z, square.p2.z, square.p3.z, square.p4.z};

    matplot::plot3(x, y, z);
    matplot::hold(true);

    x.clear();
    y.clear();
    z.clear();

    for (unsigned int i = 0; i < transformed_square.size(); i++)
    {
        x.push_back(transformed_square[i][0]);
        y.push_back(transformed_square[i][1]);
        z.push_back(transformed_square[i][2]);
    }
    matplot::plot3(x, y, z);
    matplot::xlabel("X-Axis");
    matplot::ylabel("Y-Axis");
    matplot::zlabel("Z-Axis");
    matplot::show();
}

int main()
{
    struct Point point;
    point.x = 1;
    point.y = 1;
    point.z = 0;

    struct Line line;
    line.p1.x = 1;
    line.p1.y = 1;
    line.p1.z = 0;
    line.p2.x = 2;
    line.p2.y = 2;
    line.p2.z = 0;

    struct Square square;
    square.p1.x = 1;
    square.p1.y = 1;
    square.p1.z = 0;
    square.p2.x = 2;
    square.p2.y = 1;
    square.p2.z = 0;
    square.p3.x = 2;
    square.p3.y = 2;
    square.p3.z = 0;
    square.p4.x = 1;
    square.p4.y = 2;
    square.p4.z = 0;

    struct transforms transform;
    transform.x = 0;
    transform.y = 0;
    transform.z = 0;
    transform.rot_x = 0;
    transform.rot_y = 0;
    transform.rot_z = 0.785398;
    transform.scale = 1;
    transform.rot_scale = 1;

    HomogeneousTransform ht;

    // show_transformed_point(point, ht.transform_point(point, transform));
    // show_transformed_line(line, ht.transform_line(line, transform));
    show_transformed_square(square, ht.transform_square(square, transform));

    return 0;
}