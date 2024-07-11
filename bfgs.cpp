// Libraries
//    - Ibex : Interval Arithmetic
//        - Gaol : Interval estimation from a function
//    - Eigen : Cholesky Decomposition
// References
//    - [0] C.A. Floudas! * C.S. Adjiman!, S. Dallwig” and A. Neumaier”. A global optimization method,
//      abb, for general twice-differentiable constrained nlps — i. theoretical advances. pages 1137–
//      1158. Department of Chemical Engineering, Princeton University, Princeton, NJ 08544, U.S.A
//      and Institut fur Mathematik, Universitat Wien, Strudlhofgasse 4, A-1090 Wien, Austria, 1997.
//    - [1] Jorge Nocedal and Stephen J. Wright. Numerical Optimization. Second Edition. Springer
//      Science+Business Media, LLC, 233 Spring Street, New York, NY 10013, USA, 2006
// Algorithms Implemented
//    - Diagonal Shift
//    - Gerschgorin at a point
//    - Optimization with Hessian modfication using Interval approximation(Alpha - O(n^2))
//    - Optimization with Hessian modfication using Interval approximation(Alpha - O(n^3))

// import libraries
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include "ibex.h"

// Note: Changed g++ flag to -std=c++17 during compilation to support eigen
#include "eigen/Eigen/Dense"
#include "eigen/Eigen/QR"

// bool isValid = false;

using namespace std;
using namespace std::chrono;
using namespace ibex;

// Converts Ibex Matrix to std c++ 2d vector
vector<vector<double>> ConvertIbexMatrixTo2DVector(ibex::Matrix m)
{
    int n = m.nb_rows();
    vector<vector<double>> dummyHessian(n, vector<double>(n));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            dummyHessian[i][j] = m[i][j];
        }
    }
    return dummyHessian;
}

// Converts std c++ 2d vector to Eigen Matrix
Eigen::MatrixXd ConvertToEigenMatrix(std::vector<std::vector<double>> data)
{
    Eigen::MatrixXd eMatrix(data.size(), data[0].size());
    for (int i = 0; i < data.size(); ++i)
        eMatrix.row(i) = Eigen::VectorXd::Map(&data[i][0], data[0].size());
    return eMatrix;
}

// Calculates distance between old_vec and new_vec
double normOfVector(Vector old_vec, Vector new_vec, int n, int iter)
{
    if (iter == 1)
        return 1;
    double gr = 0;
    for (int i = 0; i < n; i++)
    {
        gr += ((new_vec[i] - old_vec[i]) * (new_vec[i] - old_vec[i]));
    }
    return sqrt(gr);
}

// Returs Direction Vector from hessian and gradient (-1*Hessian*grad)
Vector DirectionVector(Eigen::MatrixXd hessian, Vector grad)
{
    int n = grad.size();

    Vector direcV(Vector ::zeros(n));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            direcV[i] += (-1 * hessian(i, j)) * grad[j];
        }
    }
    return direcV;
}

// Returns Gradient vector from gradient function(gradient) and point(xk)
Vector gradVector(Function gradient, Vector xk)
{
    int n = xk.size();

    Vector gradV(n);
    for (int i = 0; i < n; i++)
    {
        IntervalVector result = gradient[i].eval(xk);
        gradV[i] = result.lb()[0];
    }
    return gradV;
}

// Returns Hessian Matrix from double differentiation function(dff) and point(xk)
Matrix HessianMatrix(Function dff, Vector xk)
{
    int n = xk.size();
    Matrix hessian(n, n);
    for (int i = 0; i < n; i++)
    {
        // Function new_f(gradient[i], Function::DIFF);
        for (int j = i; j < n; j++)
        {
            IntervalVector result = dff[i][j].eval(xk);
            hessian[i][j] = hessian[j][i] = result.lb()[0];
        }
    }
    return hessian;
}

// Compute Lower bound on minimim eigen value of Interval Matrix(Im)
double minEigenValueIntervalMatrix(IntervalMatrix &Im)
{
    int sz = Im.nb_rows();
    double min_eigen = POS_INFINITY;
    for (int i = 0; i < sz; i++)
    {
        int sum = 0;
        for (int j = 0; j < sz; j++)
        {
            if (j != i)
            {
                sum += max(abs(Im[i][j].lb()), abs(Im[i][j].ub()));
            }
        }
        min_eigen = min(min_eigen, Im[i][i].lb() - sum);
    }
    return min_eigen;
}

// Inverse of Hessian Matrix
Eigen::MatrixXd InverseMatrix(Matrix hessian)
{
    int n = hessian.nb_rows();
    Eigen::MatrixXd hessianInverse(n, n);

    for (int i = 0; i < n; i++)
    {
        for (int j = i; j < n; j++)
        {
            hessianInverse(i, j) = hessianInverse(j, i) = hessian[i][j];
        }
    }
    // cout << "Inverse: " << hessianInverse << "\n";
    hessianInverse = hessianInverse.inverse();
    // cout << "Inverse: " << hessianInverse << "\n";
    return hessianInverse;
}

// [0] Pg 1145
Matrix CalculateModifiedMidPointMatrix(IntervalMatrix &Im)
{
    int sz = Im.nb_rows();
    Matrix ModifiedMatrix(sz, sz);
    for (int i = 0; i < sz; i++)
    {
        for (int j = 0; j < sz; j++)
        {
            if (i == j)
            {
                ModifiedMatrix[i][j] = Im[i][j].lb();
            }
            else
            {
                ModifiedMatrix[i][j] = (Im[i][j].lb() + Im[i][j].ub()) / 2;
            }
        }
    }
    return ModifiedMatrix;
}

// [0] Pg 1145
Matrix CalculateEMatrix(IntervalMatrix &Im)
{
    int sz = Im.nb_rows();
    Matrix ModifiedMatrix(sz, sz);
    for (int i = 0; i < sz; i++)
    {
        for (int j = 0; j < sz; j++)
        {
            if (i == j)
            {
                ModifiedMatrix[i][j] = (Im[i][j].ub() - Im[i][j].lb()) / 2;
            }
            else
            {
                ModifiedMatrix[i][j] = 0;
            }
        }
    }
    return ModifiedMatrix;
}

// [0] Pg 1145
Matrix CalculateModifiedRadiusMatrix(IntervalMatrix &Im)
{
    int sz = Im.nb_rows();
    Matrix ModifiedMatrix(sz, sz);
    for (int i = 0; i < sz; i++)
    {
        for (int j = 0; j < sz; j++)
        {
            if (i == j)
            {
                ModifiedMatrix[i][j] = 0;
            }
            else
            {
                ModifiedMatrix[i][j] = (Im[i][j].ub() - Im[i][j].lb()) / 2;
            }
        }
    }
    return ModifiedMatrix;
}

// Compute Lower bound on minimim eigen value of Matrix(mt) - https://www.scirp.org/journal/paperinformation.aspx?paperid=103295
double calcLowerBoundEigenValue(Matrix mt)
{
    int sz = mt.nb_rows();
    double min_eigen = POS_INFINITY;
    for (int i = 0; i < sz; i++)
    {
        int sum = 0;
        for (int j = 0; j < sz; j++)
        {
            if (j != i)
            {
                sum += abs(mt[i][j]);
            }
        }
        min_eigen = min(min_eigen, mt[i][i] - sum);
    }
    return min_eigen;
}

// Compute Upper bound on max eigen value of Matrix(mt) - https://www.scirp.org/journal/paperinformation.aspx?paperid=103295
double calcUpperBoundEigenValue(Matrix mt)
{
    int sz = mt.nb_rows();
    double max_eigen = POS_INFINITY;
    for (int i = 0; i < sz; i++)
    {
        int sum = 0;
        for (int j = 0; j < sz; j++)
        {
            if (j != i)
            {
                sum += abs(mt[i][j]);
            }
        }
        max_eigen = max(max_eigen, mt[i][i] + sum);
    }
    return max_eigen;
}
double dotProduct( Vector a, Vector b) {
    double dot = 0.0;
    for (int i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}

Matrix outerProduct(Vector a, Vector b) {
    int n = a.size();
    Matrix result(n,n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i][j] = a[i] * b[j];
        }
    }
    return result;
}
Matrix FiniteDifference(Function function, Vector xk) {

    int n = xk.size();
    Matrix hessian(n, n);
    double h = 1e-5;  

    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            Vector x0 = xk;
            Vector x1 = xk;
            Vector x2 = xk;
            Vector x3 = xk;

            x1[i] += h;
            x1[j] += h;
            x2[i] += h;
            x3[j] += h;

            IntervalVector result1 = function.eval(x0);
            IntervalVector result2 = function.eval(x1);
            IntervalVector result3 = function.eval(x2);
            IntervalVector result4 = function.eval(x3);

            double f_ij = (result2.lb()[0] - result3.lb()[0] - result4.lb()[0] + result1.lb()[0]) / (h * h);
            hessian[j][i] = hessian[i][j] = f_ij;
        }
    }
    return hessian;
}

// https://en.wikipedia.org/wiki/Spectral_radius#:~:text=In%20mathematics%2C%20the%20spectral%20radius,denoted%20by%20%CF%81(%C2%B7).
double spectralRadius(Matrix mt)
{
    double lower_bound = calcLowerBoundEigenValue(mt);
    double upper_bound = calcUpperBoundEigenValue(mt);

    return max(abs(lower_bound), abs(upper_bound));
}

// [0] Pg 1145
double On3MinEigenValueIntervalMatrix(IntervalMatrix &Im)
{
    Matrix ModifiedMPMatrix = CalculateModifiedMidPointMatrix(Im);
    Matrix EMatrix = CalculateEMatrix(Im);
    Matrix ModifiedRadiusMatrix = CalculateModifiedRadiusMatrix(Im);
    double sp = spectralRadius(ModifiedRadiusMatrix + EMatrix);
    double lb = calcLowerBoundEigenValue(ModifiedMPMatrix + EMatrix);

    return lb - sp;
}

//
bool checkCholesky(Matrix hessian)
{
    vector<vector<double>> dummyHessian = ConvertIbexMatrixTo2DVector(hessian);
    Eigen::LLT<Eigen::MatrixXd> lltOfA(ConvertToEigenMatrix(dummyHessian)); // compute the Cholesky decomposition of Hessian
    Eigen::ComputationInfo m_info = lltOfA.info();
    if (m_info == 1)
    {
        return true;
        // return 0;
    }
    else
    {
        return false;
    }
}

Matrix ModifyHessianDS(Matrix hessian)
{
    int n = hessian.nb_rows();
    Matrix IDMatrix = Matrix::eye(n);
    int cnt = 0;
    while (checkCholesky(hessian) && cnt < 100000)
    {
        cnt++;
        hessian = hessian + IDMatrix;
    }
    return hessian;
}

Matrix ModifyHessianGerschgorin(Matrix hessian)
{
    double min_eigen = POS_INFINITY;
    int n = hessian.nb_rows();
    for (int i = 0; i < n; i++)
    {
        double sum = 0;
        for (int j = 0; j < n; j++)
        {
            if (j != i || j == 0)
            {
                sum += abs(hessian[i][j]);
            }
        }
        min_eigen = min(min_eigen, hessian[i][i] - sum);
    }

    if (min_eigen < 0)
    {
        hessian = hessian + (-1 * min_eigen) * (Matrix::eye(n));
    }

    return hessian;
}

Matrix FindTestVectors(int p, int n)
{
    return ibex::Matrix::rand(p, n);
}

Matrix ModifyHessianOn3(string func, string fnc_param, Vector xk, double alp, Matrix hessian, Function dff, double range_ll, double range_ul)
{
    int m = xk.size();
    // Matrix hessian(m, m);
    // const char *func_string = func.c_str();
    // Function f(fnc_param.c_str(), func_string);
    // Function df(f, Function::DIFF);

    // Interval Calculation [xk-alp,xk+alp]
    double _x[m][2];
    for (int i = 0; i < m; i++)
    {
        _x[i][0] = max(range_ll, xk[i] - alp * (range_ul - range_ll));
        _x[i][1] = min(range_ul, xk[i] + alp * (range_ul - range_ll));
    }

    IntervalVector xy(m, _x); // build xy=([1,2],[3,4])
    IntervalMatrix im(m, m);
    for (int i = 0; i < m; i++)
    {
        // Function new_f(df[i], Function::DIFF);
        for (int j = i; j < m; j++)
        {
            im[i][j] = im[j][i] = dff[i][j].eval(xy);
        }
    }
    double lowerBoundEigenValue = On3MinEigenValueIntervalMatrix(im);
    if (lowerBoundEigenValue < -100000)
    {
        lowerBoundEigenValue = -100000;
    }
    double alpha = max(double(0), -0.5 * lowerBoundEigenValue);

    hessian = hessian + 2 * alpha * (Matrix::eye(xk.size()));
    return hessian;
}

Matrix ModifyHessianOn2(string func, string fnc_param, Vector xk, double alp, Matrix hessian, Function dff, double range_ll, double range_ul)
{
    int m = xk.size();
    // Matrix hessian(m, m);
    // const char *func_string = func.c_str();
    // Function f(fnc_param.c_str(), func_string);
    // Function df(f, Function::DIFF);

    // Interval Calculation [xk-alp,xk+alp]
    double _x[m][2];
    for (int i = 0; i < m; i++)
    {
        _x[i][0] = xk[i] - alp * (range_ul - range_ll) / 2;
        _x[i][1] = xk[i] + alp * (range_ul - range_ll) / 2;
    }

    IntervalVector xy(m, _x); // build xy=([1,2],[3,4])
    IntervalMatrix im(m, m);
    for (int i = 0; i < m; i++)
    {
        // Function new_f(df[i], Function::DIFF);
        for (int j = i; j < m; j++)
        {
            im[i][j] = im[j][i] = dff[i][j].eval(xy);
        }
    }
    // cout << im << "\n";
    double lowerBoundEigenValue = minEigenValueIntervalMatrix(im);

    // cout << "EigenValue :" << lowerBoundEigenValue << "\n";

    if (lowerBoundEigenValue < -100000)
    {
        lowerBoundEigenValue = -100000;
    }
    double alpha = max(double(0), -0.5 * lowerBoundEigenValue);

    // cout << "alpha :" << alpha << "\n";

    hessian = hessian + 2 * alpha * (Matrix::eye(xk.size()));
    return hessian;
}

Matrix ModifyHessian(Matrix hessian, int algo, string func, string fnc_param, Vector xk, double alp, Function dff, double range_ll, double range_ul)
{
    switch (algo)
    {
    case 0:
        return ModifyHessianGerschgorin(hessian);
    case 1:
        return ModifyHessianDS(hessian);
    case 2:
        return ModifyHessianOn2(func, fnc_param, xk, alp, hessian, dff, range_ll, range_ul);
    case 3:
        return ModifyHessianOn3(func, fnc_param, xk, alp, hessian, dff, range_ll, range_ul);

    default:
        return hessian;
    }
}

// Generates random number between fMin and fMax
double random(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

bool CheckCacheValidity(double range_ll, double range_ul, double alp, Vector xk, Vector xkn)
{
    int m = xk.size();
    double _x[m][2];
    for (int i = 0; i < m; i++)
    {
        _x[i][0] = max(range_ll, xk[i] - alp * (range_ul - range_ll));
        _x[i][1] = min(range_ul, xk[i] + alp * (range_ul - range_ll));
    }
    IntervalVector currentInterval(m, _x);
    for (int i = 0; i < xk.size(); i++)
    {
        if (!currentInterval[i].contains(xkn[i]))
        {
            return false;
        }
    }
    return true;
}

bool numericAwareCompare(const std::string &a, const std::string &b)
{
    auto is_digit = [](char c)
    { return std::isdigit(c) != 0; };
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size())
    {
        if (is_digit(a[i]) && is_digit(b[j]))
        {
            size_t start_i = i, start_j = j;
            while (i < a.size() && is_digit(a[i]))
                i++;
            while (j < b.size() && is_digit(b[j]))
                j++;
            std::string num_a = a.substr(start_i, i - start_i);
            std::string num_b = b.substr(start_j, j - start_j);
            // Strip leading zeros and compare numbers
            num_a.erase(0, num_a.find_first_not_of('0'));
            num_b.erase(0, num_b.find_first_not_of('0'));
            if (num_a.size() != num_b.size())
            {
                return num_a.size() < num_b.size();
            }
            if (num_a != num_b)
            {
                return num_a < num_b;
            }
        }
        else if (a[i] != b[j])
        {
            return a[i] < b[j];
        }
        else
        {
            i++;
            j++;
        }
    }
    return a.size() < b.size();
}

void replaceAll(string &source, const string &from, const string &to)
{
    if (from.empty())
    {
        return;
    }
    size_t start_pos = 0;
    while ((start_pos = source.find(from, start_pos)) != string::npos)
    {
        source.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cout << "Usage: " << argv[0] << " <directory_path>" << endl;
        return 1;
    }

    vector<string> test_probs;
    const filesystem::path dir_path{argv[1]};

    if (!filesystem::exists(dir_path) || !filesystem::is_directory(dir_path))
    {
        cout << "Provided path does not exist or is not a directory." << endl;
        return 1;
    }

    for (auto const &dir_entry : filesystem::directory_iterator{dir_path})
    {
        if (dir_entry.is_regular_file())
        {
            test_probs.push_back(dir_entry.path().filename().string());
        }
    }

    sort(test_probs.begin(), test_probs.end(), numericAwareCompare);
    for (const auto &filename : test_probs)
    {
        cout << "Processing file: " << filename << endl;
        // Add the code here to process each file
    }
    ofstream fout;
    string filename = "iter_BFGS.csv";
    fout.open(filename, std::ofstream::out | std::ofstream::trunc);
    fout << "ProblemName" << "," << "Algorithm" << "," << "Alpha" << "," << "Iterations" << "," << "FuncValue" << "," << "Gradient" << "," << "TimeTaken(s)" << "," << "HessianModified" << "," << "Solved" << "," << "Max_Iterations" << "," << "Time_Limit" << "," << "Descent_Direction" << "\n";
    //fout << "Alpha" << "," << "Function Value" << "," << "Gradient" << "," << "Point" << "\n";

    for (const string &test_problems_file : test_probs)
    {
        fstream test_problems;
        // string test_problems_file = "on3test.txt";
        filesystem::path full_path = dir_path / test_problems_file;
        // open the test_problems_file file to perform read operation using file object.
        test_problems.open(full_path, ios::in);

        if (!test_problems.is_open())
        {
            cout << "Unable to open file: " << full_path << endl;
            continue; // Skip to the next file
        }

        // Alpha option for calculating Interval Hessian
        double alp_options[3] = {0.01, 0.05, 0.1};

        string problem_name;
        while (getline(test_problems, problem_name))
        {
            try
            {
                // Extraxting the problem name
                problem_name = problem_name.substr(0, problem_name.length() - 1);

                // Open new csv file to save data for this problem
                // string filename = problem_name + ".csv";
                // fout.open(filename, std::ofstream::out | std::ofstream::trunc);

                // fout << "IterationGG" << "," << "IterationON2" << "," << "IterationDS" << "," << "IterationON3" << "," << "FuncValueGG" << "," << "FuncValueON2" << "," << "FuncValueDS" << "," << "FuncValueON3" << "," << "Alp" << "\n";

                string n_str, range_str;
                int n;
                double range_ll, range_ul;
                string func;

                // n = Variables in the problem
                getline(test_problems, n_str);
                n = stoi(n_str);
                // Input function of the problem
                getline(test_problems, func);

                // LowerBound(ll) and UpperBound(ul) for testing initial guess over this range
                getline(test_problems, range_str);
                range_ll = stod(range_str);
                getline(test_problems, range_str);
                range_ul = stod(range_str);

                // cout << "Initial Function: " << func << endl;

                // int domain_size = range_ul - range_ll;
                // for (int i = 1; i <= n; ++i)
                // {
                //     string from = "x(" + to_string(i) + ")";
                //     string to = "(" + from + "*" + to_string(domain_size) + " + " + to_string(-1*range_ll) + ")";
                //     replaceAll(func, from, to);
                // }

                // cout << "Modified Function: " << func << endl;

                const char *func_string = func.c_str();
                cout << "Non Convex Function: " << test_problems_file << "\n"
                     << func << endl;

                string fnc_param = "x[" + to_string(n) + "]";

                // Generating ibex Function(f) with func_string of the problem
                Function f(fnc_param.c_str(), func_string);

                // Differentiation(df) of the ibex function f
                Function df(f, Function::DIFF);

                // Double Differentiation(dff) of the ibex function f
                Function dff(df, Function::DIFF);

                Matrix hessian(n, n), v1(n,n), v2(n,n); // hessian matrix
                Matrix identityMatrix = Matrix::eye(n);
                Eigen::MatrixXd updated_inverseHessian(n, n);
                Eigen::MatrixXd temp(n,n);
                Eigen::MatrixXd inverseHessian(n, n);
                Eigen::MatrixXd hessian_cache(n, n);

                Vector xk(n), xkn(n), inixk(n), diff_x(n), diff_grad(n); // xk: prev point, xkn: new point, inixk: initial guess
                Vector fs_our(n), fs_ger(n);

                int num_algorithms = 6; // only using Gershgorin Algo
                string algorithms[num_algorithms] = {"Steepest Descent", "Gerschgorin", "Diagonal Shift", "On2 Interval hessian", "On3 Interval Hessian", "BFGS"};
                double func_value_ger, func_value_on2, func_value_ds, func_value_on3; // Function value of the four problems
                double func_value[num_algorithms];
                int algo_iterations[num_algorithms];
                int total_test = 0; // Total tests(initial guesses) in which hessian was modified atleast one time.

                for (int j = 0; j < n; j++)
                {
                    inixk[j] = random(range_ll, range_ul);
                }
                xk = inixk;
                xkn = xk;

                for (int i = 0; i < n; i++)
                {
                    cout << inixk[i] << " ";
                }
                cout << endl;

                Vector grad(n), direcV(n), updated_grad(n); // Gradient and Direction Vector
                double a_init = 1;
                double a = a_init; // a: alpha to be modified with the wolfe conditions

                // c constant used for check strong wolf condition
                double c = 0.8;

                double a_min = pow(10, -20); // min value for a. Note: If a becomes less than a_min, then we have reached the solution

                // rho value to update alpha value
                double rh = 0.01;
                double rho;

                int iter = 1;                 // iteration count
                int iter_max = 10000;         // max iteration allowed.(If iter goes beyond iter_max, we might have diverted exceptionally from our solution)
                int duration_max = 3600;      // max amount of time allowed for a problem
                int hessian_modify_count = 0; // Number of times the hessian is modified.
                IntervalVector result;        // Value of the function at point xk

                for (int l = 0; l < num_algorithms; l++)
                {
                    func_value[l] = 0;
                    algo_iterations[l] = 0;
                }

                for (int l = 0; l < num_algorithms; l++)
                {
                    bool algo_success = true;
                    double norm_grad = 0;
                    double gfpk = 0;
                    bool small_step = false;
                    std::chrono::duration<double> time_taken;

                    if (l != 5)
                    {
                        continue;
                    }
                    for (int k = 0; k < sizeof(alp_options) / sizeof(alp_options[0]); k++)
                    {
                        norm_grad = 0;
                        algo_success = true; // Algorithms solves the problem successfull
                        auto start = chrono::high_resolution_clock::now();
                        if (!algo_success)
                        {
                            // if any algorithm fails then we will not consider this test
                            break;
                        }
                        if (l <= 5 && algo_iterations[l] != 0)
                        {
                            //  Iterations for Gerschgorin and DS are calculated
                            // Gerschgorin and Diagonal Shift is independent of alpha(interval hessian)
                            break;
                        }
                        cout << l << " " << algorithms[l] << " Alpha: " << alp_options[k] << "\n";
                        xkn = inixk;
                        a = a_init;
                        iter = 1;                 // iteration count
                        hessian_modify_count = 0; // Number of times the hessian is modified.

                        while (true)
                        {
                            if (iter > iter_max)
                            {
                                algo_success = false;
                                break;
                            }

                            // check if we have reached the solution(norm of xk-xkn is less than tolerance)
                            double norm_vec = normOfVector(xk, xkn, n, iter);
                            result = f.eval(xkn);

                            if (iter == 1)
                            {

                                grad = gradVector(df, xkn); // gradient vector at pt. xk

                                for (int i = 0; i < n; i++)
                                {
                                    norm_grad += grad[i] * grad[i];
                                }
                                //fout << alp_options[k] << "," << result.lb()[0] << "," << norm_grad << "," << xk << "\n";
                            }

                            if (norm_grad < 1e-3)
                            {
                                break;
                            }

                            if (norm_vec < 1e-6)
                            {
                                if (norm_grad > 1e-3)
                                {
                                    algo_success = false;
                                }
                                break;
                            }

                            auto time_check = chrono::high_resolution_clock::now();
                            time_taken = duration_cast<std::chrono::duration<double>>(time_check - start);
                            if (time_taken.count() > duration_max)
                            {
                                algo_success = false;
                                break;
                            }

                            if( iter == 1 )
                            {
                                hessian = FiniteDifference(f, xk);
                                inverseHessian = InverseMatrix(hessian);
                            }

                            iter++;
                            for (int i = 0; i < n; i++)
                            {
                                xk[i] = xkn[i];
                            }
                            
                            direcV = DirectionVector(inverseHessian, grad);

                            gfpk = 0.0000;
                            for (int i = 0; i < n; i++)
                            {
                                gfpk += direcV[i] * grad[i];
                            }

                            /*if (gfpk > 0)
                            {
                                a = 1;
                                hessian_modify_count += 1;
                                hessian = ModifyHessianDS(hessian);
                                inverseHessian = InverseMatrix(hessian);
                                direcV = DirectionVector(inverseHessian, grad);
                                gfpk = 0.0000;
                                for (int i = 0; i < n; i++)
                                {
                                    gfpk += direcV[i] * grad[i];
                                }
                                if ((gfpk > 0) || isnan(gfpk))
                                {
                                    // The algorithm was unable to modify hessian to get descent direction
                                    algo_success = false;
                                    break;
                                }
                            }*/

                            for (int i = 0; i < n; i++)
                            {
                                xkn[i] = xk[i] + a * direcV[i];
                            }

                            IntervalVector func_value_xkn = f.eval(xkn);

                            /*if ((gfpk > 0) || isnan(gfpk))
                            {
                                cout << "Couldn't find descent direction" << "\n";
                                algo_success = false;
                                break;
                            }*/
                            // check wolfe condition and update xkn
                            while (func_value_xkn.lb()[0] > result.lb()[0] + c * a * gfpk)
                            {
                                a = a * rh;
                                if (a < a_min)
                                {
                                    break;
                                }

                                for (int i = 0; i < n; i++)
                                {
                                    xkn[i] = xk[i] + a * direcV[i];
                                    // xkn[i] = xkn[i]*domain_size + minx;
                                }
                                func_value_xkn = f.eval(xkn);
                            }
                            updated_grad = gradVector(df, xkn); // gradient vector at pt. xk
                            for(int i = 0; i < n ; i++)
                            {
                                diff_x[i] = xkn[i] - xk[i];
                                diff_grad[i] = updated_grad[i] - grad[i];
                            }
                            rho = dotProduct(diff_grad, diff_x);
                            rho = 1/rho;

                            for (int i = 0; i < n; i++)
                            {
                                for (int j = 0; j < n; j++)
                                {
                                    v1[i][j] = identityMatrix[i][j] - rho * (diff_x[i] * diff_grad[j]);
                                    v2[i][j] = identityMatrix[i][j] - rho * (diff_grad[i] * diff_x[j]);
                                }
                            }   

                            for (int i = 0; i < n; i++)
                            {
                                for (int j = 0; j < n; j++)
                                {
                                    for (int k = 0; k < n; k++)
                                    {
                                        temp(i,j) = v1[i][k] * inverseHessian(k,j);
                                    }
                                }
                            }

                            for (int i = 0; i < n; i++)
                            {
                                for (int j = 0; j < n; j++)
                                {    
                                    for (int k = 0; k < n; k++)
                                    {
                                        updated_inverseHessian(i,j) = temp(i,k) * v2[j][k];
                                    }
                                    updated_inverseHessian(i,j) += rho * (diff_x[i] * diff_x[j]);
                                }
                            }

                            inverseHessian = updated_inverseHessian;
                            grad = updated_grad;
                            //grad = gradVector(df, xkn); // gradient vector at pt. xk
                            norm_grad = 0;
                            for (int i = 0; i < n; i++)
                            {
                                norm_grad += grad[i] * grad[i];
                            }
                            // if a<a_min ==> we have reached the solution
                            if (a < a_min)
                            {
                                if (norm_grad > 1e-3)
                                {
                                    algo_success = false;
                                }
                                break;
                            }
                            a = a_init;
                            //fout << alp_options[k] << "," << func_value_xkn.lb()[0] << "," << norm_grad << "," << xkn << "\n";
                            // fout << iter << "," << func_value_xkn.lb()[0] << "\n";
                            func_value[l] = func_value_xkn.lb()[0];
                        }
                        auto finish = chrono::high_resolution_clock::now();

                        algo_iterations[l] = iter;
                        cout << "Algorithm : " << algorithms[l] << endl;
                        std ::cout << "Iterations: " << algo_iterations[l] << " & Hessian Modified " << hessian_modify_count << " times." << endl;

                        for (int i = 0; i < n; i++)
                        {
                            std ::cout << xkn[i] << " ";
                        }
                        std ::cout << endl;

                        std::chrono::duration<double> elapsed_seconds = duration_cast<std::chrono::duration<double>>(finish - start);
                        fout << test_problems_file << "," << algorithms[l] << "," << alp_options[k] << ","
                            << algo_iterations[l] << "," << func_value[l] << "," << norm_grad << ","
                            << elapsed_seconds.count() << "," << hessian_modify_count << ","
                            << (algo_success ? "Yes" : "No") << "," << ((iter > iter_max) ? "Yes" : "No") << ","
                            << ((time_taken.count() > duration_max) ? "Yes" : "No") << "," << (((gfpk > 0) || isnan(gfpk)) ? "Yes" : "No") << "\n";

                        if (iter > iter_max)
                        {
                            cout << "Algorithm : " << algorithms[l] << " took more than max iterations\n";
                            continue;
                        }

                        if (!algo_success)
                        {
                            continue;
                        }
                    }
                    if (!algo_success)
                        continue;
                    total_test++;
                    cout << "Algorithms evaluated " << total_test << " times\n";
                }
                cout << "Total Test when hessian was mofified : " << total_test << "times.\n";

                string dummy;
                getline(test_problems, dummy);
            }
            catch (ibex::SyntaxError)
            {
                fout << test_problems_file << "," << "Error in Ibex Function Syntax" << ",0,0,0,0,0,0,No,No,Yes,No\n";
                break;
            }
            catch (std::invalid_argument)
            {
                fout << test_problems_file << "," << "Error in Function(stoi)" << ",0,0,0,0,0,0,No,No,Yes,No\n";
                break;
            }
        }
        test_problems.close();
        fout.flush();
    }
    return 0;
}