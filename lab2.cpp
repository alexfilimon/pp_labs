// Вариант № 16. Разработать программу для решения СЛУ методом простой итерации.

#include <iostream>
#include <vector>
#include <omp.h>
#include <ctime>
#include <random>
#include <string>
#include <cmath>

using namespace std;

// ------------------
// |   CONSTANTS    |
// ------------------

const double minValue = 0.0;
const double maxValue = 360.0;
const double degreesToRadiansCoefficient = 0.0174533;

// ------------------
// |   VARIABLES    |
// ------------------

const double eps = 0.0000000000001;
int globalCyclesCount = 0;
double globalDeviage = 0.0;

// ----------------
// | CALCULATIONS |
// ----------------

bool isDiagonalDominanceBroken(vector<vector<double>> matrix)
{
    bool flag = true;
    int size = matrix.size();
    for (int i = 0; i < size && flag; i++)
    {
        double fabsSum = 0.0;
        for (int j = 0; j < size; j++)
            if (i != j)
                fabsSum += fabs(matrix[i][j]);
        flag = fabs(matrix[i][i]) <= fabsSum;
    }
    return flag;
}

bool isDiverged(vector<double> result, vector<double> temp, double eps)
{
    bool flag = true;
    int size = result.size();
    for (int i = 0; i < size && flag; i++)
        flag = fabs(temp[i] - result[i]) < eps;
    return flag;
}

vector<double> parallelAlgorithmCalculate(vector<vector<double>> matrix)
{
    int size = matrix.size();
    vector<double> result(size, 0.0);
    vector<double> temp(size, 0.0);
    int count = 0;
    for (bool flag = !isDiagonalDominanceBroken(matrix); flag; count++)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            double sum = 0.0;
            {
#pragma omp parallel for
                for (int j = 0; j < size; j++)
                    if (i != j)
                        sum += matrix[i][j] * result[j];
                temp[i] = (matrix[i][size] - sum) / matrix[i][i];
            }
        }
        flag = !isDiverged(result, temp, eps);
        if (flag)
            for (int i = 0; i < size; i++)
                result[i] = temp[i];
    }
    globalCyclesCount = count;
    return result;
}

vector<double> parallelSectionsCalculate(vector<vector<double>> matrix)
{

    int size = matrix.size();
    vector<double> result(size, 0.0);
    vector<double> temp(size, 0.0);
    int count = 0;

    for (bool flag = !isDiagonalDominanceBroken(matrix); flag; count++)
    {
#pragma omp parallel sections
        {
#pragma omp section
            {
                for (int i = 0; i < size * 0.25; i++)
                {
                    double sum = 0.0;

                    for (int j = 0; j < size; j++)
                        if (i != j)
                            sum += matrix[i][j] * result[j];
                    temp[i] = (matrix[i][size] - sum) / matrix[i][i];
                }
            }
#pragma omp section
            {
                for (int i = size * 0.25; i < size * 0.5; i++)
                {
                    double sum = 0.0;

                    for (int j = 0; j < size; j++)
                        if (i != j)
                            sum += matrix[i][j] * result[j];
                    temp[i] = (matrix[i][size] - sum) / matrix[i][i];
                }
            }
#pragma omp section
            {
                for (int i = size * 0.5; i < size * 0.75; i++)
                {
                    double sum = 0.0;

                    for (int j = 0; j < size; j++)
                        if (i != j)
                            sum += matrix[i][j] * result[j];
                    temp[i] = (matrix[i][size] - sum) / matrix[i][i];
                }
            }
#pragma omp section
            {
                for (int i = size * 0.75; i < size; i++)
                {
                    double sum = 0.0;

                    for (int j = 0; j < size; j++)
                        if (i != j)
                            sum += matrix[i][j] * result[j];
                    temp[i] = (matrix[i][size] - sum) / matrix[i][i];
                }
            }
        }
        flag = !isDiverged(result, temp, eps);
        if (flag)
            for (int i = 0; i < size; i++)
                result[i] = temp[i];
    }
    globalCyclesCount = count;
    return result;
}

vector<double> parallelCyclesCalculate(vector<vector<double>> matrix)
{
    int size = matrix.size();
    vector<double> result(size, 0.0);
    vector<double> temp(size, 0.0);
    int count = 0;
    for (bool flag = !isDiagonalDominanceBroken(matrix); flag; count++)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            double sum = 0.0;
            {
#pragma omp parallel for
                for (int j = 0; j < size; j++)
                    if (i != j)
                        sum += matrix[i][j] * result[j];
                temp[i] = (matrix[i][size] - sum) / matrix[i][i];
            }
        }
        flag = !isDiverged(result, temp, eps);
        if (flag)
#pragma omp parallel for
            for (int i = 0; i < size; i++)
                result[i] = temp[i];
    }
    globalCyclesCount = count;
    return result;
}

vector<double> sequentialCalculate(vector<vector<double>> matrix)
{
    int size = matrix.size();
    vector<double> result(size, 0.0);
    vector<double> temp(size, 0.0);
    int count = 0;
    for (bool flag = !isDiagonalDominanceBroken(matrix); flag; count++)
    {
        for (int i = 0; i < size; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < size; j++)
                if (i != j)
                    sum += matrix[i][j] * result[j];
            temp[i] = (matrix[i][size] - sum) / matrix[i][i];
        }
        flag = !isDiverged(result, temp, eps);
        if (flag)
            for (int i = 0; i < size; i++)
                result[i] = temp[i];
    }
    globalCyclesCount = count;
    return result;
}

double averageDeviationCalculate(vector<vector<double>> matrix, vector<double> result)
{
    int size = matrix.size();
    double deviage = 0.0;
    for (int i = 0; i < size; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < size; j++)
            sum += matrix[i][j] * result[j];
        deviage += sum - matrix[i][size];
    }
    return deviage / size;
}

// ----------------
// |  TEST TYPES  |
// ----------------

enum class TestTypes
{
	SEQUENTIAL,
	PARALLEL_CYCLES,
	PARALLEL_SECTIONS,
	PARALLEL_ALGORITHM,
};

template<typename T>
void getTestType(T index)
{
	switch (index)
	{
	case TestTypes::SEQUENTIAL: cout << "Sequential method"; break;
	case TestTypes::PARALLEL_CYCLES: cout << "Parallel with cycles method"; break;
	case TestTypes::PARALLEL_SECTIONS: cout << "Parallel with sections method"; break;
	case TestTypes::PARALLEL_ALGORITHM: cout << "Parallel with algorithm method"; break;
	default: cout << "Unknown method";
	}
}

// ----------------
// |   HELPERS    |
// ----------------

vector<vector<double>> random(int size, double min, double max)
{
	minstd_rand rand(clock());
	uniform_int_distribution<> distribution(min, max);

	vector<vector<double>>array(size, vector<double>(size, 0.0));

	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			array[i][j] = distribution(rand) * degreesToRadiansCoefficient;

	return array;
}

void display(vector<vector<double>> array)
{
	for (int i = 0; i < array.size(); i++)
	{
		for (int j = 0; j < array[i].size(); j++)
			cout << array[i][j] << " ";
		cout << endl;
	}
	cout << endl;
}


// ----------------
// |   ANALYZE    |
// ----------------


void analyse(vector<vector<double>> array, vector<TestTypes> types) {
	cout << "\nAnalysys:\n";
	vector<double>averages(array.size(), 0.0);

	for (int i = 0; i < array.size(); i++)
	{
		getTestType(types[i]);
		cout << " test:";
		auto minmax = minmax_element(array[i].begin(), array[i].end());
		cout << " Min: " << *minmax.first;
		cout << " Max: " << *minmax.second;
		double average = (*minmax.first + *minmax.second) / 2.0;
		cout << " Avg: " << average << endl;
		averages[i] = average;
	}

	cout << "\nConclusion:\n";
	cout << "Best method: ";
	getTestType(types[min_element(averages.begin(), averages.end()) - averages.begin()]);
	cout << " (" << *min_element(averages.begin(), averages.end());
	cout << ")\nWorst method: ";
	getTestType(types[max_element(averages.begin(), averages.end()) - averages.begin()]);
	cout << " (" << *max_element(averages.begin(), averages.end()) << ")\n\n";
}

// ----------------
// |   TESTING    |
// ----------------


double test(vector<vector<double>> matrix, TestTypes type) {
    int size = matrix.size();
    vector<double> result(size, 0.0);
    double start = omp_get_wtime();
    switch (type)
    {
    case TestTypes::SEQUENTIAL: result = sequentialCalculate(matrix); break;
    case TestTypes::PARALLEL_CYCLES: result = parallelCyclesCalculate(matrix); break;
    case TestTypes::PARALLEL_SECTIONS: result = parallelSectionsCalculate(matrix); break;
    case TestTypes::PARALLEL_ALGORITHM: result = parallelAlgorithmCalculate(matrix); break;
    default: break;
    }
    double end = omp_get_wtime(), time = (end - start) * 1000;
    globalDeviage = averageDeviationCalculate(matrix, result);
    return time;
}


void tests(vector<vector<double>> matrix, vector<TestTypes> types, int count)
{
    vector<vector<double>> times(types.size(), vector<double>(count, 0.0));
    for (int i = 0; i < types.size(); i++)
    {
        getTestType(types[i]);
        cout << " test... ";
        for (int j = 0; j < count; j++)
            times[i][j] = test(matrix, types[i]);
        cout << "successfully with " << globalCyclesCount << " iterations and deviage: " << globalDeviage << endl;
    }
    analyse(times, types);
}

// -----------------
// | INITIALIZAION |
// -----------------


vector<vector<double>> init(int depth, double min, double max)
{
	minstd_rand rand(clock());
	uniform_int_distribution<> distribution(min, max);

	int width = depth + 1;
	vector<vector<double>>matrix(depth, vector<double>(width, 0.0));
	vector<double>solution(depth, 0.0);


	for (int i = 0; i < depth; i++)
		for (int j = 0; j < depth; j++)
			matrix[i][j] = distribution(rand);

	for (int i = 0; i < depth; i++)
	{
		int fabsSum = 0;
		for (int j = 0; j < depth; j++)
			if (j != i)
				fabsSum += fabs(matrix[i][j]);
		while (fabs(matrix[i][i]) < fabsSum)
			matrix[i][i] += distribution(rand);
	}

	for (int i = 0; i < depth; i++)
		solution[i] = distribution(rand);

	for (int i = 0; i < depth; i++)
	{
		double sum = 0.0;
		for (int j = 0; j < depth; j++)
			sum += matrix[i][j] * solution[j];
		matrix[i][depth] = sum;
	}
	return matrix;
}

// ----------------
// |     MAIN     |
// ----------------

int main()
{
	int size = 0;

	cout << "Matrix size: ";
	cin >> size;

	cout << "Generate matrix... ";
	vector<vector<double>> matrix(size, vector<double>(size, 0.0));
	matrix = init(size, -size * 1.0, size * 1.0);
	cout << "successfully\n";

	int count = 0;
	cout << "Tests count: ";
	cin >> count;

	vector<TestTypes> types = { TestTypes::SEQUENTIAL, TestTypes::PARALLEL_CYCLES, TestTypes::PARALLEL_SECTIONS, TestTypes::PARALLEL_ALGORITHM };
	tests(matrix, types, count);
}