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
const double randomCoef = 0.0174533;

#define EPS 0.1;

// ------------------
// |   VARIABLES    |
// ------------------

int globalCyclesCount = 0;
double globalDeviage = 0.0;

// ----------------
// | CALCULATIONS |
// ----------------

vector<double> calculate(vector<vector<double>> matrix, vector<double>(*f)(vector<double> current, vector<vector<double>> alpha, vector<double> betta))
{
    int size = matrix.size();
    
    // calculate betta
    vector<double> betta(size, 0.0);
    for (int i = 0; i < size; i++) 
    {
        betta[i] = matrix[i][size] / matrix[i][i];
    }

    // calculate betta
    vector<vector<double>> alpha(size, vector<double>(size, 0.0));
    for (int i = 0; i < size; i++) 
    {
        for (int j = 0; j < size; j++) 
        {
            if (i != j)
            {
                alpha[i][j] = - matrix[i][j] / matrix[i][i];
            }
        }
    }

    // initial result
    vector<double> currentResult(size, 0.0);
    for (int i = 0; i < size; i++) 
    {
        currentResult[i] = betta[i];
    }

    int cyclesCount = 0;
    bool foundResult = false;
    for (cyclesCount = 0; cyclesCount < 100 && !foundResult; cyclesCount++) 
    {
        // calculate new result
        vector<double> newResult = f(currentResult, alpha, betta);

        // check epsilon
        bool hasGreaterEpsValue = false;
        int i = 0;
        for (i = 0; i < size && !hasGreaterEpsValue; i++)
        {
            hasGreaterEpsValue = fabs(currentResult[i] - newResult[i]) > EPS;
        }

        // check if should end calculation
        if (!hasGreaterEpsValue) 
        {
            foundResult = true;
        }
        
        // copy new result to current result
        for (int j = 0; j < size; j++) 
        {
            currentResult[j] = newResult[j];
        }
    }
    
    globalCyclesCount = cyclesCount;
    return currentResult;
}

vector<double> calculateSeq(vector<double> currentResult, vector<vector<double>> alpha, vector<double> betta) {
    int size = currentResult.size();
    vector<double> newResult(size, 0.0);
    for (int i = 0; i < size; i++) 
    {
        double sum = betta[i];
        for (int j = 0; j < size; j++) 
        {
            sum += alpha[i][j] * currentResult[j];
        }
        newResult[i] = sum;
    }
    return newResult;
}

vector<double> calculateParallel(vector<double> currentResult, vector<vector<double>> alpha, vector<double> betta) {
    int size = currentResult.size();
    vector<double> newResult(size, 0.0);
    int i = 0;
#pragma omp parallel for shared(newResult) private(i) schedule (static)
    for (i = 0; i < size; i++) 
    {
        double sum = betta[i];
        for (int j = 0; j < size; j++) 
        {
            sum += alpha[i][j] * currentResult[j];
        }
        newResult[i] = sum;
    }
    return newResult;
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
	PARALLEL
};

template<typename T>
void getTestType(T index)
{
	switch (index)
	{
	case TestTypes::SEQUENTIAL: 
        cout << "Sequential method"; break;
	case TestTypes::PARALLEL: 
        cout << "Parallel method"; break;
	default: 
        cout << "Unknown method";
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
			array[i][j] = distribution(rand) * randomCoef;

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

double dRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
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


double performTest(vector<vector<double>> matrix, TestTypes type) {
    int size = matrix.size();
    vector<double> result(size, 0.0);
    double start = omp_get_wtime();
    switch (type)
    {
    case TestTypes::SEQUENTIAL: 
        // result = sequentialCalculate(matrix); 
        result = calculate(matrix, &calculateSeq);
        break;
    case TestTypes::PARALLEL: 
        // result = parallelCyclesCalculate(matrix); 
        result = calculate(matrix, &calculateParallel);
        break;
    default: break;
    }
    double end = omp_get_wtime(), time = (end - start) * 1000;
    globalDeviage = averageDeviationCalculate(matrix, result);
    return time;
}


void performTests(vector<vector<double>> matrix, vector<TestTypes> types, int count)
{
    vector<vector<double>> times(types.size(), vector<double>(count, 0.0));
    for (int i = 0; i < types.size(); i++)
    {
        getTestType(types[i]);
        cout << " test... ";
        for (int j = 0; j < count; j++)
            times[i][j] = performTest(matrix, types[i]);
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
        matrix[i][i] = fabsSum * dRand(1.1, 3.0);
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

	cout << "Size of matrix: ";
	cin >> size;

	cout << "Randomizing matrix... ";
	vector<vector<double>> matrix(size, vector<double>(size, 0.0));
	matrix = init(size, -size * 1.0, size * 1.0);
	cout << "successfully\n";

	int count = 0;
	cout << "Number of tests: ";
	cin >> count;

    cout << "---------------------------------------------------------\n";
    cout << "| Types: SEQUENTIAL, PARALLEL |\n";
    cout << "---------------------------------------------------------\n";
	vector<TestTypes> types = { TestTypes::SEQUENTIAL, TestTypes::PARALLEL };
	performTests(matrix, types, count);
}