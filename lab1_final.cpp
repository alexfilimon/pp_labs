// Вариант № 5.	Написать программу с использованием технологии OpenMP, 
// которая реализует следующие действия: формирует два массива А и В, 
// размерностью N x N и формирует новый массив, 
// элементы которого равны sin(A[I,j])2+cos(B[I,j])3 
// из соответствующих элементов исходных массивов.
// Найти сумму элементов результирующего массива.

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

// ----------------
// |   FORMULA    |
// ----------------

double formula(double a, double b) {
    return pow(sin(a), 2) + pow(cos(b), 3);
}

// ----------------
// | CALCULATIONS |
// ----------------

vector<vector<double>> parallelScheduleStaticCalculate(vector<vector<double>> arrayA, vector<vector<double>> arrayB)
{
	int size = arrayA.size();
	vector<vector<double>>array(size, vector<double>(size, 0.0));

#pragma omp parallel shared(arrayA, arrayB) private(i, size);
	{
		for (int i = 0; i < size; i++)
#pragma omp for schedule (static)
			for (int j = 0; j < size; j++)
				array[i][j] = formula(arrayA[i][j], arrayB[i][j]);

		return array;
	}
}

vector<vector<double>> parallelScheduleDynamicCalculate(vector<vector<double>> arrayA, vector<vector<double>> arrayB)
{
	int size = arrayA.size();
	vector<vector<double>>array(size, vector<double>(size, 0.0));

#pragma omp parallel shared(arrayA, arrayB) private(i, size); 
	{
		for (int i = 0; i < size; i++)
#pragma omp for schedule (dynamic)
			for (int j = 0; j < size; j++)
				array[i][j] = formula(arrayA[i][j], arrayB[i][j]);

		return array;
	}
}

vector<vector<double>> parallelScheduleDynamic2Calculate(vector<vector<double>> arrayA, vector<vector<double>> arrayB)
{
	int size = arrayA.size();
	vector<vector<double>>array(size, vector<double>(size, 0.0));

#pragma omp parallel shared(arrayA, arrayB) private(i, size); 
	{
		for (int i = 0; i < size; i++)
#pragma omp for schedule (dynamic, 2)
			for (int j = 0; j < size; j++)
				array[i][j] = formula(arrayA[i][j], arrayB[i][j]);

		return array;
	}
}

vector<vector<double>> parallelScheduleDynamic4Calculate(vector<vector<double>> arrayA, vector<vector<double>> arrayB)
{
	int size = arrayA.size();
	vector<vector<double>>array(size, vector<double>(size, 0.0));

#pragma omp parallel shared(arrayA, arrayB) private(i, size); 
	{
		for (int i = 0; i < size; i++)
#pragma omp for schedule (dynamic, 4)
			for (int j = 0; j < size; j++)
				array[i][j] = formula(arrayA[i][j], arrayB[i][j]);

		return array;
	}
}

vector<vector<double>> parallelScheduleDynamic6Calculate(vector<vector<double>> arrayA, vector<vector<double>> arrayB)
{
	int size = arrayA.size();
	vector<vector<double>>array(size, vector<double>(size, 0.0));

#pragma omp parallel shared(arrayA, arrayB) private(i, size); 
	{
		for (int i = 0; i < size; i++)
#pragma omp for schedule (dynamic, 6)
			for (int j = 0; j < size; j++)
				array[i][j] = formula(arrayA[i][j], arrayB[i][j]);

		return array;
	}
}

vector<vector<double>> parallelScheduleGuidedCalculate(vector<vector<double>> arrayA, vector<vector<double>> arrayB)
{
	int size = arrayA.size();
	vector<vector<double>>array(size, vector<double>(size, 0.0));

#pragma omp parallel shared(arrayA, arrayB) private(i, size); 
	{
		for (int i = 0; i < size; i++)
#pragma omp for schedule (guided) 
			for (int j = 0; j < size; j++)
				array[i][j] = formula(arrayA[i][j], arrayB[i][j]);

		return array;
	}
}

vector<vector<double>> parallelColumnsCalculate(vector<vector<double>> arrayA, vector<vector<double>> arrayB)
{
	int size = arrayA.size();
	vector<vector<double>>array(size, vector<double>(size, 0.0));

#pragma omp parallel shared(arrayA, arrayB) private(i, size);
	{
		for (int i = 0; i < size; i++)
#pragma omp for
			for (int j = 0; j < size; j++)
				array[i][j] = formula(arrayA[i][j], arrayB[i][j]);

		return array;
	}
}

vector<vector<double>> parallelBlocksCalculate(vector<vector<double>> arrayA, vector<vector<double>> arrayB)
{
	int size = arrayA.size();
	vector<vector<double>>array(size, vector<double>(size, 0.0));

#pragma omp parallel shared(arrayA, arrayB) private(i, size);
	{
#pragma omp for
		for (int i = 0; i < size; i++)
#pragma omp for
			for (int j = 0; j < size; j++)
				array[i][j] = formula(arrayA[i][j], arrayB[i][j]);

		return array;
	}
}

vector<vector<double>> parallelRowsCalculate(vector<vector<double>> arrayA, vector<vector<double>> arrayB)
{
	int size = arrayA.size();
	vector<vector<double>>array(size, vector<double>(size, 0.0));
#pragma omp parallel 
	{
#pragma omp for
		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
				array[i][j] = formula(arrayA[i][j], arrayB[i][j]);

		return array;
	}
}

vector<vector<double>> seqentialCalculate(vector<vector<double>> arrayA, vector<vector<double>> arrayB)
{
	int size = arrayA.size();
	vector<vector<double>>array(size, vector<double>(size, 0.0));

	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			array[i][j] = formula(arrayA[i][j], arrayB[i][j]);

	return array;
}

double averageDeviationCalculate(vector<vector<double>> arrayA, vector<vector<double>> arrayB, vector<vector<double>> arrayC)
{
	int size = arrayA.size();
	double controlSum = 0.0;
	int controlCounter = 0;
	vector<vector<double>>controlArray(size, vector<double>(size, 0.0));
	controlArray = seqentialCalculate(arrayA, arrayB);

	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++, controlCounter++)
			controlSum += controlArray[i][j] - arrayC[i][j];

	return controlSum / controlCounter;
}

// ----------------
// |  TEST TYPES  |
// ----------------

enum class TestTypes
{
    SEQENTIAL,
    PARALLEL_ROWS,
    PARALLEL_COLUMNS,
    PARALLEL_BLOCKS,
    PARALLEL_SCHEDULE_STATIC,
    PARALLEL_SCHEDULE_DYNAMIC,
    PARALLEL_SCHEDULE_GUIDED,
    PARALLEL_SCHEDULE_DYNAMIC_2,
    PARALLEL_SCHEDULE_DYNAMIC_4,
    PARALLEL_SCHEDULE_DYNAMIC_6,
};

template<typename T>
string getTestType(T index)
{
    switch (index)
    {
    case TestTypes::SEQENTIAL: 
        return "Seqential method"; break;
    case TestTypes::PARALLEL_ROWS: 
        return "Parallel by rows method"; break;
    case TestTypes::PARALLEL_COLUMNS: 
        return "Parallel by columns method"; break;
    case TestTypes::PARALLEL_BLOCKS: 
        return "Parallel by blocks method"; break;
    case TestTypes::PARALLEL_SCHEDULE_STATIC: 
        return "Parallel by shedule static method"; break;
    case TestTypes::PARALLEL_SCHEDULE_DYNAMIC: 
        return "Parallel by shedule dynamic method"; break;
    case TestTypes::PARALLEL_SCHEDULE_DYNAMIC_2: 
        return "Parallel by shedule dynamic(2) method"; break;
    case TestTypes::PARALLEL_SCHEDULE_DYNAMIC_4: 
        return "Parallel by shedule dynamic(4) method"; break;
    case TestTypes::PARALLEL_SCHEDULE_DYNAMIC_6: 
        return "Parallel by shedule dynamic(6) method"; break;
    case TestTypes::PARALLEL_SCHEDULE_GUIDED: 
        return "Parallel by shedule guided method"; break;
    default: 
        return "Unknown method";
    }
}

// ----------------
// |   HELPERS    |
// ----------------

vector<vector<double>> randomMatrix(int size, double min, double max)
{
	minstd_rand rand(clock());
	uniform_int_distribution<> distribution(min, max);

	vector<vector<double>>array(size, vector<double>(size, 0.0));

	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			array[i][j] = distribution(rand) * randomCoef;

	return array;
}

void displayMatrix(vector<vector<double>> array)
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

void printStatistics(vector<vector<double>> array, vector<TestTypes> types) {
	cout << "\nAnalysys:\n";
	vector<double>averages(array.size(), 0.0);

	for (int i = 0; i < array.size(); i++)
	{
		cout << getTestType(types[i]) << " test:";
		auto minmax = minmax_element(array[i].begin(), array[i].end());
		cout << " Min: " << *minmax.first;
		cout << " Max: " << *minmax.second;
		double average = (*minmax.first + *minmax.second) / array[i].size();
		cout << " Avg: " << average << endl;
		averages[i] = average;
	}

	cout << "\nConclusion:\n";
	cout << "Best method: " << getTestType(types[min_element(averages.begin(), averages.end()) - averages.begin()]) << " (" << *min_element(averages.begin(), averages.end()) << ")\n";
	cout << "Worst method: " << getTestType(types[max_element(averages.begin(), averages.end()) - averages.begin()]) << " (" << *max_element(averages.begin(), averages.end()) << ")\n\n";
}

// ----------------
// |   TESTING    |
// ----------------

double performTest(vector<vector<double>> arrayA, vector<vector<double>> arrayB, TestTypes type) {
	int size = arrayA.size();
	vector<vector<double>>arrayC(size, vector<double>(size, 0.0));
	double start = omp_get_wtime();

	switch (type)
	{
	case TestTypes::PARALLEL_SCHEDULE_GUIDED: 
        arrayC = parallelScheduleGuidedCalculate(arrayA, arrayB); break;
	case TestTypes::PARALLEL_SCHEDULE_DYNAMIC: 
        arrayC = parallelScheduleDynamicCalculate(arrayA, arrayB); break;
	case TestTypes::PARALLEL_SCHEDULE_DYNAMIC_2: 
        arrayC = parallelScheduleDynamic2Calculate(arrayA, arrayB); break;
	case TestTypes::PARALLEL_SCHEDULE_DYNAMIC_4: 
        arrayC = parallelScheduleDynamic4Calculate(arrayA, arrayB); break;
	case TestTypes::PARALLEL_SCHEDULE_DYNAMIC_6: 
        arrayC = parallelScheduleDynamic6Calculate(arrayA, arrayB); break;
	case TestTypes::PARALLEL_SCHEDULE_STATIC: 
        arrayC = parallelScheduleStaticCalculate(arrayA, arrayB); break;
	case TestTypes::PARALLEL_BLOCKS: 
        arrayC = parallelBlocksCalculate(arrayA, arrayB); break;
	case TestTypes::PARALLEL_COLUMNS: 
        arrayC = parallelColumnsCalculate(arrayA, arrayB); break;
	case TestTypes::PARALLEL_ROWS: 
        arrayC = parallelRowsCalculate(arrayA, arrayB); break;
	case TestTypes::SEQENTIAL: 
        arrayC = seqentialCalculate(arrayA, arrayB); break;
	default: break;
	}

	double end = omp_get_wtime(), time = (end - start) * 1000;

	// cout << "Avg deviate: " << averageDeviationCalculate(arrayA, arrayB, arrayC) << " time: " << time << endl;
	if (averageDeviationCalculate(arrayA, arrayB, arrayC) != 0.0)
		cout << "ERROR: arrayC not correct";

	return time;
}


void performTests(vector<vector<double>> arrayA, vector<vector<double>> arrayB, vector<TestTypes> types, int count)
{
	vector<vector<double>> times(types.size(), vector<double>(count, 0.0));

	for (int i = 0; i < types.size(); i++)
	{
		cout << getTestType(types[i]) << " testing...\n";
		for (int j = 0; j < count; j++)
			times[i][j] = performTest(arrayA, arrayB, types[i]);

		cout << getTestType(types[i]) << " tested\n\n";
	}

	printStatistics(times, types);
}

// ----------------
// |     MAIN     |
// ----------------


int main()
{
	int size = 0;
	cout << "Input size of array:\n";
	cin >> size;
	cout << "Size of array is " << size << endl << endl;

	cout << "Randomizing array A...\n";
	vector<vector<double>> arrayA(size, vector<double>(size, 0.0));
	arrayA = randomMatrix(size, minValue, maxValue);
	cout << "Array A generated\n\n";

	cout << "Randomizing array B...\n";
	vector<vector<double>> arrayB(size, vector<double>(size, 0.0));
	arrayB = randomMatrix(size, minValue, maxValue);
	cout << "Array B generated\n\n";

	int count = 0;
	cout << "Input number of tests:\n";
	cin >> count;
	cout << "Numer of tests: " << count << endl << endl;

    cout << "----------------------------------------------------------------------\n";
    cout << "| Types: SEQENTIAL, PARALLEL_ROWS/COLUMNS/BLOCKS |\n";
    cout << "----------------------------------------------------------------------\n";
	vector<TestTypes> types = { TestTypes::SEQENTIAL, TestTypes::PARALLEL_ROWS, TestTypes::PARALLEL_COLUMNS, TestTypes::PARALLEL_BLOCKS };
	performTests(arrayA, arrayB, types, count);

    cout << "-------------------------------------------\n";
    cout << "| Types: PARALLEL_SCHEDULE_STATIC/DYNAMIC |\n";
    cout << "-------------------------------------------\n";
	types = { TestTypes::PARALLEL_SCHEDULE_STATIC, TestTypes::PARALLEL_SCHEDULE_DYNAMIC, TestTypes::PARALLEL_SCHEDULE_GUIDED };
	performTests(arrayA, arrayB, types, count);

    cout << "------------------------------------------\n";
    cout << "| Types: PARALLEL_SCHEDULE_DYNAMIC_2/4/6 |\n";
    cout << "------------------------------------------\n";
	types = { TestTypes::PARALLEL_SCHEDULE_DYNAMIC_2, TestTypes::PARALLEL_SCHEDULE_DYNAMIC_4, TestTypes::PARALLEL_SCHEDULE_DYNAMIC_6 };
	performTests(arrayA, arrayB, types, count);
}