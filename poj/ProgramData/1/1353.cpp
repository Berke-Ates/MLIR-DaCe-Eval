#include <iostream>
#include <math.h>
using namespace std;
//********************************
//*???1.cpp   **
//*?????? 1200012768 **
//*???2012.11.28  **
//*???????  **
//********************************
int fun(int n, int m);
int main()
{
	int n, num;
	cin >> n;
	for (int i = 0; i < n; i++)
	{
		cin >> num;
		cout << fun(num, 2) << endl;
	}
	return 0;
}
int fun(int n, int m)
{
	int sum = 1;					//n?????????? 
	double t = sqrt(n);
	for (int i = m; i <= t; i++)	//???????????????t 
		if (n % i == 0)				//???i?n??????????? 
			sum += fun(n /i, i);
	return sum;
}
