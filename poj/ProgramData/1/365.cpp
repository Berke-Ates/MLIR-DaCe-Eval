#include <iostream>
#include <math.h>
using namespace std;
//**************************************
//**???12.1??��????       ***
//**?????   ???1000012888    ***
//**************************************
int f(int, int);//??f??????????
int main()
{
	int n, a, i;
	cin >> n;
	for (i = 1; i <= n; i++)
	{
		cin >> a;
		cout << f(a, 2) << endl;
	}
	return 0;
}
int f(int x, int b)//f(a,b)?a?????b?????????
{
	int j, re = 1;//??a=a???????re???0
	if(x < b)
		return 0;
	for (j = b; j < x; j++)
	{
		if(x % j == 0)
			re = re + f(x / j, j);//????????????????
	}
	return re;//??re
}
