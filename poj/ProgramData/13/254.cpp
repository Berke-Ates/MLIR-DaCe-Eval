#include <iostream>
#include <math.h>
using namespace std;
//???????????
//?????? 1000012918
//???10.28
//???
int main()
{
	int n, count[101] = {0}, num;            //count[num]???num????????????0
	cin >> n;
	for (int i = 1; i <= n; i ++)
	{
		cin >> num;  
		if (count[num] == 0)                     //?????????
			if (i == 1)
				cout << num;
			else
			cout << " " << num;
			count[num]++;                       //?????????1
	}
	return 0;
}
