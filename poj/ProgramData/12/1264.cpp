#include <iostream>
#include <math.h>
using namespace std;
//***********???????*************************
//***********???????*************************
//***********?????2012?10?30?***************
//***********???1200062701**********************


int main()
{
	int x[15], sum = 0, n; 
	while (1)
	{
		for (n = 0; ; n++)			// n?????
		{
			cin >> x[n];
			if (x[n] <= 0) break;       //??????0 ? -1 ????
		}
		if (x[n] == -1) break;		    // ???-1 ????
		for (int i = 0; i < n; i++)     // ?????????
		{
			for(int j = 0; j < n; j++)
			{
				if (x[j] == x[i] * 2) sum = sum + 1;
			}
		}
		cout << sum << endl;
                   sum = 0;
	}

	return 0;
}
