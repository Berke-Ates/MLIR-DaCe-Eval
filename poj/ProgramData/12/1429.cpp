#include <iostream>
#include <math.h>
using namespace std;
//********************************
//*?????   **
//*?????? 1300012745 **
//*???2013.10.31  **
//********************************
int main()
{
	int a[16], num, i=1, j=0, k=0, l=0;
	while(cin>>a[0])
	{
		//a[15]={0};
		num=0;
		//cin>>a[0];
		if (a[0]==-1)
			break;
		for (i=1;i<=15;i++)
		{
			cin >> a[i];
			if (a[i]==0)
				break;
		}
		for (j=0;j<=15;j++)
			for (k=0;k<=15;k++)
				if ((a[j] != 0) && (a[k] != 0) && (a[j] == 2 * a[k]))
					num++;
				for (l=0;l<=15;l++)
					a[l]=0;
				cout<<num<<endl;
				}
	return 0;
}
