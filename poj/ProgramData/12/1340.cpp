#include <iostream>
#include <math.h>
using namespace std;
//****************************
//**Title:??              **
//**Author:???1300017687 **
//**Date?2013.10.30        **
//**File?1.cpp             **
//****************************
int main()
{
	while(1)
	{
		int num[16] = {0};	//???????15????,???????????????0
		cin >> num[0];		//??????????
		if (num[0] == -1)	//????????-1
			break;			//????,????

		int sum = 1;		//sum????????
		for(; ;sum++)
		{
			cin >> num[sum];	//????
			if (num[sum] == 0)	//????0,?????
			{
				sum --;			//?sum??
				break;			//??????
			}
		}
		int twice = 0;			//twice?????????
		for (int i = 0 ; i < sum ; i++)		//????i?????
		{
			for(int j = i + 1 ; j <= sum ; j++)	
			{
				//??num[j]?num[i]??????
				if((num[j] == 2 * num[i])||(num[i] == 2 * num[j]))
					twice ++;	//??????
			}
		}
		cout << twice << endl;
	}
	cin.get();
	cin.get();
	return 0;
}
