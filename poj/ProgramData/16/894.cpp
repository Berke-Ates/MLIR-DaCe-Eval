#include <iostream>
#include <math.h>
using namespace std;
//????007.cpp
//??????
//?????2013?10?25?
//???????????

int main()
{
	int a[5],b[5];
	int n,i,m;
	cin>>n;
	b[0]=n;
	for(i=0;i<5;i=i+1)
	{
		a[i]=b[i]%10;
		b[i+1]=b[i]/10;
		cout<<a[i];
		if(b[i+1]==0)
			break;
	}
	return 0;
}

