#include <iostream>
#include <math.h>
using namespace std;
int up(int x)                         //????????????????
{
	if(x%2==1)return (x-1)/2;
	else return x/2;
}
int main()
{
	int x,y;
	cin>>x>>y;
	while(x!=y)                  //?x???y????????
	{
		if(abs(x-y)==1)         //?????x?y??????????
		{
			y=up(y);   
			x=up(x);
		}
		else
		{
			if(y>x)y=up(y);  //????????????
			else x=up(x);
		}
	}
	cout<<x<<endl;   // ????????
	return 0;
}
