#include <iostream>
#include <math.h>
using namespace std;
/*
 * ????.cpp
 *
 *  Created on: 2012-11-30
 *      Author: ??
 */
int f(int min,int a)  //????????????????min???????a????
{
	if(min>a)  return 0;  //???????a???????????????0

	int i;int num=1; //??????i????num???a=a????????1
	for(i=min;i<=sqrt(a);i++)//i?min?????????????????
	{
		if(a%i==0)  //??a?i??
		{
			num = num + f(i,a/i);//????a/i?min?i??????
		}
	}

	return num;  //????num

}
int main()      //?????
{
	int m,a,j; //m????????a???????j????
	cin>>m;     //??m
	for(j=0;j<m;j++)   //?m??
	{
		cin>>a;   //???????
		cout<<f(2,a)<<endl; //???????
	}
	return 0;   //???????????
}
