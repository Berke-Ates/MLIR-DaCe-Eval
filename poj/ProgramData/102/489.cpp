#include <iostream>
#include <math.h>
using namespace std;
int main()
{
	//?????????????????????????????//??????????????????????????????????????
	//int n,i,x=0,y=0,k,t;
	//char a[100];
	//int b[100],c[100],d[100];                   ///???????????????,?????????double??
	//scanf("%d",&n);
	int n,i,m,j=0,k=0;
	scanf("%d",&n);
	double male[50],female[50],daiding[100],tmp;   //daiding[100]???????d[100]?b c????male && female
	char xingbie[10];



	for(i=0;i<n;i++)
	{
		scanf("%s",xingbie);
		scanf("%lf",&daiding[i]);                 //?????????????????????????????????????????
		
		
		
		
		
		
		
		if(xingbie[0]=='m')                      //?????xingbie[0],??xingbie[1],????????a[1];
		{
			male[j]=daiding[i];
			j++;                        //j==??�x�
		}
		else if(xingbie[0]=='f')
		{
			female[k]=daiding[i];
			k++;                        //k==??�y�
		}
	}





	//?????????????








	for(i=0;i<j;i++)
	{
		for(m=0;m<j-1;m++)
		{
			if(male[m]>male[m+1])
			{
				tmp=male[m];
				male[m]=male[m+1];
				male[m+1]=tmp;
			}

		}
	}





	//???????????????????????????????????????m<j-1???m<j;??male[m+1]??????????
	for(i=0;i<k;i++)
	{
		for(m=0;m<k-1;m++)
		{
			if(female[m]>female[m+1])
			{
				tmp=female[m];
				female[m]=female[m+1];
				female[m+1]=tmp;
			}
		}
	}


	//?????????????????????????????????????????????????????????













	for(i=0;i<j;i++)
		printf("%.2lf ",male[i]);
	for(i=k-1;i>0;i--)
		printf("%.2lf ",female[i]);


	//???????????????????????????????
	printf("%.2lf",female[0]);

	//???????????



	return 0;
}
