#include <iostream>
#include <math.h>
using namespace std;
int main()
{
	int a[16]={-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2},sum=0,i,j,s,t;

A:	for(i=0;a[i]!=-1;i++)
{
		sum=0;
a[0]=-2;a[1]=-2;a[2]=-2;a[3]=-2;a[4]=-2;a[5]=-2;a[6]=-2;a[15]=-2;a[7]=-2;a[8]=-2;a[9]=-2;a[10]=-2;a[11]=-2;a[12]=-2;a[13]=-2;a[14]=-2;
	for(s=0;s<=15;s++)
	{
		scanf("%d",&a[s]);
		if(a[s]==0)
		{
           for(t=0;t<=15;t++)
		   {
			   for(j=0;j<=15;j++)
			   {
				   if(a[j]!=0&&a[t]!=0&&a[t]==a[j]*2)
					   sum++;
			   }
		   }
		   printf("%d\n",sum);
		   break;
		}
		else if(a[s]==-1)
			goto A;
	}

}
 
	return 0;
}
