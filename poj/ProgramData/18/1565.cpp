#include <iostream>
#include <math.h>
using namespace std;
int main ()
{
	int n=0; //??n 
	cin>>n;
	getchar();//??????n?????? 
	for(int i=0;i<n;i++)//??i??? 
	{
		int a[110][110];//???? 
		for (int j=0;j<n;j++)//?? 
			for (int k=0;k<n;k++)
				cin>>a[j][k];
		int sum=0;//????a[1][1]?? 
		for (int r=1;r<n;r++)//n*n???n-1??? 
		{

			for (int k=0;k<n-r+1;k++)//???n-r??????? 
			{ 
				int min=100000;//????????? 
				for (int j=0;j<n-r+1;j++)//???????????? 
				{
					if  (min>a[k][j])
						min=a[k][j];
				}
				for (int j=0;j<n-r+1;j++)
					a[k][j]=a[k][j]-min;//?????????????? 
			} 
			for (int j=0;j<n-r+1;j++)//???n-r??????? 
			{ 
				int min=10000000;//????????? 
				for (int k=0;k<n-r+1;k++)//???????????? 
				{
					if  (min>a[k][j]) 
						min=a[k][j];
				}
				for (int k=0;k<n-r+1;k++) 
					a[k][j]=a[k][j]-min;//?????????????? 
			} 
			sum=sum+a[1][1]; 
			for (int k=1;k<n-r;k++)//??????????????????????????? ???? 
			{
				for (int j=0;j<n-r+1;j++)
				{
					a[k][j]=a[k+1][j];
				}
			}
			for (int j=1;j<n-r;j++)//??????????????????????????? ???? 
			{
				for (int k=0;k<n-r;k++)
				{
					a[k][j]=a[k][j+1];
				}
			}


		}
		cout<<sum<<endl;
	}

	return 0; 
}
