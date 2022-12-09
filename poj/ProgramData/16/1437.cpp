#include <iostream>
#include <math.h>
using namespace std;


int main()
{
	int n,a,b,c,d,e;
	scanf("%d",&n);
	while(n!=0)
	{
	a=n%10;
	printf("%d",a);
	n=n/10;
	}
	return 0;
}


