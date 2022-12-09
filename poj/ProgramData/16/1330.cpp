#include <iostream>
#include <math.h>
using namespace std;
int main()
{
	int i;
	scanf("%d",&i);
	do
	{
		printf("%d",i%10);
		i=i/10;
		
	}while(i!=0);
		return 0;
}
