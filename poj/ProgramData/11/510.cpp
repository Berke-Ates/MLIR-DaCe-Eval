#include <iostream>
#include <math.h>
using namespace std;
int main()
{
	int a,b,c,x;
    scanf ("%d%d%d",&a,&b,&c);
		
	
	 if (b==1) x=c;
	 if (b==2) x=31+c;
     if (b==3) x=31+28+c;
     if (b==4) x=31+28+31+c;
     if (b==5) x=31+28+31+30+c;
     if (b==6) x=31+28+31+30+31+c;
     if (b==7) x=31+28+31+30+31+30+c;
     if (b==8) x=31+28+31+30+31+30+31+c;
     if (b==9) x=31+28+31+30+31+30+31+31+c;
     if (b==10) x=31+28+31+30+31+30+31+31+30+c;
     if (b==11) x=31+28+31+30+31+30+31+31+30+31+c;
     if (b==12) x=31+28+31+30+31+30+31+31+30+31+30+c;

	if (b<=2) printf("%d\n",x);
	else
		if (a%4==0)
			if (a%100==0)
				if (a%400==0)
					printf("%d\n",x+1);
				else
					printf("%d\n",x);
			else
				printf("%d\n",x+1);
		else 
			printf("%d\n",x);
		return 0;
}
