#include <iostream>
#include <math.h>
using namespace std;
/*
 *  appetite.cpp
 *  Author: ???
 *  Created on: 2012-10-23
 *  function????
 */
int main()
{
	int A,As,B,Bs,C,Cs,i;//????
	char app[4][2];//???????
	for(A=1;A<=3;A++)//????
		for(B=1;B<=3;B++)
		{
			if(A==B)continue;
			for(C=1;C<=3;C++)
			{
				if(A==C||B==C)continue;//???????????????????????????
				As=(A<B)+(A==C);//A???
				Bs=(A>B)+(A>C);//B???
				Cs=(B<C)+(B>A);//C???
				if(As+A==3&&Bs+B==3&&Cs+C==3)//?????????????????
				{
					strcpy(app[A],"A");//????A?app[A]
					strcpy(app[B],"B");//????B?app[B]
					strcpy(app[C],"C");//????C?app[C]
					for(i=1;i<=3;i++)//?????????
						cout<<app[i];
				}
		    }
		}
	return 0;//????
}
