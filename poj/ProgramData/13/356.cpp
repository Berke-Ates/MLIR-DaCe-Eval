#include <iostream>
#include <math.h>
using namespace std;
void main()
{   int m,t,i,j,n;
    int a[20000],b[20000];
	a[20000]='\0';
	b[20000]='\0';
	scanf("%d",&n);
    /*??????*/
	for(i=0;i<n;i=i+1)
		scanf("%d",&b[i]);
	/*????????????????????*/
	a[0]=b[0];
	m=1;
	t=1;
	for(i=1;i<n;i=i+1)     
	{for(j=0;j<m;j=j+1)
	   {if(a[j]==b[i])t=0;}  /*???for?????b[i]?a?????????????????????t???0*/
	if(t==1){a[m]=b[i];/*??t=1???b[i]?a???????????????b[i]??a,????m?m????a????????*/
	m=m+1;}              /*?????????	if(t==1){a[m]=b[i];
	                                    m=m+1;
	                                   t=1;   }           ?t=1?????i??????????
	                                                           t??1?????t?1??0?????????????
	                                                                      ?t=1?t=1,t=0????????*/
	t=1;}/*?????b[i]??t???????????????????????b[i]*/
    /*?????*/
	for(i=0;i<m-1;i=i+1)
		printf("%d ",a[i]);
printf("%d",a[m-1]);
}
