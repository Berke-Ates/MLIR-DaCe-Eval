#include <iostream>
#include <math.h>
using namespace std;
int main(){
	int d[3],A1,B1,C1,i,j,m,a[3];
	char c[3],n;
	c[0]='A';c[1]='B';c[2]='C';
	for(d[0]=1;d[0]<=3;d[0]++)
	for(d[1]=1;d[1]<=3;d[1]++) 
	for(d[2]=1;d[2]<=3;d[2]++){
		a[0]=d[0];a[1]=d[1];a[2]=d[2];
	  if(d[0]!=d[1]&&d[0]!=d[2]&&d[1]!=d[2]){
	  	A1=(d[1]>d[0]);
	  	B1=(d[0]>d[1])+(d[0]>d[2]);
	  	C1=(d[2]>d[1])+(d[1]>d[0]);
	  	if (d[0]+A1==3&&d[1]+B1==3&&d[2]+C1==3){
	  	  for(i=0;i<2;i++)
	  	  for(j=0;j<2-i;j++){
	  	  	if(a[j]>a[j+1]) {
	  	  		m=a[j];
	  	  		a[j]=a[j+1];
	  	  		a[j+1]=m;
				n=c[j];
				c[j]=c[j+1];
				c[j+1]=n;
	  	  	    }
	  	    }
		  }
	  } 
	} 
	printf("%c%c%c\n",c[0],c[1],c[2]); 
	return 0; 
}  
