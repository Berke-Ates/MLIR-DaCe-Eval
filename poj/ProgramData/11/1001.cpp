#include <iostream>
#include <math.h>
using namespace std;
int main(){
	int a,b,c,d;
	cin>>a>>b>>c;
	if((a%4==0&&a%100!=0)||a%400==0){
		int m[13]={0,31,60,91,121,151,182,213,244,274,305,335,366};
		d=m[b-1]+c;}
	else {int m[13]={0,31,59,90,120,150,181,212,243,273,304,334,365};
		d=m[b-1]+c;}
	cout<<d;return 0;}
