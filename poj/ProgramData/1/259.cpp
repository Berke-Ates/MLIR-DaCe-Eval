#include <iostream>
#include <math.h>
using namespace std;
int num;//num???????? 
int main()
{
    int f(int,int);  //??f?? 
    int i,n,a[100],result[100];
    cin>>n;
 
    for(i=1;i<=n;i++)
    {
                     num=1; //?????????????????1 
                     cin>>a[i];
                     result[i]=f(1,a[i]);
    }
    for(i=1;i<=n;i++)
    {
                     cout<<result[i]<<endl;
    }
}
int f(int x,int y)//f?? ???y???x�???y ?y>x 
{
    int i;
    for(i=2;i<=sqrt(y);i++) //?2???y?? ??????????????????? 
    {
                           if(y%i==0&&i>=x)//??y??i?? ??i>=x(????x,i,i????? ????????????) ??y?????i????? 
                           {
                                     num++;//????+1 
                                     f(i,y/i); //???? ??y??? 
                           }
    }
    return num;  
} 
