#include <iostream>
#include <math.h>
using namespace std;

void fun()        
{
    char c;           
    if((c=getchar())!='\n') 
       fun();            
    putchar(c); 

}       

main()            
{
   fun();   
   getchar();     
} 

