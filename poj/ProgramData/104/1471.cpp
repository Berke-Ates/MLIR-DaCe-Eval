#include <iostream>
#include <math.h>
using namespace std;
/*
 * Homework7.6BinaryTree.cpp
 *
 *  Created on: 2013-11-22
 *      Author: JuiceHe
 */
int main()
{
	int x,y;
	cin>>x>>y;
	int find(int,int);
	cout<<find(x,y)<<endl;
}
int find(int x,int y)
{
	if(x==y)return x;//??????????????
	if(x>y)return (find(x/2,y));//??????????????????????
	else return (find(x,y/2));
}
