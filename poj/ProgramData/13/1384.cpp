#include <iostream>
#include <math.h>
using namespace std;
/*
 *shuchubuchongfushuzi.cpp
 *?????n????n???????????????10-100???
 *???????????????????????????????????????????
 *Created on: 2012-11-11
 *Author: ??
 */
int main() {
	int n = 0;
	int shuru[100000];
	cin >> n;
	for (int i = 0; i < n; i++) {//????
		cin >> shuru[i];
		if (getchar() == '\n')
			break;
		else
			continue;
	}
	cout << shuru[0];

	for (int i = 1; i < n; i++) {//??????
		int jishu = 0;
		for (int j = 0; j < i; j++) {
			if (shuru[i] == shuru[j])
				break;
			else {
				jishu++;
			}
		}
		if (jishu == i)//????????
			cout << ' ' << shuru[i];
		else
			continue;
	}
	return 0;
}
