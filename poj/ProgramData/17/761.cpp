#include <iostream>
#include <math.h>
using namespace std;
/*
 * ????match.cpp
 * ???? : 2012-12-4
 * ??: winneshaw
 * ??????????????????
 * ??????????????????????????????????
 * ???????????"$"??,?????????"?"??
 *
 */


char str[101];
int lenth = 0;
int main() {
	while(cin >> str){
		lenth = strlen(str);
		cout << str << endl;
		int i, j;
		for (j = 0; j < lenth; j++) {
			if (str[j] == '(')
				str[j] = '$';
			else if (str[j] == ')')
				str[j] = '?';
			else
				str[j] = ' ';
		}
		for (i = lenth - 2; i >= 0; i--) {
			if (str[i] == '$') {
				for (j = i + 1; j < lenth; j++) {
					if (str[j] == '?') {
						str[i] = ' ';
						str[j] = ' ';
						break;
					}
				}
			}
		}/*
		if (str[lenth - 1] == '?')
			for (i = 0; i < lenth; i++)
				if (str[i] == '$') {
					str[lenth - 1] = ' ';
					str[i] = ' ';
					break;
				}*/
		cout << str << endl;
	}
	return 0;
}
