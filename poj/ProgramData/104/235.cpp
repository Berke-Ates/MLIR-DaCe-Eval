#include <iostream>
#include <math.h>
using namespace std;
/*
 * ???????.cpp
 * ??????????x?y???????????1000
 *      ?????????????????
 * ????: 2010-11-20
 * ??: ??
 */



int main() {
	//??x?y
	int x, y;
	cin >> x;
	cin >> y;

	//?xArray?yArray??x?y????2??
	int xArray[11], yArray[11];
	int i, j;
	for (i = 0; i < 11; i++) {
		xArray[i] = x;
		yArray[i] = y;
		x /= 2;
		y /= 2;
	}

	//????????xArray?yArray???????????????
	for (i = 0; i < 11; i++) {
		for (j = 0; j < 11; j++) {
			//??????????????????
			if (xArray[i] == yArray[j]) {
				cout << xArray[i] << endl;
				i = j = 11;
			}
		}
	}
	return 0; //????
}
