#include <iostream>
#include <math.h>
using namespace std;
int main()                                        // ?????
{
    int a[17] = {0};                              //
    while (cin >> a[0] && a[0] != -1)             // while ???????? -1 ???
    {
        int i = 0, j = 0, k = 0, n = 0;           // ????????
        for (i = 1; i <= 15 ; i++)                // ?????????? 0 ???
        {
            cin >> a[i];
            if (a[i] == 0)
                break;
        }                                         // ????
        for (; j <= i - 1; j++)                   // ???????????
        {
            for (n = 0; n <= i - 1; n++)
                if ((a[n] == 2 * a[j]) || (a[j] == 2 * a[n]))  // ???????????
                    k++;
        }                                         // ??????
        k = k / 2;                                // ????????? 2
        cout << k << endl;                        // ?? k
    }
    return 0;
}                                                 // ?????
