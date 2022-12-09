#include <iostream>
#include <math.h>
using namespace std;

void reverse() {
    int k;
    if ((k = getchar()) == '\n') return;
    reverse();
    printf("%c", k);
}

int main() {
    reverse();
    return 0;
}
