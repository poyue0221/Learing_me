#include <stdio.h>
#include <stdlib.h>
#include<assert.h>

char* my_strcpy(char*a, const char*b)
{
    char * ret =  a; 
    assert(a != NULL);
    assert(b != NULL);
    while (*a++ = *b++)
    {
        ;
    }
    return ret ;
}

int main()
{
    int arr[10] = {0,1,3,3};
    int a = 5;
    int b = -5;
    int c  =  b >>-2 ;
    int* bc =&b;
    int d =1 ;
    char g[] = "abc";
    char h[] = "$$$$$$$$$$";
    // printf("%d\n", &arr[0]);
    
    // printf("%d\n", &arr[1]);
    // printf("%d\n", &arr[1]-&arr[0]);
    printf("%s\n",g);
    my_strcpy(g,h);
    printf("%s\n",g);
    printf("%d\n",~d);
    // printf("%d\n",sizeof(a));
    // printf("%d\n",&a);
    // printf("%d\n",&b);
    // printf("%p\n",bc);
    system("pause");
    return 0 ;
}
