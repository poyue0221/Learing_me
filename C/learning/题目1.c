#include <stdio.h>

// 找出数组中出现一次的数字.(其他出现两次)
// int dwg1 = 1997;
// void count(int a[], int len)
// {
//     int i =0;
//     int j =1;
//     for ( i = 0; i < len; i++)
//     {
//         while (a[i] != a[j]  )
//         {
//             j++;
//             // printf("haha\n");
//             // printf("%d\n",i);
//             // printf("%d\n",j);
//             // printf"%s\n","gg");
//             // printf("%d\n",j);
//             // printf("%d\n",j); 
//             if (j == len)
//             {
//                 break;    
//             }
//             if (i == j  )
//             {
//                 j++ ;
//                 if (j == len)
//                 {
//                     break;
//                 }
                
//             }
//         };
//         if (j == len)
//         {
//             printf("%d\n",a[i]);
//             // printf("%s\n","结束");
//             break;    
//         } ;
//         j  = 0;
//     }
    
// }

// int main()
// {
//     int a[] ={1,2,3,4,5,1,2,9,5,3,4};
//     int len =sizeof(a)/sizeof(a[0]) ;
//     // printf("%d\n",len);
//     count(a, len); 
//     return 0 ;
// }


//异或满足交换率
// 3^3 = 0
// 5^5 =0
// 0^3 = 3
// 0^5 = 5
// 3^5^3 = 5
// 3^3^5 = 5

int main()
{
    int a[] ={1,2,3,4,5,1,2,9,5,3,4};
    int len =sizeof(a)/sizeof(a[0]) ;
    int b = 0;
    for(int i =0; i<len; i++)
    {
        b = b ^ a[i];
    };
    printf("%d\n",b);
}

// shutdown -s -t 10            -s 关机   -t 读秒关机
// shutdown -a  取消关机