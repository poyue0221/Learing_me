#define _CRT_SECURE_NOWARNINGS 1  //消除警告

//包含一个头文件
//std -标准 standard input output 
#include <stdio.h>

#include <string.h>   // strlen的头文件
int a =10;   //全局变量

int main()
{
    int num1 =10; // 局部变量
    int num2 = 20;
    int sum =0;
    // extern int dwg1; // 声明外部符号  

    // scanf("%d,%d",&num1,&num2);  //输入函数
    sum = num1+num2;
    printf("sum=%d\n", sum);
    printf("haha\n"); // 打印函数
    printf("%d\n", strlen(a)); // 打印长度
    printf("%d\n", sizeof(int));   //打印变量/类型所占空间大小,单位是字节
    printf("%d\n",a);


    
    // printf("%d\n",  dwg1 );   
    // printf("%d\n",dwg1);
    return 0;
}

//数据类型
// char 字符类型   1
     // char ch ="A";
    // printf("%c\n,  ch");
//int 整型  4
        //int age =20;

//short 短整型   2
        // short int age =20;
        // long int age = 20;

//long长整型   4/8    C语言标准规定sizeof(long)>=sizeof(int)  是8还是4主要看编译器环境32/64位的

//long long 更长的整型   8

//float 单精度浮点数   4
            // float  f = 22.2f(不加f系统默认为double)

//double  双精度浮点数  8

//%d 打印整形      
        // c   字符
        // f   浮点型
        //lf  double浮点型
        //s  字符串
        // p  以地址形式打印
        // x  打印16进制数字
        // o  ..


//字节
//  bit -比特位   0/1存放的大小
//  byte -字节      一个字节等于8个比特位
//  kb    1kb=1024byte
//  mb    1024
//  gb      1024
// tb       1024
// pb       1024 



// 变量
    // 局部变量: 名字最好不要全局变量名字相同,容易产生bug,否则局部优先.   局部使用
    // 全局变量: 全局使用


//作用域
        // 1,局部变量的作用域是变量所在的局部范围
        // 2,全局变量的作用域是整个工程

//生命周期
    //局部生命周期:生在进局部作用域,结束在出作用域
    //全局作用域:整个工程中

    //常量
        //字面常量
            // 3.14;

        //const 修饰常量
            // const int  a =10;
            // int b = 10
            // int  arr[b] = {0};  报错
            // int arr[a] = {0}；正确   []内需要一个常属性值

        // define  int a =10;   define定义常量


        //枚举常量

        enum  color
        {
            a,
            b,
            c,
        };

        int main()
        {
            enum color  x = a;
            return 0 ;

        }
//************************


// 字符串
    // "abc"
    // "" 空字符串
    //  \t  -  水平制表符
    //  ??+字符  三字母词   
    //   \?   防止多个问号形成三字母词
    //  \\  表示一个普通斜杠

    // \ ddd    ddd表示1到3个八进制的数字   
    //  \32 - 32是2个8进制数字
    //  32作为8进制代表的那个十进制数字,作为ASCII码值,对应的字符
    //  32  -->  10进制26  ->作为ASCI码值代表的字符(右箭头)

    // \xdd    dd表示2个十六进制数字 
    //strlen("c:\test\328\test.c")   13长度

    /*   注释   */  
    int main()
    {
        //数据在计算机上存储的时候,存储的二进制
        //@#$
        //a -97
        //A -65
        //...
        //ASCII 编码
        //ASCII 码值
        //char arr1[] ="abc" //数组
           //"abc"默认存储方式为 'a'  'b'  'c'   '\0'    //'\0' 字符串的结束标志   默认有个\0结束字符
        //    char arr2 [] = {'a','b','c',0}  必须加0否则输出后出现随机值  对应长度也是随机值(strlen(arr2))    0代表字符的值为0
        // '\0'  - 0
    }
        
//************************
// 语句

int main()
{
    int line = 0;
    printf("");
    while (line<200)
    {
        printf("");
    }
    if (line<200)
        printf("");
    return 0;
}
//************************
//数组

int main()
{
    int  arr[10];  //定义一个存放10个整数数字的数组
    int arr1[10] = {0,1,2,3,4,5,6,7,8,9};
    int arr2[10] = {0,1,2,3} ;//不完全初始化,剩下值都默认为0
    char arr3[] = "abcdef"; //根据内容大小指定初始化(7)
    char arr4[5] = {"a",98} ;//98='b'
    printf("%d",sizeof(arr3)); // 7  arr3所占空间大小     //sizeof计算变量,数组,类型的大小单位是字节-操作符
    printf("%d",strlen(arr3));//6    求字符串长度'\0'之前的字符个数    //strlen只能求字符串长度 -库函数使用得引头文件
    int arr6[3][4] = {{1,2,3},{45}}    //1  2  3  0
                                                                // 4  5 0  0
                                                               // 0 0  0  0
    printf("%p",arr1[0]) //首元素地址     输出为0     arr1[0]+1 = 4   
    printf("%p",&arr1[0])           //首元素地址            输出为0     arr1[0]+1 = 4                                      
    printf("%p",&arr1)                //整个数组的地址            输出为0     arr1[0]+1 = 40                       

}

//************************
//操作符

// <<左移 
//>>右移
    //  int a = 1
    //int b = a<<2;    //b = 4    a=1  //移位移的比特位
        右移操作符:
            1,算数右移
                右边丢弃,左边补原符号位
            2,逻辑右移
                右边丢弃,左边补0
        左移 操作符:
                补0完事
   


//   &  按位与  只有两个数的二进制同时为1，结果才为1，否则为0。（负数按补码形式参加按位与运算）
//  丨 按位或   参加运算的两个数只要两个数中的一个为1，结果就为1 
//  ^   按位异或   对应二进制位相同为0   否则为1

// 单目操作符  !  -  +    sizeof       ~a(按位取反)       
// 双目操作符
// 三目操作符   b = a > 5 ? 3 :-3;


//  逗号表达式    1,从左到右运算,最后一个才算值
                                         int a =1
                                         int b =2
                                         int c = (a>b , a = a +b , a , b = a+1)     //c为4

                                2 ,  if(a =b +1 ,c =a/2,d>0)  //最后判断d是否为真
                                3,  while (a = get_(),b = get_l(),a>0)
                                {
                                    /* code */
                                }
                                


//************************
//原码,反玛补马
// 原数用二进制写的 ---  符号位不变,其他按位取反---反码+1

// 只要是整数,内存中存储的都是二进制的补码
// 正数--原码,反码,补码 相同
// 负数--补码


//************************
//计算机  存储数据

//  内存大小由小变大,速度由大到小
    // 寄存器   register
    // 高速缓存
    // 内存
    // 硬盘

//************************

// 常见关键字
// auto(一般局部变量前都都,只是省略了)
break
case
char
const
default
do
double
else
enum
extern
float
goto
if 
int
long
register  (register int a =10  建议把整数a放到寄存器上,是否放看编译器)
return
short
signed  (有符号   unsigned无符号 无符号永远是正数)
sizeof
static  
        // 修饰局部变量
                局部变量的生命周期变长
        //修饰全局变量
                改变了变量的作用域 - 让静态的全局变量只能在自己所在的源文件内部使用,出了源文件就没法使用了
        //修饰函数
                改变了函数链接属性,限制了外部链接属性

struct
switch  
typedef   类型定义(起别名)   typedef unsigned int u_int;  unsigned int 等价于 u_int
union  联合体/共用体
unsigned 无符号
void
volatile   
while 

// *******************
  //#define 定义常量

//#define 宏定义
#define MAX (X,Y)(X>Y?X:Y)
int main()
{
    int a =10;
    int b =20;
    int sum = MAX(a,b);
    return 0;

}

//************************
//指针

// 系统x32  x64
// 32  指32根地址线/数据线
// 每根地址线  有正负两个信号   0/1
// 那么   32 有2^32种信号/地址位置
// 每个地址标识一个字节位

// 指针类型决定了,指针能够访问空间的大小

//指针类型决定了 :指针走一步有多远(指针的步长)
        int *p ; p+1 -->4
        char *p ;  p+1 -->1
        double *p ;  p+1 --> 8
//int* p = NULL //当不知道给指针P赋值什么的时候,就用NULL代替
            //用处,当用完一个指针你可以选择把内存释放掉用NULL代替,如果以后继续用该指针,判断下是否有效if(*p != NULL)

// 指针可判断大小 

// 指针相减等于中间元素个数   printf("%d\n", &arr[1]-&arr[0]);   1   必须指向同一块区域

// 允许指向数组元素的指针与指向数组最后一个元素后面的那个内存位置的指针比较,但是不允许与指向第一个元素之前的那个内存位置的指针比较

//const int * a    *a不能再次赋值了  但地址a 可以改变   /     int * const a   相反
int arr[] = {0};
printf("%p\n",arr);//地址-首元素的地址
printf("%p\n",&arr[0]) //  地址-首元素的地址
printf("%p\n",&arr) // 整个数组的地址

int pa = 10
int** pa = &arr //二级指针    解引用*pa =20
int*** pa = &arr  //三级指针    解引用 **pa =30

int * pa = {&a, &b, &c}

int main()
{
    int a = 10; // 4个字节
    int* p = &a; //取地址  p是变量其类型为int*
    //有一种变量是用来存放地址的-指针变量
    *p = 20  //* - 解引用操作符号
    return 0;

};


// ***********************
// 结构体
//定义一个结构体 ,但无占用内存空间
struct test
{
    char name[20];
}a1,a2,a3;  //a1 a2 a3全局的结构体变量
int main()
{
    //当创建一个结构体变量的时候才会占用空间,生效    局部的结构体变量
    struct test b1=
    {
        "haha"
    };
    //strcpy(b1.name, "dada");  //  数组名本质上 是个地址  所以不能直接改变
    //  结构体传参的时候尽可能是一个指针,这样减少消耗,增加性能     每个函数调用的时候都会出现压栈(先进后出,后进先出)的情况,当不传指针时候,占用内存大,消耗性能


    struct test *z = &b1;
    printf("%s\n",(*z).name);
    //等同于;
    printf("%s\n",z->name);

    return 0 ;

}


//*************************
 
 int main()
 {
     int ch = 0;
    //  ctrt +z     -1    等同于EOF
     while (ch = getchar() != EOF)
     {
         putchar(ch)
     }
     
 }

 //*******************
函数

头文件:
#ifndef __ADD_H_      //如果没有定义过 __ADD_H_  就为真   否则执行 #endif
#define __ADD_H_

//函数声明
int Add(int x, int y);
#endif

C文件:
    #include<stdio.h>
    #incude  "add.h"



// **************

// 递归

int main()
{
    main();
    printf("%s","gaga")
    return 1 ;
}

// 栈区: 局部变量  函数形参  函数调用 
        //1,栈区的默认使用:
                //先使用高地址处的空间
                //再使用低地址处的空间
        //2,数组随着下标的增加,地址是由低到高变化的
// 堆区: 动态开辟的内存 malloc 
// 静态区:全局变量,static修饰的变量


